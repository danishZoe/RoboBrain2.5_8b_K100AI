#!/usr/bin/env python3
"""
离线估算 n-gram 自投机（vLLM method="ngram"）在 RoboBrain2.5-8B-NV 上的接受率。

为什么做这个：投机解码能否带来收益，几乎完全由"接受率"决定
（草稿边际成本 ~0，接受率 p<60% 时任何草稿长度都亏）。
本脚本用 vLLM 自己那份 numba 提案器内核 _find_longest_matched_ngram_and_propose_tokens
对**已保存的真实生成文本**做逐位置重放，因此接受率是实测口径而非估计。

不启动服务、不占 GPU、不改服务配置；只读 predictions.jsonl 与 tokenizer.json。

用法（远程）：
  source /opt/dtk/env.sh  (需 set +u 保护，见下方脚本)
  python3 tools/spec_accept_sim.py --out outputs/spec_accept_sim.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

MODEL_DIR = (
    "/public/home/scnjz2s4uc/SothisAI/model/Aihub/"
    "RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV"
)
PRED_GLOB = "/root/private_data/workspace/robo/outputs/bench_public_noneager2/*/predictions.jsonl"
TOKEN_ACCT = "/root/private_data/workspace/robo/outputs/phase5_token_accounting/result.json"
IMAGE_TOKEN_ID = 151655


def load_tokenizer(model_dir):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))


def load_vllm_kernel():
    """取 vLLM 自带的 n-gram 提案器内核，保证仿真语义与线上完全一致。

    优先直接 import；若 torch/DTK 运行时未就绪，则从源码文件里抽出该函数
    单独 exec（该内核只依赖 numpy + numba，不依赖 torch）。
    """
    try:
        from vllm.v1.spec_decode.ngram_proposer import (
            _find_longest_matched_ngram_and_propose_tokens as kernel,
        )

        return kernel
    except Exception as exc:  # noqa: BLE001
        print("direct import failed ({}), falling back to source extraction".format(exc),
              flush=True)

    import ast
    import types

    src_path = "/usr/local/lib/python3.10/dist-packages/vllm/v1/spec_decode/ngram_proposer.py"
    source = open(src_path, encoding="utf-8").read()
    tree = ast.parse(source)
    fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == \
                "_find_longest_matched_ngram_and_propose_tokens":
            fn = node
            break
    if fn is None:
        raise RuntimeError("kernel not found in " + src_path)

    import numpy as np
    from numba import jit

    mod = ast.Module(body=[fn], type_ignores=[])
    ns = {"np": np, "jit": jit}
    exec(compile(mod, src_path, "exec"), ns)  # noqa: S102
    kernel = ns["_find_longest_matched_ngram_and_propose_tokens"]
    print("kernel loaded from source extraction:", kernel, flush=True)
    return kernel


def simulate(kernel, hist_tokens, gen_tokens, min_n, max_n, k, max_model_len):
    """逐位置重放：history = hist_tokens + gen[:pos]，draft 与真实续写比对。

    返回统计 dict。
    """
    steps = 0
    drafted = 0
    accepted = 0
    steps_with_draft = 0
    tokens_per_forward = 0.0
    accept_hist = {}

    hist = np.asarray(hist_tokens, dtype=np.int32)
    gen = np.asarray(gen_tokens, dtype=np.int32)
    n_gen = len(gen)

    for pos in range(n_gen):
        draft = kernel(hist, min_n, max_n, max_model_len, k)
        dlen = len(draft)
        acc = 0
        for j in range(dlen):
            if pos + j < n_gen and int(draft[j]) == int(gen[pos + j]):
                acc += 1
            else:
                break
        steps += 1
        drafted += dlen
        accepted += acc
        accept_hist[acc] = accept_hist.get(acc, 0) + 1
        if dlen > 0:
            steps_with_draft += 1
        tokens_per_forward += 1 + acc
        # 真实续写进入历史（不是 draft，保证重放的是模型真实轨迹）
        hist = np.append(hist, gen[pos])

    return {
        "steps": steps,
        "drafted": drafted,
        "accepted": accepted,
        "steps_with_draft": steps_with_draft,
        "draft_hit_rate": steps_with_draft / steps if steps else 0.0,
        "accept_rate_of_drafted": accepted / drafted if drafted else 0.0,
        "mean_accepted_per_step": accepted / steps if steps else 0.0,
        "tokens_per_forward": tokens_per_forward / steps if steps else 0.0,
        "accept_hist": {str(a): c for a, c in sorted(accept_hist.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/private_data/workspace/robo/outputs/spec_accept_sim.json")
    ap.add_argument("--max-samples", type=int, default=40, help="每个数据集最多取多少条")
    ap.add_argument("--max-gen-tokens", type=int, default=768, help="每条最多重放多少生成 token")
    ap.add_argument("--image-tokens", type=int, default=0,
                    help="在历史中额外注入多少 image token（模拟真实视觉前缀）")
    args = ap.parse_args()

    kernel = load_vllm_kernel()
    tok = load_tokenizer(MODEL_DIR)

    # (min_ngram, max_ngram, num_speculative_tokens)
    configs = [
        (5, 5, 4),   # vLLM method=ngram 的默认窗口 (prompt_lookup_min=max=5)
        (3, 5, 4),
        (3, 8, 4),
        (2, 8, 4),
        (3, 8, 2),
        (3, 8, 8),
        (3, 8, 16),
    ]

    samples = []
    for f in sorted(glob.glob(PRED_GLOB)):
        ds = os.path.basename(os.path.dirname(f))
        got = 0
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                raw = d.get("raw_output")
                if not isinstance(raw, str) or len(raw.strip()) < 40:
                    continue
                samples.append({"dataset": ds, "raw_output": raw,
                                "prompt": d.get("prompt") or "",
                                "sample_id": d.get("sample_id")})
                got += 1
                if got >= args.max_samples:
                    break

    # 压测负载族（gibberish 随机 token 提示）的真实输出（仅 head+tail 片段）
    acct_texts = []
    if os.path.exists(TOKEN_ACCT):
        with open(TOKEN_ACCT, encoding="utf-8") as fh:
            acct = json.load(fh)
        for r in acct.get("results", []):
            txt = (r.get("text_head") or "") + " ... " + (r.get("text_tail") or "")
            acct_texts.append({"dataset": "bench_gibberish:" + r.get("family", "?"),
                               "raw_output": txt, "prompt": "", "sample_id": r.get("family")})

    results = {"config": {"min_ngram_max_ngram_k": configs},
               "samples": len(samples), "results": []}

    def run_group(items, label, use_prompt_hist, image_tokens):
        for cfg in configs:
            min_n, max_n, k = cfg
            agg = {"steps": 0, "drafted": 0, "accepted": 0, "steps_with_draft": 0,
                   "tok_per_forward_sum": 0.0, "n_items": 0}
            per_ds = {}
            for it in items:
                gen = tok.encode(it["raw_output"]).ids
                if not gen:
                    continue
                gen = gen[: args.max_gen_tokens]
                hist = []
                if use_prompt_hist and it.get("prompt"):
                    hist = tok.encode(it["prompt"]).ids
                if image_tokens:
                    hist = [IMAGE_TOKEN_ID] * image_tokens + hist
                st = simulate(kernel, hist, gen, min_n, max_n, k, 32768)
                for key in ("steps", "drafted", "accepted", "steps_with_draft"):
                    agg[key] += st[key]
                agg["tok_per_forward_sum"] += st["tokens_per_forward"] * st["steps"]
                agg["n_items"] += 1
                d = per_ds.setdefault(it["dataset"], {"steps": 0, "drafted": 0, "accepted": 0})
                d["steps"] += st["steps"]
                d["drafted"] += st["drafted"]
                d["accepted"] += st["accepted"]
            if agg["steps"] == 0:
                continue
            rec = {
                "group": label,
                "min_ngram": min_n, "max_ngram": max_n, "num_spec_tokens": k,
                "image_tokens_injected": image_tokens,
                "steps": agg["steps"],
                "draft_hit_rate": agg["steps_with_draft"] / agg["steps"],
                "accept_rate_of_drafted": agg["accepted"] / agg["drafted"] if agg["drafted"] else 0.0,
                "mean_accepted_per_step": agg["accepted"] / agg["steps"],
                "tokens_per_forward": agg["tok_per_forward_sum"] / agg["steps"],
                "per_dataset": {ds: {"steps": v["steps"],
                                      "accept_rate_of_drafted": (v["accepted"] / v["drafted"]) if v["drafted"] else 0.0,
                                      "mean_accepted_per_step": v["accepted"] / v["steps"]}
                                for ds, v in per_ds.items()},
            }
            results["results"].append(rec)
            print(f"[{label}] ngram({min_n},{max_n}) k={k} img={image_tokens} "
                  f"hit={rec['draft_hit_rate']:.3f} acc/draft={rec['accept_rate_of_drafted']:.3f} "
                  f"tok/fwd={rec['tokens_per_forward']:.3f}", flush=True)

    # 组 1：真实任务输出，history 只含生成前缀（最保守）
    run_group(samples, "tasks/gen-only", False, 0)
    # 组 2：真实任务输出，history 含 prompt 文本
    run_group(samples, "tasks/with-prompt", True, 0)
    # 组 3：真实任务输出，history 前缀注入 4096 个 image token（模拟真实视觉前缀）
    run_group(samples, "tasks/with-4096-image", True, 4096)
    # 组 4：压测负载族（gibberish 提示）输出
    if acct_texts:
        run_group(acct_texts, "bench_gibberish", False, 0)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print("written:", args.out)


if __name__ == "__main__":
    sys.exit(main())
