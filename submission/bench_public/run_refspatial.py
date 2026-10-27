#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RefSpatial-Bench 评测（官方 RoboRefer 判分脚本，逐字运行）。

上游官方件
----------
* 数据：``BAAI/RefSpatial-Bench``（HF）→ Location / Placement / Unseen
* 推理入口：``RoboRefer/Evaluation/test_benchmark.py``
* **判分**：``RoboRefer/Evaluation/summarize_acc.py``（本脚本**逐字调用其 CLI**
  与其 ``compute_accuracy``，不改写任何判分逻辑、不改阈值、不改聚合规则）

本脚本只做推理层：按冻结提示词调用本节点的 vLLM 服务，把结果落成官方
``question.json`` 期望的 jsonl 字段，然后交给官方脚本判分。

协议差异（2×2 因子设计，见 reports/15_public_bench_protocol.md）
--------------------------------------------------------------
官方有两处需要适配的地方，且**不是**判分逻辑的修改：

1. **提示词**：官方 ``get_prompt`` 的 "RoboBrain" 分支是
   ``f"{prompt} Please provide its 2D coordinates."``；上游 RoboBrain2.5
   ``inference.py`` 的 pointing 模板在此基础上多一句格式约束。
2. **坐标尺度**：官方 ``summarize_acc.py`` 的 "RoboBrain" 分支用
   ``is_absolute=True``（按**像素**读）。这是 RoboBrain **2.0** 的坐标契约
   （2.0 输出像素坐标）；RoboBrain **2.5** 改用 0~1000 相对坐标
   （上游 ``draw_on_image`` 的 docstring + ``rel_to_abs``）。直接套用会把
   0~1000 当像素 → 越界 → 静默判 0 分。

因此本次跑满 2×2：
  提示词 ∈ {upstream（上游 pointing 模板）, official（RoboRefer 逐字）}
  尺度   ∈ {scaled（推理侧 0~1000→原图像素）, raw（原样不换算）}
主口径 = ``upstream_scaled``（模型原生契约 + 官方判分）。其余三组是单变量
对照，用来量化"提示词差异"与"尺度差异"各自造成的分差。

分母（验收要求 §5.2）
-------------------
官方 ``compute_accuracy`` 在**解析抛异常**时 ``continue`` 跳过该题，其均值
分母 = 实际计分题数。本脚本同时给出：
  * total（总题数）
  * evaluated（官方实际计分题数）
  * parse_failed（解析失败题数）
  * audit_score（全量分母口径，失败计零）
两种结果分别命名，差异写进报告。

默认套件
--------
《精度测试验收要求》§3.1：Location 100 + Placement 100 为默认验收范围；
Unseen 77 是 Location/Placement 的**重叠子集**，不得与前两组相加当 277 道
独立题。因此本脚本默认对 Unseen 单列（``scope=overlapping_subset``），不计入
默认总分。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C          # noqa: E402
import datasets as D        # noqa: E402
import engine as E          # noqa: E402

ROBEREFER = os.path.join(C.WORK_DIR, "third_party/upstream_eval/RoboRefer/Evaluation")
OFFICIAL_MODEL_PREFIX = "RoboBrain2.5-8B-NV"
DEFAULT_SPLITS = ("Location", "Placement", "Unseen")
# 官方默认验收范围（Unseen 为重叠子集，单列）
DEFAULT_SCOPE_SPLITS = ("Location", "Placement")
OVERLAPPING_SPLITS = ("Unseen",)


# --------------------------------------------------------------------------- #
# 提示词构造（两个提示词因子）
# --------------------------------------------------------------------------- #
def build_prompt(q_prompt: str, prompt_mode: str) -> str:
    if prompt_mode == "upstream":
        # 上游 pointing 模板会自己补一个句点，故去掉原文的一个结尾句点避免 ".."
        return C.prompt_upstream(C.strip_one_trailing_period(q_prompt), "pointing")
    if prompt_mode == "official":
        # 官方逐字：不改动原 prompt
        return C.prompt_refspatial_official(q_prompt)
    raise ValueError(prompt_mode)


def variant_dir_name(prompt_mode: str, scale_mode: str) -> str:
    return f"{prompt_mode}_{scale_mode}"


# --------------------------------------------------------------------------- #
# 推理
# --------------------------------------------------------------------------- #
def run_variant(variant: str, prompt_mode: str, scale_mode: str,
                splits: List[str], args) -> Dict[str, Any]:
    out_root = os.path.join(args.out_dir, variant)
    model_dir = os.path.join(out_root, "outputs", f"{OFFICIAL_MODEL_PREFIX}_{variant}")
    os.makedirs(model_dir, exist_ok=True)

    info: Dict[str, Any] = {"variant": variant, "prompt_mode": prompt_mode,
                             "scale_mode": scale_mode, "splits": {}}
    for split in splits:
        samples = D.load_refspatial(split)
        if args.limit:
            samples = samples[:args.limit]
        jobs = []
        for s in samples:
            s = dict(s)
            s["prompt"] = build_prompt(s["fields"]["prompt"], prompt_mode)
            jobs.append(s)

        pred_path = os.path.join(out_root, "predictions", f"{split}.jsonl")
        recs = E.run_jobs(
            jobs, pred_path, args.url, args.model,
            max_tokens=args.max_tokens, concurrency=args.concurrency,
            resume=not args.no_resume,
            parse_fn=lambda t: C.parse_points_1000(t, "pointing"),
            label=f"[refspatial {variant}/{split}]",
        )
        by_id = {r["sample_id"]: r for r in recs}

        # 落成官方 jsonl 字段
        official_jsonl = os.path.join(model_dir, f"{split}.jsonl")
        with open(official_jsonl, "w", encoding="utf-8") as fh:
            for s in samples:
                r = by_id.get(s["sample_id"], {})
                raw = r.get("raw_output", "")
                text = raw
                if scale_mode == "scaled":
                    try:
                        w, h = C.image_size(s["image_paths"][0])
                    except Exception:
                        w = h = 0
                    text = C.scale_1000_to_pixels(raw, w, h, "pointing")
                rec = dict(s["fields"])
                rec["model_id"] = f"{OFFICIAL_MODEL_PREFIX}_{variant}"
                rec["instruction"] = s["prompt"]
                rec["text"] = text
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        info["splits"][split] = {
            "n": len(samples),
            "predictions": pred_path,
            "official_jsonl": official_jsonl,
            "terminal_states": _count_states(recs),
        }
    return info


def _count_states(recs: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in recs:
        out[r["terminal_state"]] = out.get(r["terminal_state"], 0) + 1
    return out


# --------------------------------------------------------------------------- #
# 判分：官方 CLI（逐字）+ 官方 compute_accuracy（直接 import）
# --------------------------------------------------------------------------- #
def score_official_cli(out_dir: str, variant: str, split: str) -> Dict[str, Any]:
    """在官方脚本自己的工作目录约定下，逐字运行其 summarize_acc.py CLI。"""
    run_dir = os.path.join(out_dir, variant)
    link = os.path.join(run_dir, "RefSpatial-Bench")
    if not os.path.exists(link):
        os.symlink(D.REFSPATIAL_DIR, link)
    model_id = f"{OFFICIAL_MODEL_PREFIX}_{variant}"
    env = dict(os.environ)
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + os.pathsep + ROBEREFER
    env["PYTHONWARNINGS"] = "ignore"
    cmd = [sys.executable, os.path.join(ROBEREFER, "summarize_acc.py"),
           "--model_name", model_id, "--task_name", split]
    r = subprocess.run(cmd, cwd=run_dir, env=env, capture_output=True,
                       text=True, timeout=3600)
    stdout = r.stdout.strip()
    stderr = r.stderr.strip()
    res: Dict[str, Any] = {"cmd": " ".join(cmd), "cwd": run_dir,
                            "returncode": r.returncode, "stdout": stdout,
                            "stderr": stderr[-800:]}
    acc = None
    evaluated = None
    total = None
    for line in stdout.splitlines():
        if line.startswith("Accuracy:"):
            for part in line.split(","):
                part = part.strip()
                if part.startswith("Accuracy:"):
                    acc = float(part.split(":")[1].strip())
                elif part.startswith("Evaluated:"):
                    evaluated = int(part.split(":")[1].strip())
                elif part.startswith("Total:"):
                    total = int(part.split(":")[1].strip())
    res["official_accuracy"] = acc
    res["official_evaluated"] = evaluated
    res["official_total"] = total
    return res


def score_official_api(out_dir: str, variant: str, split: str) -> Dict[str, Any]:
    """直接 import 官方 ``compute_accuracy``，并给出全量分母的审计分数。"""
    run_dir = os.path.join(out_dir, variant)
    model_id = f"{OFFICIAL_MODEL_PREFIX}_{variant}"
    jsonl = os.path.join(run_dir, "outputs", model_id, f"{split}.jsonl")
    code = (
        "import sys, json, os, numpy as np\n"
        "sys.path.insert(0, %r)\n"
        "import summarize_acc as S\n"
        "answers = [json.loads(l) for l in open(%r, encoding='utf-8') if l.strip()]\n"
        "S.compute_accuracy(answers, %r, lambda t, w, h: S.text2pts(t, w, h, is_absolute=True))\n"
        "accs = [a.get('accuracy', 0.0) for a in answers]\n"
        "n_eval = len(accs)\n"
        "mean_eval = float(np.mean(accs)) if accs else 0.0\n"
        "audit = float(np.sum(accs) / len(answers)) if answers else 0.0\n"
        "n_empty = int(sum(1 for a in answers if len(S.text2pts(a.get('text',''), 1, 1)) == 0))\n"
        "json.dump({'evaluated': n_eval, 'total': len(answers),\n"
        "           'mean_accuracy': mean_eval, 'audit_score_full_denominator': audit,\n"
        "           'no_point_parsed': n_empty,\n"
        "           'per_sample': [{'question_id': a.get('question_id'),\n"
        "                             'accuracy': a.get('accuracy')} for a in answers]},\n"
        "          open('score_%s.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)\n"
        "print('[official-api] evaluated=%%d total=%%d mean=%%.4f audit=%%.4f' %%\n"
        "      (n_eval, len(answers), mean_eval, audit))\n"
        % (ROBEREFER, jsonl, split, split)
    )
    tmp = os.path.join(run_dir, f"_score_{split}.py")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(code)
    env = dict(os.environ)
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + os.pathsep + ROBEREFER
    env["PYTHONWARNINGS"] = "ignore"
    r = subprocess.run([sys.executable, tmp], cwd=run_dir, env=env,
                       capture_output=True, text=True, timeout=3600)
    out = {"stdout": r.stdout.strip(), "stderr": r.stderr.strip()[-600:],
           "returncode": r.returncode}
    sp = os.path.join(run_dir, f"score_{split}.json")
    if os.path.exists(sp):
        out.update(json.load(open(sp, encoding="utf-8")))
    return out


def score_flagevalmm_norm1k(out_dir: str, variant: str, split: str) -> Dict[str, Any]:
    """交叉核对：FlagEvalMM 官方 ``text2pts_norm1k`` + ``calculate_mask_score``。

    这是与 RoboRefer 口径不同的**另一套官方实现**（0~1000 原生，越界点 clamp
    回图内），用来验证我们的像素换算没有引入偏差。
    """
    run_dir = os.path.join(out_dir, variant)
    model_id = f"{OFFICIAL_MODEL_PREFIX}_{variant}"
    jsonl = os.path.join(run_dir, "outputs", model_id, f"{split}.jsonl")
    code = (
        "import sys, json, os, numpy as np\n"
        "sys.path.insert(0, %r)\n"
        "from flagevalmm.evaluator.point_utils import text2pts_norm1k, calculate_mask_score\n"
        "from PIL import Image\n"
        "answers = [json.loads(l) for l in open(%r, encoding='utf-8') if l.strip()]\n"
        "scores = []\n"
        "for a in answers:\n"
        "    mask = Image.open(os.path.join(%r, a['mask_path']))\n"
        "    pts = text2pts_norm1k(a.get('text',''), width=mask.size[0], height=mask.size[1])\n"
        "    scores.append(calculate_mask_score(pts, mask))\n"
        "out = {'n': len(scores), 'mean_accuracy': float(np.mean(scores)) if scores else 0.0,\n"
        "       'per_sample': [{'question_id': a.get('question_id'), 'accuracy': s}\n"
        "                       for a, s in zip(answers, scores)]}\n"
        "json.dump(out, open('score_norm1k_%s.json','w',encoding='utf-8'),\n"
        "          ensure_ascii=False, indent=2)\n"
        "print('[flagevalmm-norm1k] n=%%d mean=%%.4f' %% (out['n'], out['mean_accuracy']))\n"
        % (os.path.join(C.WORK_DIR, "third_party/upstream_eval/FlagEvalMM"),
           jsonl, D.REFSPATIAL_DIR, split)
    )
    tmp = os.path.join(run_dir, f"_score_norm1k_{split}.py")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(code)
    env = dict(os.environ)
    env["PYTHONPATH"] = (env.get("PYTHONPATH", "") + os.pathsep +
                          os.path.join(C.WORK_DIR,
                                       "third_party/upstream_eval/FlagEvalMM") +
                          os.pathsep + os.path.join(C.WORK_DIR, "envs/pylibs"))
    env["PYTHONWARNINGS"] = "ignore"
    r = subprocess.run([sys.executable, tmp], cwd=run_dir, env=env,
                       capture_output=True, text=True, timeout=3600)
    out = {"stdout": r.stdout.strip(), "stderr": r.stderr.strip()[-600:],
           "returncode": r.returncode}
    sp = os.path.join(run_dir, f"score_norm1k_{split}.json")
    if os.path.exists(sp):
        out.update(json.load(open(sp, encoding="utf-8")))
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=C.DEFAULT_URL)
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--out-dir",
                    default=os.path.join(C.WORK_DIR, "outputs/bench_public/refspatial"))
    ap.add_argument("--variants",
                    default="upstream_scaled,upstream_raw,official_scaled,official_raw",
                    help="逗号分隔的 提示词模式_坐标口径 组合")
    ap.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    ap.add_argument("--limit", type=int, default=0, help="每个 split 的题数上限（0=全部）")
    ap.add_argument("--max-tokens", type=int, default=C.MAX_NEW_TOKENS)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--run-id", default="refspatial-baseline")
    ap.add_argument("--no-guard", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    print("[refspatial] 提示词一致性自检")
    if not C.selfcheck_prompt_parity():
        print("[refspatial] 警告：上游提示词与 robo_infer 不一致，仍继续但请人工确认")

    if not args.no_guard:
        try:
            C.guard_no_official_bench()
        except RuntimeError as exc:
            print("[refspatial] %s" % exc)
            return 3

    if not C.service_alive(args.url):
        print("[refspatial] 服务不可用：%s" % args.url)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    summary: Dict[str, Any] = {
        "run_id": args.run_id,
        "dataset": "RefSpatial-Bench",
        "dataset_source": D.DATASET_SOURCES["RefSpatial-Bench"],
        "protocol": {
            "decode": C.DECODE,
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "images_per_sample": C.IMAGES_PER_SAMPLE,
            "coordinate_contract": "0~1000 relative to ORIGINAL image size",
        },
        "scorers": {
            "primary": "RoboRefer/Evaluation/summarize_acc.py (official CLI, verbatim)",
            "cross_check": "FlagEvalMM flagevalmm/evaluator/point_utils.text2pts_norm1k",
        },
        "default_scope_splits": list(DEFAULT_SCOPE_SPLITS),
        "overlapping_subset_splits": list(OVERLAPPING_SPLITS),
        "variants": {},
    }

    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        prompt_mode, scale_mode = variant.split("_", 1)
        print("\n########## variant=%s ##########" % variant, flush=True)
        vinfo = run_variant(variant, prompt_mode, scale_mode, splits, args)
        for split in splits:
            cli = score_official_cli(args.out_dir, variant, split)
            api = score_official_api(args.out_dir, variant, split)
            fe = score_flagevalmm_norm1k(args.out_dir, variant, split)
            vinfo["splits"][split]["scores"] = {
                "official_cli": cli,
                "official_api": api,
                "flagevalmm_norm1k": fe,
            }
            print("[refspatial] %s/%s  CLI acc=%s (evaluated=%s total=%s) | "
                  "audit=%s | flagevalmm-norm1k=%s" %
                  (variant, split, cli.get("official_accuracy"),
                   cli.get("official_evaluated"), cli.get("official_total"),
                   api.get("audit_score_full_denominator"),
                   fe.get("mean_accuracy")), flush=True)
        summary["variants"][variant] = vinfo

    # 默认总分：只算 Location + Placement（Unseen 是重叠子集，单列）
    for variant, vinfo in summary["variants"].items():
        scope_scores = []
        for split in DEFAULT_SCOPE_SPLITS:
            s = vinfo["splits"].get(split, {})
            cli = s.get("scores", {}).get("official_cli", {})
            if cli.get("official_accuracy") is not None:
                scope_scores.append(cli["official_accuracy"])
        vinfo["default_scope_mean_accuracy"] = (
            round(sum(scope_scores) / len(scope_scores), 4) if scope_scores else None)
        vinfo["default_scope_formula"] = "mean(Location, Placement) 官方 CLI accuracy"

    sp = os.path.join(args.out_dir, "summary.json")
    C.dump_json(sp, summary)
    print("\n[refspatial] 汇总 -> %s" % sp)
    for variant, vinfo in summary["variants"].items():
        print("  %-18s 默认范围均值 = %s" %
              (variant, vinfo.get("default_scope_mean_accuracy")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
