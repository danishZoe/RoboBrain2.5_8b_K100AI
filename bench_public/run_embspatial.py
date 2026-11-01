#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EmbSpatial-Bench 评测（3640 道多选题，generation 方法）。

上游官方件
----------
* 数据：``Phineas476/EmbSpatial-Bench``（HF）→ ``embspatial_bench.json``
  （论文官方仓库 mengfeidu/EmbSpatial-Bench 给出的下载链接；图像 base64 内嵌）。
* 论文：arXiv 2406.05756 / ACL 2024 short。

⚠️ 协议差异披露（验收要求 §4.1）
-------------------------------
官方 GitHub 仓库**没有发布评测代码**（README 只有数据说明；评测方法论文描述为
generation 或 likelihood 两种，主指标 Accuracy）。因此：

* prompt 采用与 CV-Bench 主口径一致的 FlagEvalMM 风格 MC 格式
  （选项 + "Answer with the option's letter from the given choices directly."），
  该格式已在本项目验证适用于 RoboBrain2.5；
* 判分为自建：抽取首个 A-D 字母（容忍 ``A.``/``(A)`` 形态），与 ``answer`` 索引比对；
* 只实现 generation 方法；likelihood 方法不在本次范围。

逐题落盘：prompt / 原始输出 / 抽取字母 / 正误 / relation 分组。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C          # noqa: E402
import engine as E          # noqa: E402

BENCH_DIR = "/root/private_data/workspace/robo/third_party/benchmarks"
DATA_JSON = os.path.join(BENCH_DIR, "embspatial_bench.json")
IMG_DIR = "/root/private_data/workspace/robo/tmp/embspatial_images"

MC_SUFFIX = "\nAnswer with the option's letter from the given choices directly."
_LETTER_RE = re.compile(r"\(?([A-D])\)?")


def build_prompt(question: str, options: List[str]) -> str:
    lines = [question.strip(), ""]
    for i, opt in enumerate(options):
        lines.append("%s. %s" % (chr(65 + i), opt))
    return "\n".join(lines) + MC_SUFFIX


def extract_letter(text: str) -> str:
    t = (text or "").strip()
    m = _LETTER_RE.match(t)
    if m:
        return m.group(1)
    m = _LETTER_RE.search(t[:40])
    return m.group(1) if m else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=C.DEFAULT_URL)
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--run-id", default="baseline-eager-embspatial")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=C.MAX_NEW_TOKENS)
    args = ap.parse_args()

    data = json.load(open(DATA_JSON, encoding="utf-8"))
    if args.limit > 0:
        data = data[:args.limit]
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    jobs: List[Dict[str, Any]] = []
    metas: Dict[str, Dict[str, Any]] = {}
    for i, d in enumerate(data):
        sid = d["question_id"]
        img_path = os.path.join(IMG_DIR, "%s.jpg" % sid)
        if not os.path.exists(img_path):
            with open(img_path, "wb") as fh:
                fh.write(base64.b64decode(d["image"]))
        prompt = build_prompt(d["question"], d["answer_options"])
        jobs.append({
            "sample_id": sid,
            "split": d.get("relation", ""),
            "image_paths": [img_path],
            "prompt": prompt,
            "fields": {"question": d["question"], "options": d["answer_options"],
                       "answer_index": d["answer"], "relation": d.get("relation"),
                       "data_source": d.get("data_source")},
        })
        metas[sid] = d

    out_jsonl = os.path.join(args.out_dir, "predictions.jsonl")
    recs = E.run_jobs(jobs, out_jsonl, args.url, args.model,
                      max_tokens=args.max_tokens, concurrency=args.concurrency,
                      run_meta={"run_id": args.run_id, "bench": "embspatial"},
                      parse_fn=extract_letter)

    # ---- 判分（自建，协议见文件头披露） ----
    per_sample = []
    rel_correct: Dict[str, Counter] = {}
    src_correct: Dict[str, Counter] = {}
    for r in recs:
        sid = r["sample_id"]
        meta = metas.get(sid, {})
        gt_idx = meta.get("answer")
        pred = (r.get("parsed") or "")
        pred_ok = bool(pred) and pred in "ABCD"
        correct = bool(pred_ok) and (ord(pred) - 65) == gt_idx
        rel = meta.get("relation") or "?"
        src = meta.get("data_source") or "?"
        rel_correct.setdefault(rel, Counter())["total"] += 1
        src_correct.setdefault(src, Counter())["total"] += 1
        if correct:
            rel_correct[rel]["correct"] += 1
            src_correct[src]["correct"] += 1
        per_sample.append({
            "question_id": sid, "relation": rel, "data_source": src,
            "gt_index": gt_idx, "pred_letter": pred, "correct": correct,
            "terminal_state": r.get("terminal_state"),
        })

    total = len(per_sample)
    n_correct = sum(1 for p in per_sample if p["correct"])
    summary = {
        "bench": "EmbSpatial-Bench",
        "protocol": "generation-method MC; FlagEvalMM-style prompt; self-built letter extraction",
        "n": total,
        "accuracy": round(n_correct / total, 4) if total else None,
        "n_correct": n_correct,
        "terminal_states": dict(Counter(r.get("terminal_state") for r in recs)),
        "per_relation": {k: {"n": v["total"], "acc": round(v["correct"] / v["total"], 4)}
                         for k, v in sorted(rel_correct.items())},
        "per_source": {k: {"n": v["total"], "acc": round(v["correct"] / v["total"], 4)}
                       for k, v in sorted(src_correct.items())},
        "per_sample": per_sample,
        "run_id": args.run_id,
    }
    C.dump_json(os.path.join(args.out_dir, "summary.json"), summary)
    print("[embspatial] n=%d acc=%.4f | %s" %
          (total, summary["accuracy"] or 0, summary["terminal_states"]))
    for k, v in summary["per_relation"].items():
        print("  relation %-8s n=%-4d acc=%.4f" % (k, v["n"], v["acc"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
