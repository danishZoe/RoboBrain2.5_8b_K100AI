#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CrossPoint-Bench 评测（1000 题：定位/可见性/对应判定/对应指点）。

上游官方件
----------
* 数据：``WangYipu2002/CrossPoint-Bench``（HF）→ ``CrossPoint-Bench.jsonl`` + ``image/``
* 判分（官方代码零改动）：
    - ``CrossPoint-repo/eval/extract_answers.py``（``--coord-format relative_1000``，
      官方原生支持 0-1000 归一化坐标，RoboBrain2.5 输出无需换算语义）；
    - ``CrossPoint-repo/eval/calc_score.py``（点判 mask 命中 / 选项字母判分）。

提示词（冻结，逐字取自官方 eval/eval_opensource.py）
--------------------------------------------------
* Fine-grained Grounding / Correspondence-Pointing：问题 + 官方后缀
  ``" Output the point coordinates in JSON format."``；
* 其余类型：原题。
聊天模板用服务端 RoboBrain2.5 原生模板（与官方 apply_chat_template 同构）。

流程：本脚本产出官方 eval jsonl（含 ``assistant`` 字段）→ 依次调用官方
``extract_answers.py`` / ``calc_score.py``（子进程，代码零改动）。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C          # noqa: E402
import engine as E          # noqa: E402

BENCH_DIR = "/root/private_data/workspace/robo/third_party/benchmarks/CrossPoint-Bench"
REPO_DIR = "/root/private_data/workspace/robo/third_party/benchmarks/CrossPoint-repo"
JSONL = os.path.join(BENCH_DIR, "CrossPoint-Bench.jsonl")
IMAGE_BASE = os.path.join(BENCH_DIR, "image")
POINT_SUFFIX = " Output the point coordinates in JSON format."
POINT_TYPES = ("Fine-grained Grounding", "Correspondence-Pointing")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=C.DEFAULT_URL)
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--run-id", default="baseline-eager-crosspoint")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=C.MAX_NEW_TOKENS)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(JSONL, encoding="utf-8") if l.strip()]
    if args.limit > 0:
        rows = rows[:args.limit]
    os.makedirs(args.out_dir, exist_ok=True)

    jobs: List[Dict[str, Any]] = []
    for i, d in enumerate(rows):
        q = d["question"]
        if d.get("type") in POINT_TYPES:
            q = q + POINT_SUFFIX
        img_paths = [os.path.join(IMAGE_BASE, p) for p in d.get("images", [])]
        jobs.append({
            "sample_id": "CrossPoint/%d" % d.get("idx", i),
            "split": d.get("type", ""),
            "image_paths": img_paths,
            "prompt": q,
            "fields": {"idx": d.get("idx", i), "type": d.get("type"),
                       "level": d.get("level")},
        })

    out_jsonl = os.path.join(args.out_dir, "predictions.jsonl")
    recs = E.run_jobs(jobs, out_jsonl, args.url, args.model,
                      max_tokens=args.max_tokens, concurrency=args.concurrency,
                      run_meta={"run_id": args.run_id, "bench": "crosspoint"})

    # ---- 组装官方 eval jsonl（字段与官方 eval_opensource 输出对齐） ----
    by_idx = {r["sample_id"]: r for r in recs}
    eval_path = os.path.join(args.out_dir, "eval_%s.jsonl" % args.model.replace("/", "_"))
    with open(eval_path, "w", encoding="utf-8") as fh:
        for d in rows:
            sid = "CrossPoint/%d" % d.get("idx", rows.index(d))
            r = by_idx.get(sid, {})
            item = dict(d)
            item["assistant"] = r.get("raw_output") or ""
            item["prompt"] = r.get("prompt") or d["question"]
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ---- 官方判分（代码零改动；relative_1000 为官方原生坐标格式） ----
    # extract_answers 输出目录规则：<out_dir 父目录>/extracted/<out_dir 名>/
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_DIR + os.pathsep + env.get("PYTHONPATH", "")
    r1 = subprocess.run(
        [sys.executable, os.path.join(REPO_DIR, "eval/extract_answers.py"),
         "--file", eval_path, "--image_base", IMAGE_BASE,
         "--coord_format", "relative_1000"],
        env=env, capture_output=True, text=True)
    print("[crosspoint] extract_answers rc=%d" % r1.returncode)
    if r1.returncode != 0:
        print(r1.stdout[-2000:]); print(r1.stderr[-2000:])
        return 1
    extracted_root = os.path.join(os.path.dirname(os.path.abspath(args.out_dir)),
                                  "extracted", os.path.basename(os.path.abspath(args.out_dir)))
    r2 = subprocess.run(
        [sys.executable, os.path.join(REPO_DIR, "eval/calc_score.py"),
         "--extracted_root", extracted_root, "--output_dir", args.out_dir],
        env=env, capture_output=True, text=True)
    print("[crosspoint] calc_score rc=%d" % r2.returncode)
    report = r2.stdout
    with open(os.path.join(args.out_dir, "official_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(report)
    print(report[-2500:])

    C.dump_json(os.path.join(args.out_dir, "summary.json"), {
        "bench": "CrossPoint-Bench",
        "protocol": "official prompts + official extract_answers/calc_score (relative_1000)",
        "n": len(rows),
        "terminal_states": dict(Counter(r.get("terminal_state") for r in recs)),
        "official_report_path": os.path.join(args.out_dir, "official_report.txt"),
        "run_id": args.run_id,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
