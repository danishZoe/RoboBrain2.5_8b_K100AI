#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoboSpatial-Home 评测（350 题：configuration/compatibility VQA + context 指点）。

上游官方件
----------
* 数据：``chanhee-luke/RoboSpatial-Home``（HF，3 个 parquet，图像/mask 内嵌）。
  本项目从本地 parquet 读取并物化图像与 mask（输入来源适配，与 CV-Bench
  走本地 parquet 同一披露口径）。
* **判分**：``RoboSpatial-Eval/evaluation.py`` 的
  ``eval_pregenerated_results``（官方"预生成结果"评测路径，**代码零改动**）。
  匹配键 = (question, img)；yes/no 题看输出前缀；context 题取前
  ``num_points_to_match=2`` 个点，任一落入 GT mask 即正确。

坐标契约（披露）
----------------
官方 context 判分接收 **0-1 归一化** 坐标；RoboBrain2.5 原生输出 0-1000。
本脚本在「结果适配层」把模型原始输出的坐标除以 1000 后交给官方判分，
原始输出全量保留在逐题记录里（与 RefSpatial 的 0-1000→像素换算同一性质）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C          # noqa: E402
import engine as E          # noqa: E402

_WS = os.environ.get("ROBO_WS", "/root/private_data/workspace/robo")
BENCH_DIR = os.path.join(_WS, "third_party/benchmarks")
HOME_DIR = os.path.join(BENCH_DIR, "RoboSpatial-Home")
EVAL_REPO = os.path.join(BENCH_DIR, "RoboSpatial-Eval")
DATA_DIR = os.path.join(_WS, "tmp/robospatial_data")
SPLITS = ("configuration", "compatibility", "context")


def load_split(split: str) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq
    t = pq.read_table(os.path.join(HOME_DIR, "%s-00000-of-00001.parquet" % split))
    return t.to_pylist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=C.DEFAULT_URL)
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--run-id", default="noneager-robospatial")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=C.MAX_NEW_TOKENS)
    ap.add_argument("--with-depth", action="store_true",
                    help="诊断模式：把官方 depth_image 作为第二路图像输入"
                         "（官方 harness 的模型接口本就提供 depth_path；"
                         " 诊断用途，主口径仍为纯 RGB）")
    args = ap.parse_args()

    sys.path.insert(0, EVAL_REPO)
    import evaluation as RS_EVAL  # 官方判分模块，零改动

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "masks"), exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    jobs: List[Dict[str, Any]] = []
    gt_entries: List[Dict[str, Any]] = []
    gt_sids: List[str] = []
    for split in SPLITS:
        rows = load_split(split)
        if args.limit > 0:
            rows = rows[:args.limit]
        for idx, r in enumerate(rows):
            sid = "RoboSpatial/%s_%d" % (split, idx)
            img_rel = os.path.join("images", "img_%s_%d.png" % (split, idx))
            img_abs = os.path.join(DATA_DIR, img_rel)
            if not os.path.exists(img_abs):
                with open(img_abs, "wb") as fh:
                    fh.write(r["img"]["bytes"])
            mask_rel = None
            if r.get("mask") is not None:
                mask_rel = os.path.join("masks", "mask_%s_%d.png" % (split, idx))
                mask_abs = os.path.join(DATA_DIR, mask_rel)
                if not os.path.exists(mask_abs):
                    with open(mask_abs, "wb") as fh:
                        fh.write(r["mask"]["bytes"])
            question = r["question"]
            imgs = [img_abs]
            if args.with_depth and r.get("depth_image") is not None:
                depth_rel = os.path.join("images", "depth_%s_%d.png" % (split, idx))
                depth_abs = os.path.join(DATA_DIR, depth_rel)
                if not os.path.exists(depth_abs):
                    with open(depth_abs, "wb") as fh:
                        fh.write(r["depth_image"]["bytes"])
                imgs.append(depth_abs)
            gt_entries.append({"category": split, "question": question,
                               "answer": r["answer"], "img": img_rel,
                               "mask": mask_rel})
            gt_sids.append(sid)
            jobs.append({
                "sample_id": sid, "split": split,
                "image_paths": imgs, "prompt": question,
                "fields": {"answer_gt": r["answer"]},
            })

    out_jsonl = os.path.join(args.out_dir, "predictions.jsonl")
    recs = E.run_jobs(jobs, out_jsonl, args.url, args.model,
                      max_tokens=args.max_tokens, concurrency=args.concurrency,
                      run_meta={"run_id": args.run_id, "bench": "robospatial"})

    # ---- 结果适配层：原始输出 -> 官方预生成结果格式 ----
    by_sid = {r["sample_id"]: r for r in recs}
    results_data: List[Dict[str, Any]] = []
    for gt, sid in zip(gt_entries, gt_sids):
        rec = by_sid.get(sid, {})
        raw = rec.get("raw_output") or ""
        adapted = raw
        if gt["category"] == "context":
            # 坐标契约：0-1000 -> 0-1（仅适配层，原始输出保留在逐题记录）
            import re
            def _sc(m):
                return str(round(int(m.group(1)) / 1000.0, 4))
            adapted = re.sub(r"\b(\d{1,3})\b", _sc, raw)
        results_data.append({"img": gt["img"], "category": gt["category"],
                             "question": gt["question"], "answer": adapted})

    stats = RS_EVAL.eval_pregenerated_results(
        gt_entries, results_data, DATA_DIR, num_points_to_match=2)

    C.dump_json(os.path.join(args.out_dir, "summary.json"), {
        "bench": "RoboSpatial-Home",
        "protocol": "official eval_pregenerated_results; 0-1000->0-1 coord adaptation"
                    + ("; WITH-DEPTH diagnostic (depth as 2nd image)" if args.with_depth else ""),
        "n": len(gt_entries),
        "official_stats": stats,
        "terminal_states": dict(Counter(r.get("terminal_state") for r in recs)),
        "run_id": args.run_id,
    })
    print("[robospatial] official stats: %s" % json.dumps(stats, ensure_ascii=False)[:800])
    print("[robospatial] terminal: %s" %
          dict(Counter(r.get("terminal_state") for r in recs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
