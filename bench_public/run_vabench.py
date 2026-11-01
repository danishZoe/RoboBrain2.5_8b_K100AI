#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VABench-V 评测（300 题视频视觉轨迹；指标 DFD/HD/RMSE，越低越好）。

上游官方件
----------
* 数据：``Embodied1/vabench-v``（HF parquet，图像内嵌；answer 为像素轨迹点）。
* 论文出处：RoboBrain 2.0 技术报告（arXiv 2507.02029）的 VABench 评测协议。

⚠️ 判分适配披露（重要）
----------------------
VABench **官方评测代码未公开发布**（GitHub/HF 均未找到；FlagEvalMM 也无此任务）。
本脚本判分**沿用 FlagEvalMM PR#84 ShareRobot 轨迹任务的同款几何指标与抽取逻辑**
（``tasks/share_robot/trajectory/evaluate.py``，零改动 import；该口径同时是
RoboBrain2.0 技术报告对 ShareRobot-Trajectory 的口径），属于"按最接近官方公开
实现复刻"，抽取与指标逐题落盘可审计。若后续拿到 VABench 官方判分器，应以官方
版本重判并对比。

坐标契约：RoboBrain2.5 原生 0-1000；本脚本在结果适配层换算到像素后交判分。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C          # noqa: E402
import engine as E          # noqa: E402

BENCH_DIR = "/root/private_data/workspace/robo/third_party/benchmarks"
PARQUET = os.path.join(BENCH_DIR, "VABench-V/vabench-v.parquet")
FLAGEVAL = "/root/private_data/workspace/robo/third_party/upstream_eval/FlagEvalMM"
TRAJ_EVAL = os.path.join(FLAGEVAL, "tasks/share_robot/trajectory/evaluate.py")
IMG_DIR = "/root/private_data/workspace/robo/tmp/vabench_images"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=C.DEFAULT_URL)
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--run-id", default="noneager-vabench")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=C.MAX_NEW_TOKENS)
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("sr_trajectory", TRAJ_EVAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    import pyarrow.parquet as pq
    rows = pq.read_table(PARQUET).to_pylist()
    if args.limit > 0:
        rows = rows[:args.limit]
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    jobs: List[Dict[str, Any]] = []
    metas: Dict[str, Dict[str, Any]] = {}
    for i, r in enumerate(rows):
        sid = "VABench/%s" % r["id"]
        img_path = os.path.join(IMG_DIR, "%03d.jpg" % i)
        if not os.path.exists(img_path):
            with open(img_path, "wb") as fh:
                fh.write(r["image"]["bytes"])
        w, h = C.image_size(img_path)
        jobs.append({
            "sample_id": sid, "split": str(r.get("data_type", "")),
            "image_paths": [img_path], "prompt": r["problem"],
            "fields": {"question_id": str(r["id"]), "gt": json.dumps(r["answer"]),
                       "width": w, "height": h},
        })
        metas[sid] = {"question_id": str(r["id"]), "gt": json.dumps(r["answer"]),
                      "w": w, "h": h}

    out_jsonl = os.path.join(args.out_dir, "predictions.jsonl")
    recs = E.run_jobs(jobs, out_jsonl, args.url, args.model,
                      max_tokens=args.max_tokens, concurrency=args.concurrency,
                      run_meta={"run_id": args.run_id, "bench": "vabench"})

    # ---- 结果适配层（按样本内检测尺度：0-1 / 0-1000 / 像素，见下）+ 判分 ----
    import re
    # 与官方 extract_points 接受的格式一致：() 与 [] 两种括号
    # （2026-10-29 修复：旧正则只认 []，圆括号元组未被换算导致双重归一化、
    #   DFD 从真实 ~0.20 虚高到 0.82）
    _PAIR_RE = re.compile(r"[\(\[]\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*[\)\]]")

    def to_pixels(text: str, w: int, h: int) -> str:
        """尺度判定（冻结规则）：max<=1.5 视为 0-1 不动；<=1000.5 视为 0-1000
        换算到像素；更大视为像素不动。"""
        vals = [float(g) for m in _PAIR_RE.finditer(text or "") for g in m.groups()]
        if not vals:
            return text
        mx = max(vals)
        if mx <= 1.5 or mx > 1000.5:
            return text
        def rep(m):
            return "[%d, %d]" % (round(float(m.group(1)) * w / 1000.0),
                                 round(float(m.group(2)) * h / 1000.0))
        return _PAIR_RE.sub(rep, text)

    annotations: Dict[str, Any] = {}
    predictions: List[Dict[str, Any]] = []
    for r in recs:
        m = metas.get(r["sample_id"], {})
        qid = m.get("question_id")
        if not qid:
            continue
        raw = r.get("raw_output") or ""
        adapted = to_pixels(raw, m.get("w", 1000), m.get("h", 1000))
        annotations[qid] = {"answer": m["gt"], "width": m["w"], "height": m["h"]}
        predictions.append({"question_id": qid, "answer": adapted})

    metrics = mod.get_result(annotations, predictions)
    summary = {
        "bench": "VABench-V",
        "protocol": "problem verbatim; scoring = FlagEvalMM PR84 trajectory evaluate.py (disclosed substitution for missing official scorer)",
        "n": len(predictions),
        "metrics": metrics,
        "terminal_states": dict(Counter(r.get("terminal_state") for r in recs)),
        "run_id": args.run_id,
    }
    C.dump_json(os.path.join(args.out_dir, "summary.json"), summary)
    print("[vabench] n=%d metrics=%s" % (len(predictions), metrics))
    print("[vabench] terminal: %s" % summary["terminal_states"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
