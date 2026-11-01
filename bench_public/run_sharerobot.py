#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ShareRobot-Bench 评测（affordance 200 题框定位 + trajectory 200 题轨迹）。

上游官方件
----------
* 数据：``BAAI/ShareRobot-Bench``（HF 仓库，affordance/ + trajectory/ 各 200 题）。
* **判分**：FlagEvalMM **PR#84** 新增的 ``tasks/share_robot/{affordance,trajectory}/evaluate.py``
  （已按 diff 原样 apply 到本地 FlagEvalMM 工作副本，代码零改动，本脚本 import 调用）。
* prompt（冻结，逐字取自 PR#84）：
  - affordance：原题 + post_prompt ``"Only return the box in format: [x1, y1, x2, y2] with no other output."``；
  - trajectory：官方 process.py 把原题里的
    "Please predict up to 10 key trajectory points to complete the task."
    替换为带格式说明的长句（逐字复刻该替换）。

坐标契约（披露）
----------------
RoboBrain2.5 原生输出 0-1000 归一化坐标；官方判分按像素坐标处理
（affordance 直接 IoU；trajectory 对 >1 的值除以图像宽高）。
本脚本在「结果适配层」把模型输出的坐标 0-1000 → 像素（×w/1000, ×h/1000），
原始输出逐题保留。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C          # noqa: E402
import engine as E          # noqa: E402

BENCH_DIR = "/root/private_data/workspace/robo/third_party/benchmarks/ShareRobot-Bench/repo"
FLAGEVAL = "/root/private_data/workspace/robo/third_party/upstream_eval/FlagEvalMM"

TRAJ_OLD = "Please predict up to 10 key trajectory points to complete the task."
TRAJ_NEW = ("Please predict up to 10 key trajectory points to complete the task. "
            "Your answer should be formatted as a list of tuples, i.e. "
            "[[x1, y1], [x2, y2], ...], where each tuple contains the x and y "
            "coordinates of a point.")
AFF_SUFFIX = "Only return the box in format: [x1, y1, x2, y2] with no other output."


def _load_eval(split: str):
    spec = importlib.util.spec_from_file_location(
        "sr_" + split, os.path.join(FLAGEVAL, "tasks/share_robot/%s/evaluate.py" % split))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# 与官方 PR84 extract_points 接受的格式一致：() 与 [] 两种括号
_PAIR_RE = re.compile(r"[\(\[]\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*[\)\]]")


def scale_1000_text_to_pixels(text: str, w: int, h: int) -> str:
    """结果适配层：按**样本内检测到的坐标尺度**统一换算到像素。

    尺度判定（冻结规则，逐题可审计）：
    * 样本内坐标最大值 <= 1.5      → 视为 0-1 归一化，不换算；
    * <= 1000.5                    → 视为 0-1000（RoboBrain2.5 原生），×w/1000、×h/1000；
    * > 1000.5                     → 视为像素坐标，不换算。

    历史缺陷修复（2026-10-29 排查）：旧正则只匹配方括号，模型输出的
    圆括号元组 "(803, 325)" 未被换算，被官方 extract 当作像素再除一次
    图像宽高（如 803/224=3.6），导致 ShareRobot-T DFD 从真实的 ~0.18
    虚高到 1.76。
    """
    nums = [float(m.group(1)) for m in _PAIR_RE.finditer(text or "")]
    nums += [float(m.group(2)) for m in _PAIR_RE.finditer(text or "")]
    if not nums:
        return text
    mx = max(nums)
    if mx <= 1.5 or mx > 1000.5:
        return text
    def rep(m):
        x = round(float(m.group(1)) * w / 1000.0)
        y = round(float(m.group(2)) * h / 1000.0)
        return "[%d, %d]" % (x, y)
    return _PAIR_RE.sub(rep, text)


def run_split(split: str, args) -> Dict[str, Any]:
    data = json.load(open(os.path.join(BENCH_DIR, split, "test.json"), encoding="utf-8"))
    if args.limit > 0:
        data = data[:args.limit]
    out_dir = os.path.join(args.out_dir, split)
    os.makedirs(out_dir, exist_ok=True)

    jobs: List[Dict[str, Any]] = []
    metas: Dict[str, Dict[str, Any]] = {}
    for i, item in enumerate(data):
        sid = "ShareRobot-%s/%s" % (split, item["id"])
        conv = {c["from"]: c["value"] for c in item["conversations"]}
        q = conv["human"].replace("<image>", "").strip()
        gt = conv["gpt"].strip()
        if split == "trajectory":
            q = q.replace(TRAJ_OLD, TRAJ_NEW)
        elif split == "affordance":
            q = q + " " + AFF_SUFFIX
        img_rel = item["image"]
        img_path = os.path.join(BENCH_DIR, split,
                                img_rel[0] if isinstance(img_rel, list) else img_rel)
        w, h = C.image_size(img_path)
        jobs.append({
            "sample_id": sid, "split": split,
            "image_paths": [img_path], "prompt": q,
            "fields": {"question_id": str(item["id"]), "gt": gt,
                       "width": w, "height": h},
        })
        metas[sid] = {"question_id": str(item["id"]), "gt": gt, "w": w, "h": h}

    out_jsonl = os.path.join(out_dir, "predictions.jsonl")
    recs = E.run_jobs(jobs, out_jsonl, args.url, args.model,
                      max_tokens=args.max_tokens, concurrency=args.concurrency,
                      run_meta={"run_id": args.run_id + "-" + split,
                                "bench": "sharerobot-" + split})

    # ---- 结果适配层 + 官方判分（PR#84 evaluate.py，零改动） ----
    mod = _load_eval(split)
    annotations: Dict[str, Any] = {}
    predictions: List[Dict[str, Any]] = []
    for r in recs:
        m = metas.get(r["sample_id"], {})
        qid = m.get("question_id")
        if not qid:
            continue
        raw = r.get("raw_output") or ""
        adapted = scale_1000_text_to_pixels(raw, m.get("w", 1000), m.get("h", 1000))
        annotations[qid] = {"answer": m["gt"], "width": m["w"], "height": m["h"]}
        predictions.append({"question_id": qid, "answer": adapted})

    metrics = mod.get_result(annotations, predictions)
    summary = {
        "bench": "ShareRobot-Bench/" + split,
        "protocol": "FlagEvalMM PR#84 official evaluate.py; prompts verbatim; 0-1000->pixel adaptation",
        "n": len(predictions),
        "metrics": metrics,
        "terminal_states": dict(Counter(r.get("terminal_state") for r in recs)),
        "run_id": args.run_id + "-" + split,
    }
    C.dump_json(os.path.join(out_dir, "summary.json"), summary)
    print("[sharerobot-%s] n=%d metrics=%s" % (split, len(predictions), metrics))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=C.DEFAULT_URL)
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--run-id", default="noneager-sharerobot")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=C.MAX_NEW_TOKENS)
    ap.add_argument("--splits", nargs="+", default=["trajectory", "affordance"])
    args = ap.parse_args()
    for split in args.splits:
        run_split(split, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
