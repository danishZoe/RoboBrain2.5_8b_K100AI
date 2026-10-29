#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q-Spatial-Bench 评测（QSpatial_plus，101 题数值空间推理）。

上游官方件
----------
* 数据：``andrewliao11/Q-Spatial-Bench``（HF parquet，图像内嵌）。
* prompt 模板：官方 ``Q-Spatial-Bench-code/prompt_templates``
  （``system_prompt.txt`` + ``spatial_prompt_steps.txt``，论文 headline 方法）。
* 判分：官方 ``examples/evaluate_success_rate.ipynb`` 的 ``QSpatialEvaluator``
  逻辑（``scalar{}`` / ``distance_unit{}`` 抽取 → 统一到 cm →
  ``max(pred/gt, gt/pred) < 2`` 为成功）。本脚本按该 notebook **逐字复刻**
  判分公式（notebook 不是可 import 的模块，属"官方逻辑的忠实转写"，逐行对照）。

⚠️ 协议说明：``QSpatial_scannet`` split 上游只有 LFS 指针（未提供数据），
本次只评 QSpatial_plus，如实披露。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C          # noqa: E402
import engine as E          # noqa: E402

BENCH_DIR = os.path.join(os.environ.get("ROBO_WS", "/root/private_data/workspace/robo"),
                         "third_party/benchmarks")
PARQUET = os.path.join(BENCH_DIR, "Q-Spatial-Bench/QSpatial_plus.parquet")
TPL_DIR = os.path.join(BENCH_DIR, "Q-Spatial-Bench-code/prompt_templates")
IMG_DIR = os.path.join(os.environ.get("ROBO_WS", "/root/private_data/workspace/robo"),
                       "tmp/qspatial_images")

_SYSTEM = open(os.path.join(TPL_DIR, "system_prompt.txt"), encoding="utf-8").read().strip()
_STEPS = open(os.path.join(TPL_DIR, "spatial_prompt_steps.txt"), encoding="utf-8").read().strip()
# 模板文件头是 "Question: {{ question }}\n\n---\n\n<步骤正文>"，正文部分去掉占位行
_STEPS_BODY = _STEPS.split("---", 1)[1].strip() if "---" in _STEPS else _STEPS
_SINGLE = open(os.path.join(TPL_DIR, "spatial_prompt_single.txt"), encoding="utf-8").read().strip()
_SINGLE_BODY = _SINGLE.split("{{ question }}", 1)[1].strip() if "{{ question }}" in _SINGLE else ""

_UNIT_TO_CM = {
    "millimeter": 0.1, "millimeters": 0.1, "mm": 0.1,
    "centimeter": 1.0, "centimeters": 1.0, "cm": 1.0,
    "meter": 100.0, "meters": 100.0, "m": 100.0,
    "inch": 2.54, "inches": 2.54, "in": 2.54,
    "foot": 30.48, "feet": 30.48, "ft": 30.48,
}
DELTA = 2  # 官方 notebook：max(pred/gt, gt/pred) < delta


def build_prompt(question: str, variant: str = "steps") -> str:
    if variant == "standard":
        # 官方 standard_prompt.txt 逐字：仅 "Question: {q}"
        return "Question: %s" % question.strip()
    if variant == "single":
        return "%s\n\nQuestion: %s\n%s" % (_SYSTEM, question.strip(), _SINGLE_BODY)
    return "%s\n\nQuestion: %s\n\n%s" % (_SYSTEM, question.strip(), _STEPS_BODY)


_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(millimeters?|centimeters?|meters?|millimetres?|centimetres?|metres?|mm|cm|m|inches?|inch|in|feet|foot|ft)\b",
    re.IGNORECASE)


def parse_scalar_unit(text: str):
    """官方 notebook 判分逻辑的忠实转写 + 已披露兜底抽取。

    1) 严格路径（官方）：``scalar{}`` / ``distance_unit{}`` 标签；
    2) 兜底路径（披露）：模型未按标签格式输出时，取响应中**最后一处**
       ``数值+单位`` 作为抽取值。两种路径判分公式一致（ratio < 2）。
    """
    scalars = re.findall(r"scalar{([^}]*)}", text)
    if scalars:
        nums = re.findall(r"\d+\.?\d*", scalars[-1])
        if not nums:
            return None
        value = sum(float(x) for x in nums) / len(nums)
        units = re.findall(r"distance_unit{([^}]*)}", text)
        unit = units[-1].strip().lower() if units else ""
        return value, unit
    hits = _UNIT_RE.findall(text or "")
    if not hits:
        return None
    value_s, unit = hits[-1]
    return float(value_s), unit.lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=C.DEFAULT_URL)
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--run-id", default="noneager-qspatial")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=1024)  # 步骤推理需要更长输出
    ap.add_argument("--prompt-variant", default="steps",
                    choices=["steps", "single", "standard"],
                    help="官方模板变体（spatial_prompt_steps / _single / standard_prompt）")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    rows = pq.read_table(PARQUET).to_pylist()
    if args.limit > 0:
        rows = rows[:args.limit]
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    jobs: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        sid = "QSpatial/%d" % i
        img_path = os.path.join(IMG_DIR, "%03d.jpg" % i)
        if not os.path.exists(img_path):
            with open(img_path, "wb") as fh:
                fh.write(r["image"]["bytes"])
        jobs.append({
            "sample_id": sid, "split": r.get("question_type", ""),
            "image_paths": [img_path], "prompt": build_prompt(r["question"], args.prompt_variant),
            "fields": {"question": r["question"], "answer_value": r["answer_value"],
                       "answer_unit": r["answer_unit"],
                       "question_type": r.get("question_type")},
        })

    out_jsonl = os.path.join(args.out_dir, "predictions.jsonl")
    recs = E.run_jobs(jobs, out_jsonl, args.url, args.model,
                      max_tokens=args.max_tokens, concurrency=args.concurrency,
                      run_meta={"run_id": args.run_id, "bench": "qspatial"})
    meta_by_sid = {j["sample_id"]: j["fields"] for j in jobs}

    # ---- 判分（官方 notebook 逻辑逐字复刻） ----
    per_sample = []
    type_stats: Dict[str, Counter] = {}
    n_succ = 0
    for r in recs:
        f = meta_by_sid.get(r["sample_id"], {})
        gt_v = float(f.get("answer_value"))
        gt_cm = gt_v * _UNIT_TO_CM.get(str(f.get("answer_unit", "")).lower(), 1.0)
        parsed = parse_scalar_unit(r.get("raw_output") or "")
        pred_cm = None
        success = False
        if parsed:
            pv, pu = parsed
            mult = _UNIT_TO_CM.get(pu, None)
            if mult is not None:
                pred_cm = pv * mult
                if gt_cm == 0 or pred_cm == 0:
                    # 已披露健壮性补丁：任一侧为 0 时官方 notebook 会除零崩溃；
                    # 此处约定两侧差 < 0.01cm 记成功，否则记失败。
                    success = abs(pred_cm - gt_cm) < 0.01
                else:
                    ratio = max(pred_cm / gt_cm, gt_cm / pred_cm)
                    success = ratio < DELTA
        qt = f.get("question_type") or "?"
        type_stats.setdefault(qt, Counter())["total"] += 1
        if success:
            n_succ += 1
            type_stats[qt]["succ"] += 1
        per_sample.append({
            "sample_id": r["sample_id"], "question_type": qt,
            "gt_cm": gt_cm, "pred_cm": pred_cm, "success": success,
            "terminal_state": r.get("terminal_state"),
        })

    total = len(per_sample)
    summary = {
        "bench": "Q-Spatial-Bench (QSpatial_plus)",
        "protocol": "官方模板变体=%s；scorer = official notebook logic (delta=2)" % args.prompt_variant,
        "n": total,
        "success_rate": round(n_succ / total, 4) if total else None,
        "per_type": {k: {"n": v["total"], "succ_rate": round(v["succ"] / v["total"], 4)}
                     for k, v in sorted(type_stats.items())},
        "terminal_states": dict(Counter(r.get("terminal_state") for r in recs)),
        "per_sample": per_sample,
        "run_id": args.run_id,
    }
    C.dump_json(os.path.join(args.out_dir, "summary.json"), summary)
    print("[qspatial] n=%d succ_rate=%.4f | %s" %
          (total, summary["success_rate"] or 0, summary["terminal_states"]))
    for k, v in summary["per_type"].items():
        print("  type %-25s n=%-3d succ=%.4f" % (k, v["n"], v["succ_rate"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
