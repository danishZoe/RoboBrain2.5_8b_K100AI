#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线重新判分，验证公开基准结果可复现（验收要求 §7.2）。

做法：不重新推理，只从**已落盘的逐题预测**重新调用官方判分脚本，
与第一次的 summary.json 逐指标比对。

用法::

    python3 submission/bench_public/rescore.py \
        --bench-root outputs/bench_public
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C          # noqa: E402

import run_refspatial as RR      # noqa: E402
import run_tracespatial as RT    # noqa: E402
import run_cv_bench as RC        # noqa: E402


def _close(a, b, tol=1e-6) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return a == b


def rescore_refspatial(root: str) -> Dict[str, Any]:
    out_dir = os.path.join(root, "refspatial")
    sp = os.path.join(out_dir, "summary.json")
    if not os.path.exists(sp):
        return {"status": "NOT_RUN", "reason": "缺少 %s" % sp}
    base = json.load(open(sp, encoding="utf-8"))
    rows = []
    for variant, vinfo in base.get("variants", {}).items():
        for split, sinfo in vinfo.get("splits", {}).items():
            if not sinfo.get("official_jsonl"):
                continue
            cli = RR.score_official_cli(out_dir, variant, split)
            api = RR.score_official_api(out_dir, variant, split)
            fe = RR.score_flagevalmm_norm1k(out_dir, variant, split)
            o = sinfo.get("scores", {})
            rows.append({
                "dataset": "RefSpatial-Bench", "variant": variant, "split": split,
                "metric": "official_cli_accuracy",
                "first": o.get("official_cli", {}).get("official_accuracy"),
                "rescored": cli.get("official_accuracy"),
                "match": _close(o.get("official_cli", {}).get("official_accuracy"),
                                cli.get("official_accuracy")),
                "first_evaluated": o.get("official_cli", {}).get("official_evaluated"),
                "rescored_evaluated": cli.get("official_evaluated"),
                "first_total": o.get("official_cli", {}).get("official_total"),
                "rescored_total": cli.get("official_total"),
                "audit_score_full_denominator": api.get("audit_score_full_denominator"),
                "flagevalmm_norm1k_mean": fe.get("mean_accuracy"),
            })
    return {"status": "OK", "rows": rows,
            "all_match": all(r["match"] for r in rows) if rows else None}


def rescore_tracespatial(root: str) -> Dict[str, Any]:
    out_dir = os.path.join(root, "tracespatial")
    sp = os.path.join(out_dir, "summary.json")
    if not os.path.exists(sp):
        return {"status": "NOT_RUN", "reason": "缺少 %s" % sp}
    base = json.load(open(sp, encoding="utf-8"))
    rows = []
    for variant, vinfo in base.get("variants", {}).items():
        for task, tinfo in vinfo.get("tasks", {}).items():
            if not tinfo.get("official_jsonl"):
                continue
            sc = RT.score_official_cli(out_dir, variant, task)
            o = tinfo.get("scores", {})
            for key in ("metrics_2d", "metrics_3d", "start_success_rate",
                        "end_success_rate", "overall_success_rate",
                        "no_collision_rate", "start_in_mask", "end_in_bbox"):
                rows.append({
                    "dataset": "TraceSpatial-Bench", "variant": variant,
                    "split": task, "metric": key,
                    "first": o.get(key), "rescored": sc.get(key),
                    "match": o.get(key) == sc.get(key),
                })
    return {"status": "OK", "rows": rows,
            "all_match": all(r["match"] for r in rows) if rows else None}


def rescore_cvbench(root: str) -> Dict[str, Any]:
    out_dir = os.path.join(root, "cvbench")
    sp = os.path.join(out_dir, "summary.json")
    if not os.path.exists(sp):
        return {"status": "NOT_RUN", "reason": "缺少 %s" % sp}
    base = json.load(open(sp, encoding="utf-8"))
    sc = RC.score_official(out_dir,
                           os.path.join(out_dir, "annotations.json"),
                           os.path.join(out_dir, "predictions_official.json"))
    first = base.get("scores", {}).get("official_result", {})
    second = sc.get("official_result", {})
    rows = []
    for key in ("accuracy_2d_ade", "accuracy_2d_coco", "accuracy_3d_omni",
                "accuracy_2d", "accuracy_3d", "accuracy"):
        rows.append({"dataset": "CV-Bench", "variant": "-", "split": "all",
                     "metric": key, "first": first.get(key),
                     "rescored": second.get(key),
                     "match": _close(first.get(key), second.get(key), 1e-9)})
    return {"status": "OK", "rows": rows,
            "all_match": all(r["match"] for r in rows) if rows else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root",
                    default=os.path.join(C.WORK_DIR, "outputs/bench_public"))
    args = ap.parse_args()

    report = {
        "refspatial": rescore_refspatial(args.bench_root),
        "tracespatial": rescore_tracespatial(args.bench_root),
        "cvbench": rescore_cvbench(args.bench_root),
    }
    n_rows = 0
    n_bad = 0
    for k, v in report.items():
        for r in v.get("rows", []):
            n_rows += 1
            if not r["match"]:
                n_bad += 1
    report["summary"] = {"compared_metrics": n_rows, "mismatched": n_bad,
                         "reproducible": n_bad == 0 and n_rows > 0}
    out = os.path.join(args.bench_root, "rescore_check.json")
    C.dump_json(out, report)
    print("[rescore] -> %s" % out)
    print("[rescore] 比较指标 %d，不一致 %d" % (n_rows, n_bad))
    for r in report["refspatial"].get("rows", []):
        if not r["match"]:
            print("  MISMATCH", r)
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
