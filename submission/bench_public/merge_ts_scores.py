#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把分片判分的结果合并回 TraceSpatial 的 summary.json。

分片调度器（``score_3d_chunked.py``）把每个 (变体, 任务) 的官方判分结果写进
``<bench-root>/tracespatial/<variant>/scores_3dchunk_<task>.json``；
本脚本把这些结果回填到 ``summary.json`` 的
``variants.<variant>.tasks.<task>.scores``，让汇总表 / 回归对比脚本能统一读取。

只搬运结果，**不重算任何指标**。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def load(p: str):
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root",
                    default=os.path.join(C.WORK_DIR, "outputs/bench_public"))
    ap.add_argument("--alt-3d-root", default=None,
                    help="备选 3D 判分产物根目录（形如 outputs/bench_public_3d96/"
                         "<bench_root 名>/tracespatial/<variant>/scores_3dchunk_3D.json）。"
                         "当主目录缺少 3D 产物时用它兜底（分母会如实标注排除的样本数）")
    args = ap.parse_args()

    ts = os.path.join(args.bench_root, "tracespatial")
    sp = os.path.join(ts, "summary.json")
    if not os.path.exists(sp):
        print("[merge_ts] 缺 %s" % sp)
        return 1
    summary: Dict[str, Any] = json.load(open(sp, encoding="utf-8"))

    alt_ts = None
    if args.alt_3d_root:
        alt_ts = os.path.join(args.alt_3d_root,
                              os.path.basename(os.path.normpath(args.bench_root)),
                              "tracespatial")

    touched = 0
    for variant, vi in summary.get("variants", {}).items():
        for task, ti in vi.get("tasks", {}).items():
            cpath = os.path.join(ts, variant, "scores_3dchunk_%s.json" % task)
            cd = load(cpath)
            src_note = ("score_3d_chunked.py（官方 summarize_acc.py 逐字调用，"
                        "逐样本分片并行；分片合并已用整集单进程对拍验证）")
            if not cd and alt_ts and task == "3D":
                cpath = os.path.join(alt_ts, variant, "scores_3dchunk_3D.json")
                cd = load(cpath)
                if cd:
                    src_note = ("score_3d_chunked.py（官方 summarize_acc.py 逐字调用；"
                                "**兜底口径：排除 %d 个深景病态样本（官方判分单条 >65 min），"
                                "分母 = %d**）" % (
                                    cd.get("n_excluded_samples", 0),
                                    cd.get("audit", {}).get("n_evaluated_total", 0)))
            if not cd:
                print("[merge_ts] %s/%s: 无分片结果（%s）" % (variant, task, cpath))
                continue
            scores = {
                "metrics_2d": cd.get("metrics_2d"),
                "start_in_mask": cd.get("start_in_mask"),
                "end_in_bbox": cd.get("end_in_bbox"),
                "metrics_3d": cd.get("metrics_3d"),
                "start_distance": cd.get("start_distance"),
                "end_distance": cd.get("end_distance"),
                "no_collision_rate": cd.get("no_collision_rate"),
                "start_success_rate": cd.get("start_success_rate"),
                "end_success_rate": cd.get("end_success_rate"),
                "overall_success_rate": cd.get("overall_success_rate"),
                "collision_warnings": sum(
                    c.get("counts", {}).get("warn3d_collision", 0)
                    for c in cd.get("per_chunk", [])),
                "skip_depthless": sum(c.get("counts", {}).get("skip", 0)
                                      for c in cd.get("per_chunk", [])),
                "open3d_available": not cd.get("audit", {}).get("open3d_degraded", False),
                "scoring_mode": cd.get("merge_method"),
                "scoring_audit": cd.get("audit"),
                "scoring_source": src_note,
                "n_excluded_samples": cd.get("n_excluded_samples", 0),
            }
            ti["scores"] = scores
            touched += 1
            print("[merge_ts] %s/%s -> start=%s end=%s overall=%s collision_free=%s" % (
                variant, task, scores["start_success_rate"],
                scores["end_success_rate"], scores["overall_success_rate"],
                scores["no_collision_rate"]))

    with open(sp, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print("[merge_ts] 已回写 %s（%d 个组合）" % (sp, touched))
    return 0


if __name__ == "__main__":
    sys.exit(main())
