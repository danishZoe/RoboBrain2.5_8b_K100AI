#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CV-Bench 失分层级分析（按 task / 选项数 / source）。

用途：把"综合分低 X 个百分点"拆成"哪一个子任务低"，避免笼统归因。
只读已落盘的逐题产物，不发任何请求。

用法::

    python3 submission/bench_public/analyze_cvbench.py \
        --bench-root outputs/bench_public/cvbench \
        --out outputs/bench_public/cvbench/task_breakdown.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def _load(path: str):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def analyze(bench_root: str) -> Dict[str, Any]:
    anns = _load(os.path.join(bench_root, "annotations.json")) or {}
    preds = _load(os.path.join(bench_root, "predictions_official.json")) or []
    if not anns or not preds:
        return {"status": "NOT_RUN",
                "reason": "缺 annotations.json / predictions_official.json"}

    by_task: Dict[str, List[int]] = collections.defaultdict(lambda: [0, 0])
    by_nopt: Dict[int, List[int]] = collections.defaultdict(lambda: [0, 0])
    by_src_task: Dict[str, List[int]] = collections.defaultdict(lambda: [0, 0])
    conf: collections.Counter = collections.Counter()
    pred_letter: collections.Counter = collections.Counter()
    gt_letter: collections.Counter = collections.Counter()
    wrong: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)

    for p in preds:
        qid = str(p.get("question_id"))
        g = anns.get(qid)
        if g is None:
            continue
        pa = (p.get("answer") or "")[:1].upper()
        ga = (g.get("answer") or "")[:1].upper()
        ok = (ga == pa)
        task = g.get("task") or "?"
        nopt = len(g.get("options") or [])
        by_task[task][0] += ok
        by_task[task][1] += 1
        by_nopt[nopt][0] += ok
        by_nopt[nopt][1] += 1
        by_src_task["%s/%s" % (g.get("source"), task)][0] += ok
        by_src_task["%s/%s" % (g.get("source"), task)][1] += 1
        conf[(ga, pa)] += 1
        pred_letter[pa] += 1
        gt_letter[ga] += 1
        if not ok and len(wrong[task]) < 3:
            wrong[task].append({"question_id": qid,
                                "question": (g.get("question") or "")[:160],
                                "options": g.get("options"),
                                "gt": ga, "pred": pa,
                                "filename": g.get("filename")})

    def rate(pair):
        r, n = pair
        return round(100.0 * r / n, 2) if n else None

    return {
        "status": "OK",
        "n": len(preds),
        "by_task": {k: {"correct": v[0], "total": v[1], "accuracy": rate(v)}
                    for k, v in sorted(by_task.items(), key=lambda kv: -kv[1][1])},
        "by_n_options": {str(k): {"correct": v[0], "total": v[1], "accuracy": rate(v)}
                         for k, v in sorted(by_nopt.items())},
        "by_source_task": {k: {"correct": v[0], "total": v[1], "accuracy": rate(v)}
                           for k, v in sorted(by_src_task.items())},
        "pred_letter_hist": dict(pred_letter.most_common()),
        "gt_letter_hist": dict(gt_letter.most_common()),
        "confusion_top15": [{"gt": g, "pred": p, "n": c}
                            for (g, p), c in conf.most_common(15)],
        "wrong_samples": dict(wrong),
        "overall_accuracy_by_letter": rate([sum(v[0] for v in by_task.values()),
                                             sum(v[1] for v in by_task.values())]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root",
                    default=os.path.join(C.WORK_DIR, "outputs/bench_public/cvbench"))
    ap.add_argument("--out",
                    default=os.path.join(C.WORK_DIR,
                                         "outputs/bench_public/cvbench/task_breakdown.json"))
    args = ap.parse_args()

    res = analyze(args.bench_root)
    C.dump_json(args.out, res)
    print("[analyze_cvbench] -> %s" % args.out)
    if res.get("status") == "OK":
        print("  按 task:")
        for k, v in res["by_task"].items():
            print("    %-12s %4d/%4d = %6.2f%%" %
                  (k, v["correct"], v["total"], v["accuracy"]))
    else:
        print("  %s: %s" % (res.get("status"), res.get("reason")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
