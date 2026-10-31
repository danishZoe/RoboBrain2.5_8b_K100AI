#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CV-Bench Count 误差量级分布 + per-source/task 拆解（只读产物）。"""
import collections
import json
import os
import re
import sys

WS = "/root/private_data/workspace/robo"
ROOT = sys.argv[1] if len(sys.argv) > 1 else WS + "/outputs/bench_public/cvbench"

anns = json.load(open(ROOT + "/annotations.json", encoding="utf-8"))
per = json.load(open(ROOT + "/per_sample.json", encoding="utf-8"))

by_st = collections.defaultdict(lambda: [0, 0])
diff_hist = collections.Counter()
diff_by_src = collections.defaultdict(collections.Counter)
absdiff_hist = collections.Counter()
n_correct = 0
n_cnt = 0
exact_value_correct_but_wrong_letter = 0

for r in per:
    qid = r["question_id"]
    g = anns.get(qid, {})
    task = g.get("task") or "?"
    src = g.get("source") or "?"
    gt = (r.get("gt_answer") or "").strip()[:1].upper()
    ex = (r.get("extracted_answer") or "").strip()
    pred = ex[:1].upper()
    ok = (gt == pred)
    by_st["%s/%s" % (src, task)][0] += ok
    by_st["%s/%s" % (src, task)][1] += 1
    if task != "Count":
        continue
    n_cnt += 1
    n_correct += ok
    opts = g.get("options") or []
    gi, pi = ord(gt) - 65, ord(pred) - 65
    if 0 <= gi < len(opts) and 0 <= pi < len(opts):
        try:
            gv, pv = int(opts[gi]), int(opts[pi])
        except ValueError:
            continue
        d = pv - gv
        diff_hist[d] += 1
        diff_by_src[src][d] += 1
        absdiff_hist[abs(d)] += 1
        if d != 0 and pv == gv:
            exact_value_correct_but_wrong_letter += 1

print("Count 总数 %d，正确 %d (%.2f%%)" % (n_cnt, n_correct, 100.0 * n_correct / n_cnt))
print("\n=== Count 数值误差 (pred_value - gt_value) 分布 ===")
for d in sorted(diff_hist):
    print("   %+3d : %4d  (%.1f%%)" % (d, diff_hist[d], 100.0 * diff_hist[d] / n_cnt))
print("\n=== |误差| 分布 ===")
for d in sorted(absdiff_hist):
    print("   %3d : %4d  (%.1f%%)" % (d, absdiff_hist[d], 100.0 * absdiff_hist[d] / n_cnt))
print("\n=== 按 source 的误差分布 ===")
for src, c in diff_by_src.items():
    print("  ", src, dict(sorted(c.items())))

print("\n=== per source/task 准确率 ===")
for k in sorted(by_st):
    v = by_st[k]
    print("   %-28s %4d/%4d = %6.2f%%" % (k, v[0], v[1], 100.0 * v[0] / v[1]))

# 官方聚合口径
def acc(src, task=None):
    v = by_st.get("%s/%s" % (src, task) if task else src)
    if not v or not v[1]:
        return None
    return v[0] / v[1]

srcs = sorted({k.split("/")[0] for k in by_st})
print("\n=== 官方口径聚合 ===")
parts = {}
for s in srcs:
    tasks = sorted({k.split("/", 1)[1] for k in by_st if k.startswith(s + "/")})
    print("  %s: %s" % (s, {t: round(100 * acc(s, t), 2) for t in tasks}))
    parts[s] = acc(s)
acc2d = (parts.get("ADE20K", 0) + parts.get("COCO", 0)) / 2
acc3d = parts.get("Omni3D", 0)
print("  accuracy_2d = %.2f  accuracy_3d = %.2f  accuracy = %.2f" %
      (100 * acc2d, 100 * acc3d, 100 * (acc2d + acc3d) / 2))
