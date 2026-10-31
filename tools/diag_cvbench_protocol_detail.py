#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CV-Bench 图像协议实验的 per-source/task 拆解（只跑过的 cell 才统计）。"""
import collections
import json
import os
import sys

WS = "/root/private_data/workspace/robo"
PROTO_ROOT = WS + "/outputs/bench_public/cvbench_protocol"

ORDER = ["canonical_png", "canonical_jpeg95", "raw_png", "raw_jpeg95",
         "long1500_png", "long1500_jpeg95", "maxpix501760_png", "maxpix501760_jpeg95"]

def load(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None

def breakdown(root):
    anns = load(os.path.join(root, "annotations.json"))
    per = load(os.path.join(root, "per_sample.json"))
    if not anns or not per:
        return None
    st = collections.defaultdict(lambda: [0, 0])
    for r in per:
        g = anns.get(r["question_id"], {})
        k = "%s/%s" % (g.get("source"), g.get("task"))
        st[k][0] += ((r.get("gt_answer") or "")[:1].upper() ==
                     (r.get("extracted_answer") or "")[:1].upper())
        st[k][1] += 1
    return st

def rate(v):
    return 100.0 * v[0] / v[1] if v[1] else None

print("%-20s %-22s %7s %7s %8s" % ("protocol", "cell", "correct", "total", "acc%"))
base = {}
for p in ORDER:
    root = os.path.join(PROTO_ROOT, p)
    st = breakdown(root)
    if not st:
        print("%-20s  <缺产物>" % p)
        continue
    for k in sorted(st):
        c, n = st[k]
        a = 100.0 * c / n
        d = ""
        if p == ORDER[0]:
            base[k] = a
        elif k in base:
            d = "  (Δ%+.2f)" % (a - base[k])
        print("%-20s   %-20s %7d %7d %8.2f%s" % (p, k, c, n, a, d))
    allc = sum(v[0] for v in st.values())
    alln = sum(v[1] for v in st.values())
    print("%-20s   %-20s %7d %7d %8.2f" % (p, "ALL", allc, alln, 100.0 * allc / alln))
    print()
