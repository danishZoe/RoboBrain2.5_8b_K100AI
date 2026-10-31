#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐分片打印官方 stdout 的解析结果与计数，定位合并口径差异的来源。"""
import json
import os
import sys

WS = "/root/private_data/workspace/robo"
root = sys.argv[1] if len(sys.argv) > 1 else WS + "/outputs/bench_public"
variant = sys.argv[2] if len(sys.argv) > 2 else "official"
task = sys.argv[3] if len(sys.argv) > 3 else "2D"

p = os.path.join(root, "tracespatial", variant, "scores_3dchunk_%s.json" % task)
d = json.load(open(p, encoding="utf-8"))
per = d.get("per_chunk") or []
print("文件:", p)
print("合并值 metrics_2d:", json.dumps(d.get("metrics_2d"), ensure_ascii=False))
print("合并值 metrics_3d:", json.dumps(d.get("metrics_3d"), ensure_ascii=False))
print("分片数:", len(per))

n_skip = n_warn2d = n_warnm = 0
allzero = 0
empty = 0
sumv = {k: 0.0 for k in ("average_discrete_frechet_distance",
                         "average_hausdorff_distance",
                         "average_root_mean_square_error",
                         "average_mean_absolute_error")}
n_used = 0
for c in per:
    k = c.get("counts", {})
    ev = c["n_answers"] - k.get("skip", 0)
    n_skip += k.get("skip", 0)
    n_warn2d += k.get("warn2d", 0)
    n_warnm += k.get("warn_metric", 0)
    m = c.get("metrics_2d") or {}
    if not m:
        empty += 1
        continue
    if all(abs(v) < 1e-12 for v in m.values()):
        allzero += 1
        continue
    cnt = max(ev - k.get("warn_metric", 0) - k.get("warn2d", 0), 0)
    n_used += cnt
    for kk, vv in m.items():
        sumv[kk] += vv * cnt

print("skip 合计=%d  warn2d 合计=%d  warn_metric 合计=%d" % (n_skip, n_warn2d, n_warnm))
print("metrics_2d 为空的分片数=%d   全零分片数=%d   参与求和的分片数=%d" % (empty, allzero, n_used))
print("按'均值×cnt'重算:", json.dumps({kk: round(vv / n_used, 8) for kk, vv in sumv.items()},
                                      ensure_ascii=False) if n_used else "N/A")
# 哪些分片有 warn
odd = [c for c in per if c.get("counts", {}).get("warn_metric") or c.get("counts", {}).get("warn2d")
       or c.get("counts", {}).get("skip") or not (c.get("metrics_2d") or {})]
print("\n异常分片（skip/warn/空 metrics）共 %d 个：" % len(odd))
for c in odd[:12]:
    print("  chunk %-4d n=%d skip=%d warn2d=%d warn_metric=%d warn3d_c=%d metrics_2d=%s" %
          (c["chunk_index"], c["n_answers"], c["counts"].get("skip", 0),
           c["counts"].get("warn2d", 0), c["counts"].get("warn_metric", 0),
           c["counts"].get("warn3d_collision", 0),
           json.dumps(c.get("metrics_2d"), ensure_ascii=False)))
    tail = c.get("stdout_tail", "")
    for ln in tail.splitlines():
        if "[WARN]" in ln or "[Warn]" in ln or "[Skip]" in ln:
            print("      | " + ln[:180])
