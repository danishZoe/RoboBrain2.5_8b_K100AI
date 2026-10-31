#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""原始输出对拍（按 sample_id 索引）"""
import json
import os

WS = "/root/private_data/workspace/robo/outputs/bench_public"
A = os.path.join(WS, "cvbench_upscale", "canonical_png")
B = os.path.join(WS, "cvbench_prompt", "flagevalmm")
TARGETS = ["CVBench/30", "CVBench/52", "CVBench/221", "CVBench/635", "CVBench/665"]


def load(p):
    out = {}
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[str(d.get("sample_id"))] = d
    return out


pa = load(os.path.join(A, "predictions", "cvbench.jsonl"))
pb = load(os.path.join(B, "predictions", "cvbench.jsonl"))
common = sorted(set(pa) & set(pb), key=lambda s: int(s.split("/")[-1]))
print("记录数 A=%d B=%d 共同=%d" % (len(pa), len(pb), len(common)))

n_raw = n_parsed = n_geom = 0
shown = 0
for q in common:
    ra, rb = pa[q], pb[q]
    if ra.get("raw_output") != rb.get("raw_output"):
        n_raw += 1
        if shown < 8:
            shown += 1
            print("\n--- %s ---" % q)
            print("  A raw=%r" % (str(ra.get("raw_output"))[:110]))
            print("  B raw=%r" % (str(rb.get("raw_output"))[:110]))
            print("  A parsed=%s" % json.dumps(ra.get("parsed"), ensure_ascii=False)[:160])
            print("  B parsed=%s" % json.dumps(rb.get("parsed"), ensure_ascii=False)[:160])
    if json.dumps(ra.get("parsed"), sort_keys=True) != json.dumps(rb.get("parsed"), sort_keys=True):
        n_parsed += 1
    if json.dumps(ra.get("image_geometry"), sort_keys=True) != json.dumps(rb.get("image_geometry"), sort_keys=True):
        n_geom += 1

print("\nraw_output 不同的题数 = %d / %d = %.2f%%" % (n_raw, len(common), 100.0 * n_raw / len(common)))
print("parsed 不同的题数   = %d / %d = %.2f%%" % (n_parsed, len(common), 100.0 * n_parsed / len(common)))
print("image_geometry 差异  = %d" % n_geom)

print("\n=== 目标题号 ===")
for q in TARGETS:
    if q in pa and q in pb:
        print("  %-14s A raw=%r" % (q, str(pa[q].get("raw_output"))[:60]))
        print("  %-14s B raw=%r" % ("", str(pb[q].get("raw_output"))[:60]))

# 官方判分输入（predictions_official.json）里这些题的答案
print("\n=== 官方判分输入 predictions_official.json 对照 ===")
for tag, d in (("A", A), ("B", B)):
    po = json.load(open(os.path.join(d, "predictions_official.json"), encoding="utf-8"))
    m = {str(x.get("question_id")): x.get("answer") for x in po}
    print("  %s: " % tag, {q.split("/")[-1]: m.get(q) for q in TARGETS})
