#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""原始输出对拍（修正：cvbench.jsonl 实为单个 JSON 对象）"""
import json
import os

WS = "/root/private_data/workspace/robo/outputs/bench_public"
A = os.path.join(WS, "cvbench_upscale", "canonical_png")
B = os.path.join(WS, "cvbench_prompt", "flagevalmm")
TARGETS = ["30", "52", "221", "635", "665"]


def load(p):
    with open(p, encoding="utf-8") as fh:
        d = json.load(fh)
    if isinstance(d, list):
        out = {}
        for r in d:
            out[str(r.get("sample_id", r.get("question_id")))] = r
        return out
    # dict：可能形如 {"records": [...]} 或 {sid: rec}
    if "records" in d and isinstance(d["records"], list):
        return {str(r.get("sample_id", r.get("question_id"))): r for r in d["records"]}
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[str(k)] = v
    return out


pa = load(os.path.join(A, "predictions", "cvbench.jsonl"))
pb = load(os.path.join(B, "predictions", "cvbench.jsonl"))
print("记录数 A=%d B=%d  共同=%d" % (len(pa), len(pb), len(set(pa) & set(pb))))

common = set(pa) & set(pb)
n_raw = 0
n_parsed = 0
shown = 0
for q in sorted(common, key=lambda x: int(x) if x.isdigit() else 0):
    ra, rb = pa[q], pb[q]
    ta, tb = ra.get("raw_output"), rb.get("raw_output")
    pa_, pb_ = ra.get("parsed"), rb.get("parsed")
    if ta != tb:
        n_raw += 1
    if json.dumps(pa_, sort_keys=True, ensure_ascii=False) != \
       json.dumps(pb_, sort_keys=True, ensure_ascii=False):
        n_parsed += 1
        if shown < 10:
            shown += 1
            print("\n--- q%s ---" % q)
            print("  A raw=%r" % (str(ta)[:100]))
            print("  B raw=%r" % (str(tb)[:100]))
            print("  A parsed=%s" % json.dumps(pa_, ensure_ascii=False)[:200])
            print("  B parsed=%s" % json.dumps(pb_, ensure_ascii=False)[:200])

print("\n原始 raw_output 不同的题数 = %d / %d = %.2f%%" % (n_raw, len(common),
                                                          100.0 * n_raw / max(1, len(common))))
print("parsed 不同的题数 = %d / %d = %.2f%%" % (n_parsed, len(common),
                                                100.0 * n_parsed / max(1, len(common))))
print("\n=== 目标题号 ===")
for q in TARGETS:
    if q in pa and q in pb:
        print("  q%-5s A raw=%r" % (q, str(pa[q].get("raw_output"))[:80]))
        print("         B raw=%r" % (q, str(pb[q].get("raw_output"))[:80]))
        print("         A parsed=%s | B parsed=%s"
              % (json.dumps(pa[q].get("parsed"), ensure_ascii=False)[:110],
                 json.dumps(pb[q].get("parsed"), ensure_ascii=False)[:110]))
