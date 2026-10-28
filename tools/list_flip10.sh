#!/usr/bin/env bash
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
python3 - <<'PY'
import json, os
WS = "/root/private_data/workspace/robo/outputs/bench_public"
A = os.path.join(WS, "cvbench_upscale", "canonical_png", "predictions", "cvbench.jsonl")
B = os.path.join(WS, "cvbench_prompt", "flagevalmm", "predictions", "cvbench.jsonl")

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

pa, pb = load(A), load(B)
letter = punct = other = 0
rows = []
for q in sorted(set(pa) & set(pb), key=lambda s: int(s.split("/")[-1])):
    x, y = pa[q].get("raw_output"), pb[q].get("raw_output")
    if x == y:
        continue
    sx, sy = str(x).strip(), str(y).strip()
    kind = "标点差异" if (sx.rstrip(".") == sy.rstrip(".")) else "字母不同"
    if kind == "字母不同":
        letter += 1
    else:
        punct += 1
    rows.append((q, sx, sy, kind))
print("共 %d 题不同：字母不同 %d，标点差异 %d" % (len(rows), letter, punct))
for q, sx, sy, k in rows:
    print("  %-14s %-6s A=%-6r B=%-6r" % (q, k, sx, sy))
PY
