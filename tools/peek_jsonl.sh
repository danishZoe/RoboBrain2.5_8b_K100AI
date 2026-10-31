#!/usr/bin/env bash
WS=/root/private_data/workspace/robo
P="$WS/outputs/bench_public/cvbench_upscale/canonical_png/predictions/cvbench.jsonl"
echo "=== 行数 ==="
wc -l "$P"
echo "=== 前 2 行（各截 700 字符） ==="
head -2 "$P" | cut -c1-700
echo
echo "=== 逐行 JSON 可解析性统计 ==="
python3 - <<'PY'
import json
p = "/root/private_data/workspace/robo/outputs/bench_public/cvbench_upscale/canonical_png/predictions/cvbench.jsonl"
ok = bad = 0
badlines = []
with open(p, encoding="utf-8") as fh:
    for i, line in enumerate(fh, 1):
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
            ok += 1
        except ValueError as e:
            bad += 1
            if len(badlines) < 3:
                badlines.append((i, str(e)[:80], line[:120]))
print("可解析行 =", ok, " 不可解析行 =", bad)
for i, e, s in badlines:
    print("  行%d: %s\n     %s" % (i, e, s))
PY
