#!/usr/bin/env bash
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
python3 - <<'PY'
import json
p = "outputs/bench_public/cvbench_upscale/canonical_png/per_sample.json"
d = json.load(open(p, encoding="utf-8"))
print("type:", type(d).__name__)
if isinstance(d, dict):
    ks = list(d.keys())
    print("keys[:5]:", ks[:5], "len:", len(ks))
    k = ks[0]
    print("value[%r] = %s" % (k, json.dumps(d[k], ensure_ascii=False)[:600]))
else:
    print("len:", len(d))
    print("elem0 =", json.dumps(d[0], ensure_ascii=False)[:600])
PY
