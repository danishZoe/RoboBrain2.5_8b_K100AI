#!/usr/bin/env bash
W=/root/private_data/workspace/robo
cd "$W" || exit 1

echo "=== 1. phase3 baseline_raw.json 结构 ==="
python3 - <<'PY'
import json
p="/root/private_data/workspace/robo/outputs/phase3_baseline/baseline_raw.json"
d=json.load(open(p,encoding="utf-8"))
print("type:",type(d).__name__)
if isinstance(d,dict):
    print("keys:",list(d.keys()))
    for k,v in d.items():
        if isinstance(v,list) and v:
            print(f"  [{k}] n={len(v)} first-keys={list(v[0].keys())}")
            print("  first:", json.dumps(v[0],ensure_ascii=False)[:400])
elif isinstance(d,list) and d:
    print("n=",len(d),"keys=",list(d[0].keys()))
    print("first:", json.dumps(d[0],ensure_ascii=False)[:500])
PY

echo
echo "=== 2. verify.py 期望的 subset 记录字段 ==="
grep -n "rec\[\|rec\.get\|pick_field\|image\|reference\|prompt\|task" "$W/submission/verify.py" | sed -n '1,60p'
