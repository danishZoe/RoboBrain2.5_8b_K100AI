#!/usr/bin/env bash
W=/root/private_data/workspace/robo
cd "$W" || exit 1
echo "=== outputs/ 结构 ==="
find outputs -maxdepth 2 -type f 2>/dev/null | head -40
echo
echo "=== 含 baseline / transformers 参考输出的文件 ==="
for f in $(find outputs -maxdepth 3 -type f -name '*.json' 2>/dev/null | head -40); do
  if grep -lq 'baseline' "$f" 2>/dev/null; then echo "  $f"; fi
done
echo
echo "=== task_results.json 顶层结构 ==="
python3 - <<'PY'
import json, glob, os
p = "/root/private_data/workspace/robo/outputs/tasks_official/task_results.json"
if os.path.exists(p):
    d = json.load(open(p, encoding="utf-8"))
    print("type:", type(d).__name__)
    if isinstance(d, list) and d:
        print("keys:", list(d[0].keys()))
        print("first id:", d[0].get("id"))
    elif isinstance(d, dict):
        print("keys:", list(d.keys())[:12])
PY
echo
echo "=== 压测状态 ==="
date +%H:%M:%S
wc -l submission/bench/round1.jsonl submission/bench/round2.jsonl 2>/dev/null
tr '\r' '\n' < submission/bench/round1_16384_1024_c64.log | grep -E '^[0-9]+%' | tail -1
