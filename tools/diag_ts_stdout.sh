#!/usr/bin/env bash
# 看官方 3D 判分的原始 stdout（定位 start/end distance 为 N/A 的原因）
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONWARNINGS=ignore
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

python3 - <<'PY'
import json
p = "/root/private_data/workspace/robo/tmp/probe_3d_fixed.json"
d = json.load(open(p, encoding="utf-8"))
for c in d["per_chunk"][:2]:
    print("=" * 70)
    print("chunk", c["chunk_index"], "counts:", c["counts"])
    print(c.get("stdout_tail", "(无)"))
PY

echo
echo "===== 官方脚本里 start distance 的原始逻辑（逐字） ====="
sed -n '948,1000p' "$WS/third_party/upstream_eval/RoboTracer/Evaluation/summarize_acc.py"
