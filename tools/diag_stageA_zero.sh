#!/usr/bin/env bash
set -uo pipefail
WS=/root/private_data/workspace/robo
cd "$WS"
export PYTHONPATH="$WS/envs/pylibs_core:$WS/third_party/upstream_eval/FlagEvalMM:$WS/scripts:${PYTHONPATH:-}"
export ROBO_O3D_PREFIX="$WS/envs/pylibs_o3d_v2"
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

echo "=== 阶段A 产物的 stdout_tail（official/3D）==="
python3 - <<'PY'
import json, os
WS = "/root/private_data/workspace/robo"
p = WS + "/outputs/bench_public/tracespatial/official/scores_3dchunk_3D.json"
d = json.load(open(p, encoding="utf-8"))
pc = d.get("per_chunk") or []
print("per_chunk 数:", len(pc))
for c in pc:
    print("chunk", c.get("chunk_index"), "n_answers", c.get("n_answers"),
          "rc", c.get("returncode"), "wall", c.get("wall_s"))
    print("--- stdout_tail ---")
    print(c.get("stdout_tail", "")[:3000])
    print("--- stderr_tail ---")
    print(c.get("stderr_tail", "")[:2000])
    break
PY

echo
echo "=== 该 chunk 目录内容 ==="
ls -la "$WS/tmp/ts3d_verify_full/tracespatial_official_3D/c0000/" 2>/dev/null
ls -la "$WS/tmp/ts3d_verify_full/tracespatial_official_3D/c0000/outputs/" 2>/dev/null
echo "--- jsonl 行数 ---"
wc -l "$WS/tmp/ts3d_verify_full/tracespatial_official_3D/c0000/outputs/"*.jsonl 2>/dev/null
echo "--- 第一行前 300 字符 ---"
head -c 300 "$WS/tmp/ts3d_verify_full/tracespatial_official_3D/c0000/outputs/"*.jsonl 2>/dev/null
echo
echo "--- 官方脚本如何定位 benchmark 目录 / answers ---"
grep -n "TraceSpatial-Bench\|benchmark_folder\|def main\|argparse\|output_save_folder" "$WS/third_party/upstream_eval/RoboTracer/Evaluation/summarize_acc.py" | head -30
