#!/usr/bin/env bash
# 测速：官方 summarize_acc.py 的 3D 判分单样本成本（独立目录，不影响正在跑的任务）
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONPATH="$WS/envs/pylibs_o3d:$WS/envs/pylibs:$WS/envs/pylibs_core:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

ROBO=$WS/third_party/upstream_eval/RoboTracer/Evaluation
SRC=$WS/outputs/bench_public/tracespatial/official/outputs
MID=RoboTracer_RoboBrain2.5-8B-NV
N=${1:-10}

for K in 3 10; do
  D="$WS/tmp/speed3d_$K"
  rm -rf "$D" 2>/dev/null || true
  mkdir -p "$D/outputs"
  ln -sfn "$WS/third_party/benchmarks/TraceSpatial-Bench" "$D/TraceSpatial-Bench"
  head -n "$K" "$SRC/${MID}_3D.jsonl" > "$D/outputs/${MID}_3D.jsonl"
  echo "===== 官方 3D 判分：$K 条样本 ====="
  START=$(date +%s)
  ( cd "$D" && python3 "$ROBO/summarize_acc.py" --model_name "$MID" --task_name 3D \
      > "$D/out.txt" 2> "$D/err.txt" )
  END=$(date +%s)
  EL=$((END-START))
  echo "  用时 ${EL}s -> 单样本 $((EL/K))s ; 推算 100 条 = $((EL*100/K))s"
  grep -E 'Start Success|End Success|Overall Success|No-Collision' "$D/out.txt" | head -6
  echo
done
