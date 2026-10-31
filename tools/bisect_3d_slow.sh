#!/usr/bin/env bash
# 分块定位：官方 3D 判分里哪些样本是"病理慢"的（10 条一组，并行计时）
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
BASE=$WS/tmp/chunk3d
rm -rf "$BASE" 2>/dev/null || true
mkdir -p "$BASE"

run_chunk() {
  local i=$1
  local D="$BASE/c$i"
  mkdir -p "$D/outputs"
  ln -sfn "$WS/third_party/benchmarks/TraceSpatial-Bench" "$D/TraceSpatial-Bench"
  sed -n "$((i*10-9)),$((i*10))p" "$SRC/${MID}_3D.jsonl" > "$D/outputs/${MID}_3D.jsonl"
  local s=$(date +%s)
  ( cd "$D" && timeout 900 python3 "$ROBO/summarize_acc.py" --model_name "$MID" --task_name 3D \
      > "$D/out.txt" 2> "$D/err.txt" )
  local e=$(date +%s)
  echo "chunk $((i*10-9))-$((i*10)) : $((e-s))s" >> "$BASE/timings.txt"
}

for i in 1 2 3 4 5 6 7 8 9 10; do
  run_chunk "$i" &
done
wait
echo "===== 每 10 条一块的耗时 ====="
sort -t: -k2 -n "$BASE/timings.txt"
echo
echo "===== 慢块里的逐条耗时再切（对 >60s 的块二分）====="
awk -F' : ' '{split($1,a," "); gsub(/[^0-9]/,"",a[1]); print a[1], $2}' "$BASE/timings.txt" | sort -k2 -n | tail -3
