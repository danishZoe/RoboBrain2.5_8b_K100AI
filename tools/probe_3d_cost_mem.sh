#!/usr/bin/env bash
# 测：官方 3D 判分的耗时/内存结构 —— 病理样本是集中还是均匀？单进程峰值内存多少？
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
BASE=$WS/tmp/probe3d
rm -rf "$BASE" 2>/dev/null || true
mkdir -p "$BASE"

run_range() {
  local lo=$1 hi=$2 tag=$3
  local D="$BASE/$tag"
  mkdir -p "$D/outputs"
  ln -sfn "$WS/third_party/benchmarks/TraceSpatial-Bench" "$D/TraceSpatial-Bench"
  sed -n "${lo},${hi}p" "$SRC/${MID}_3D.jsonl" > "$D/outputs/${MID}_3D.jsonl"
  local s=$(date +%s)
  ( cd "$D" && /usr/bin/time -v python3 "$ROBO/summarize_acc.py" --model_name "$MID" --task_name 3D \
      > "$D/out.txt" 2> "$D/err.txt" )
  local e=$(date +%s)
  local rss=$(grep -o 'Maximum resident set size (kbytes): [0-9]*' "$D/err.txt" | grep -o '[0-9]*$')
  echo "  $tag (样本 $lo-$hi): $((e-s))s  峰值RSS=$((rss/1024))MB  -> $((e-s))s/样本" \
      | tee -a "$BASE/result.txt"
}

echo "===== 病理区间二分（并发跑，每块独立计时 + 峰值内存）====="
run_range 51 55 "c51-55" &
run_range 56 60 "c56-60" &
run_range 71 75 "c71-75" &
run_range 76 80 "c76-80" &
wait
echo
echo "===== 结果 ====="
cat "$BASE/result.txt"
echo
echo "===== 各块是否出现 Skip / Warn（决定分片合并是否可精确）====="
for d in "$BASE"/*/; do
  t=$(basename "$d")
  sk=$(grep -c '^\[Skip\]' "$d/out.txt" 2>/dev/null || echo 0)
  wn=$(grep -c '^\[Warn\]' "$d/out.txt" 2>/dev/null || echo 0)
  echo "  $t: Skip=$sk Warn=$wn"
done
