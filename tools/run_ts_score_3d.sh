#!/usr/bin/env bash
# TraceSpatial 3D 官方判分（分片并行，4 个组合共用一个线程池）
#
# 关键点：
#   * 官方 summarize_acc.py 一字未改，只改并发调度；
#   * --chunk-size 1（逐样本分片）让各 Rate 的计数无损，分片合并可精确复现整集结果；
#   * ROBO_O3D_PREFIX 指向修好的 open3d 前缀（0.20 的 ScalableTSDFVolume 失效）；
#   * --mem-budget-gb 是 cgroup 内存闸门，防止并发判分把 115GiB 上限打爆。
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONPATH="$WS/envs/pylibs_core:$WS/third_party/upstream_eval/FlagEvalMM:$WS/scripts:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore
export TMPDIR="$WS/tmp"

O3D_PREFIX="${ROBO_O3D_PREFIX:-$WS/envs/pylibs_o3d}"
export ROBO_O3D_PREFIX="$O3D_PREFIX"
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

CONC="${1:-12}"
BUDGET="${2:-45}"
LOG="$WS/logs/ts_score_3d.log"

echo "===== TraceSpatial 3D 官方判分 $(date '+%F %T') =====" | tee "$LOG"
echo "  open3d 前缀: $O3D_PREFIX" | tee -a "$LOG"
python3 -c "import sys; sys.path.insert(0,'$O3D_PREFIX'); import open3d; print('  open3d', open3d.__version__)" 2>&1 | tail -2 | tee -a "$LOG"

python3 "$WS/submission/bench_public/score_3d_chunked.py" \
  --job "$WS/outputs/bench_public/tracespatial|official|3D" \
  --job "$WS/outputs/bench_public/tracespatial|upstream|3D" \
  --job "$WS/outputs/bench_public_noneager/tracespatial|official|3D" \
  --job "$WS/outputs/bench_public_noneager/tracespatial|upstream|3D" \
  --chunk-size 1 --concurrency "$CONC" --mem-budget-gb "$BUDGET" \
  2>&1 | tee -a "$LOG"

echo "===== 3D 判分结束 $(date '+%F %T') =====" | tee -a "$LOG"
