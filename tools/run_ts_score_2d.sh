#!/usr/bin/env bash
# TraceSpatial 2D 官方判分（4 个组合共用一个线程池）
# 2D 判分只用 numpy/scipy/PIL，不依赖 open3d 的 TSDF，可与 3D 判分并行
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONPATH="$WS/envs/pylibs_core:$WS/third_party/upstream_eval/FlagEvalMM:$WS/scripts:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore
export TMPDIR="$WS/tmp"
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

LOG="$WS/logs/ts_score_2d.log"
echo "===== TraceSpatial 2D 官方判分（分片并行）$(date '+%F %T') =====" | tee "$LOG"

python3 "$WS/submission/bench_public/score_3d_chunked.py" \
  --job "$WS/outputs/bench_public/tracespatial|official|2D" \
  --job "$WS/outputs/bench_public/tracespatial|upstream|2D" \
  --job "$WS/outputs/bench_public_noneager/tracespatial|official|2D" \
  --job "$WS/outputs/bench_public_noneager/tracespatial|upstream|2D" \
  --chunk-size 1 --concurrency 16 --mem-budget-gb 40 \
  2>&1 | tee -a "$LOG"

echo "===== 2D 判分结束 $(date '+%F %T') =====" | tee -a "$LOG"
