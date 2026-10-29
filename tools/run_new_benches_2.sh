#!/usr/bin/env bash
# 第二批：RoboSpatial-Home 全量（等第一批 EmbSpatial/CrossPoint 完成后启动，
# 避免并发混批；服务常驻，Client 并发 16 不变）
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONPATH="$WS/envs/pylibs_core:$WS/third_party/upstream_eval/FlagEvalMM:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore
export TMPDIR="$WS/tmp"
set +u; source /opt/dtk/env.sh >/dev/null 2>&1; set -u
OUT="$WS/outputs/bench_public_noneager2"

for i in $(seq 1 240); do
  [ -f "$WS/tmp/newbenches_1.end.ts" ] && break
  sleep 30
done
[ -f "$WS/tmp/newbenches_1.end.ts" ] || { echo "[batch2] 等待第一批超时"; exit 1; }

echo "===== RoboSpatial-Home 全量 $(date '+%F %T') ====="
python3 submission/bench_public/run_robospatial.py --out-dir "$OUT/robospatial" \
  --run-id noneager-robospatial --concurrency 16 2>&1 | tail -6
echo "[robospatial] rc=${PIPESTATUS[0]}"
date '+%s' > "$WS/tmp/newbenches_2.end.ts"
echo "===== 第二批结束 $(date '+%F %T') ====="
