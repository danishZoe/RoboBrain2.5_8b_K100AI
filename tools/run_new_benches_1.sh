#!/usr/bin/env bash
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONPATH="$WS/envs/pylibs_core:$WS/third_party/upstream_eval/FlagEvalMM:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore
export TMPDIR="$WS/tmp"
set +u; source /opt/dtk/env.sh >/dev/null 2>&1; set -u
OUT="$WS/outputs/bench_public_noneager2"
mkdir -p "$OUT"
echo "===== EmbSpatial 全量 $(date +%F %T) ====="
python3 submission/bench_public/run_embspatial.py --out-dir "$OUT/embspatial" \
  --run-id noneager-embspatial --concurrency 16 2>&1 | tail -12
echo "[embspatial] rc=${PIPESTATUS[0]}"
echo "===== CrossPoint 全量 $(date +%F %T) ====="
python3 submission/bench_public/run_crosspoint.py --out-dir "$OUT/crosspoint" \
  --run-id noneager-crosspoint --concurrency 16 2>&1 | tail -30
echo "[crosspoint] rc=${PIPESTATUS[0]}"
date +%s > "$WS/tmp/newbenches_1.end.ts"
echo "===== 第一批结束 $(date +%F %T) ====="
