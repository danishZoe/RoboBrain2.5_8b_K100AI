#!/usr/bin/env bash
# 统一主驱动（第三批）：等当前队列（EmbSpatial→CrossPoint→RoboSpatial）完成后，
# 依次跑 Q-Spatial → MSMU → ShareRobot → VABench。
# 单进程顺序执行，产物与判分逐题落盘；每个阶段有独立结束标记。
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONPATH="$WS/envs/pylibs_core:$WS/third_party/upstream_eval/FlagEvalMM:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore
export TMPDIR="$WS/tmp"
set +u; source /opt/dtk/env.sh >/dev/null 2>&1; set -u
OUT="$WS/outputs/bench_public_noneager2"
LOG="$WS/logs/master_bench_queue.log"
mkdir -p "$OUT"

log() { echo "[$(date '+%F %T')] $*"; }

# 等第二批（RoboSpatial）结束标记，最多等 4 小时
for i in $(seq 1 480); do
  [ -f "$WS/tmp/newbenches_2.end.ts" ] && break
  sleep 30
done
if [ ! -f "$WS/tmp/newbenches_2.end.ts" ]; then
  log "ERROR: 等第二批队列超时，主驱动退出"
  exit 1
fi
log "前两批完成，开始第三批"

run_stage() {
  local name="$1"; shift
  log "===== stage $name start ====="
  "$@" 2>&1 | tail -n 25
  local rc=${PIPESTATUS[0]}
  log "===== stage $name rc=$rc ====="
  date '+%s' > "$WS/tmp/stage_${name}.end.ts"
}

run_stage qspatial python3 submission/bench_public/run_qspatial.py \
  --out-dir "$OUT/qspatial" --run-id noneager-qspatial --concurrency 16 --max-tokens 1024

run_stage msmu python3 submission/bench_public/run_msmu.py \
  --out-dir "$OUT/msmu" --run-id noneager-msmu --concurrency 16

run_stage sharerobot python3 submission/bench_public/run_sharerobot.py \
  --out-dir "$OUT/sharerobot" --run-id noneager-sharerobot --concurrency 16

run_stage vabench python3 submission/bench_public/run_vabench.py \
  --out-dir "$OUT/vabench" --run-id noneager-vabench --concurrency 16

log "第三批全部结束"
date '+%s' > "$WS/tmp/master_queue.end.ts"
