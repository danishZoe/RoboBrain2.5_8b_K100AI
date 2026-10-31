#!/usr/bin/env bash
# Phase 4: SGLang serving path for RoboBrain2.5-8B-NV on K100AI (HCU).
#
#   bash scripts/serve_sglang.sh [--port 30000] [--mem-frac 0.85] [--ctx 8192]
#                                [--max-running-requests 16] [--no-radix-cache]
#
# Flag names verified against the installed build
# (sglang 0.5.12+das.opt1.dtk2604, srt/server_args.py):
#   --mem-fraction-static        present
#   --context-length             present
#   --dtype / --tp-size          present
#   --served-model-name          present
#   --trust-remote-code          present
#   --limit-mm-data-per-request  present  (NOTE: NOT --limit-mm-per-prompt)
#   radix cache (prefix caching) is ON by default; the opt-out is
#   --disable-radix-cache. There is no --enable-prefix-caching in this build.
# NOTE: do not use 'set -u' here - /opt/dtk/env.sh references unset vars and would abort.
set -o pipefail
cd "$(dirname "$0")/.."
source ./env.sh

PORT=30000
MEM_FRAC=0.85
CTX=8192
MAX_RUNNING=16
RADIX_FLAG=""
MM_LIMIT='{"image": 8}'
EXTRA_ARGS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --port)                 PORT="$2"; shift 2 ;;
    --mem-frac)             MEM_FRAC="$2"; shift 2 ;;
    --ctx)                  CTX="$2"; shift 2 ;;
    --max-running-requests) MAX_RUNNING="$2"; shift 2 ;;
    --no-radix-cache)       RADIX_FLAG="--disable-radix-cache"; shift ;;
    --extra)                EXTRA_ARGS="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

TS="$(date +%Y%m%d_%H%M%S)"
LOG="${WORK_DIR}/logs/sglang_${TS}.log"
mkdir -p "${WORK_DIR}/logs"

echo "sglang log: ${LOG}"
echo "model=${MODEL_PATH} port=${PORT} mem_frac=${MEM_FRAC} ctx=${CTX} max_running=${MAX_RUNNING}"

# --dtype bfloat16 matches the checkpoint dtype and the HCU bf16 capability.
# --mem-fraction-static bounds the KV-cache pool so the card stays healthy.
# NOTE: the CLI entry point in sglang 0.5.12 is `sglang serve`
# (there is no `sglang.launch_server` executable in this build).
exec sglang serve \
  --model-path "${MODEL_PATH}" \
  --served-model-name robobrain2.5-8b-nv \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tp-size 1 \
  --dtype bfloat16 \
  --mem-fraction-static "${MEM_FRAC}" \
  --context-length "${CTX}" \
  --max-running-requests "${MAX_RUNNING}" \
  --limit-mm-data-per-request "${MM_LIMIT}" \
  --trust-remote-code \
  ${RADIX_FLAG} ${EXTRA_ARGS} 2>&1 | tee -a "${LOG}"
