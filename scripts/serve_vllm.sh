#!/usr/bin/env bash
# Phase 4: vLLM serving path (comparison backend) for RoboBrain2.5-8B-NV on K100AI.
#
#   bash scripts/serve_vllm.sh [--port 8000] [--mem-frac 0.85] [--ctx 8192]
# NOTE: do not use 'set -u' here - /opt/dtk/env.sh references unset vars and would abort.
set -o pipefail
cd "$(dirname "$0")/.."
source ./env.sh

PORT=8000
MEM_FRAC=0.85
CTX=8192
EXTRA_ARGS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --port)     PORT="$2"; shift 2 ;;
    --mem-frac) MEM_FRAC="$2"; shift 2 ;;
    --ctx)      CTX="$2"; shift 2 ;;
    --extra)    EXTRA_ARGS="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

TS="$(date +%Y%m%d_%H%M%S)"
LOG="${WORK_DIR}/logs/vllm_${TS}.log"
mkdir -p "${WORK_DIR}/logs"

echo "vllm log: ${LOG}"
exec python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name robobrain2.5-8b-nv \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization "${MEM_FRAC}" \
  --max-model-len "${CTX}" \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image": 8}' \
  ${EXTRA_ARGS} 2>&1 | tee -a "${LOG}"
