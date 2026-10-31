#!/usr/bin/env bash
# Phase 4 (comparison backend): start the vLLM OpenAI server, measure it with the
# SAME fixed samples / method / aggregation as the transformers baseline and the
# SGLang run, then shut it down.
#
#   bash scripts/run_vllm_bench.sh [--port 8000] [--mem-frac 0.85] [--ctx 8192]
#                                 [--tag vllm_base] [--concurrency 1 4 8]
# NOTE: do not use 'set -u' here - /opt/dtk/env.sh references unset vars and would abort.
set -o pipefail
cd "$(dirname "$0")/.."
source ./env.sh

PORT=8000
MEM_FRAC=0.85
CTX=8192
TAG=vllm_base
CONC="1 4 8"
EXTRA_ARGS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --port)         PORT="$2"; shift 2 ;;
    --mem-frac)     MEM_FRAC="$2"; shift 2 ;;
    --ctx)          CTX="$2"; shift 2 ;;
    --tag)          TAG="$2"; shift 2 ;;
    --concurrency)  CONC="$2"; shift 2 ;;
    --extra)        EXTRA_ARGS="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

TS="$(date +%Y%m%d_%H%M%S)"
SRV_LOG="${WORK_DIR}/logs/vllm_server_${TS}.log"
BENCH_LOG="${WORK_DIR}/logs/${TAG}_${TS}.log"
mkdir -p "${WORK_DIR}/logs" "${WORK_DIR}/outputs/phase4"

cleanup() {
  echo "--- stopping vllm (pgid ${SRV_PID:-?}) ---"
  if [ -n "${SRV_PID:-}" ]; then
    kill -TERM -- "-${SRV_PID}" 2>/dev/null || kill -TERM "${SRV_PID}" 2>/dev/null
    sleep 10
    kill -KILL -- "-${SRV_PID}" 2>/dev/null || kill -KILL "${SRV_PID}" 2>/dev/null
  fi
  sleep 5
  echo "--- remaining vllm processes (should be none of ours) ---"
  ps -eo pid,cmd | grep "vllm.entrypoints" | grep -v grep || echo "none"
}
trap cleanup EXIT

echo "server log : ${SRV_LOG}"
echo "bench log  : ${BENCH_LOG}"
echo "port=${PORT} mem_frac=${MEM_FRAC} ctx=${CTX} conc=${CONC}"

setsid python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name robobrain2.5-8b-nv \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization "${MEM_FRAC}" \
  --max-model-len "${CTX}" \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image": 8}' \
  --enable-prefix-caching \
  ${EXTRA_ARGS} > "${SRV_LOG}" 2>&1 &
SRV_PID=$!
echo "vllm pgid=${SRV_PID}"

sleep 20
if ! kill -0 "${SRV_PID}" 2>/dev/null; then
  echo "vllm died during startup; last 40 log lines:"
  tail -40 "${SRV_LOG}"
  exit 1
fi

python3 scripts/bench_http.py \
  --backend vllm --port "${PORT}" --tag "${TAG}" \
  --tasks samples/tasks_official.json \
  --image-dir samples/images \
  --repeat 3 --concurrency ${CONC} \
  --out-dir outputs/phase4 2>&1 | tee "${BENCH_LOG}"
RC=${PIPESTATUS[0]}
echo "bench exit=${RC}"
exit ${RC}
