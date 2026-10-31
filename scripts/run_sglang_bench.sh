#!/usr/bin/env bash
# Phase 4: start the SGLang server, measure it with the SAME fixed samples and the
# SAME measurement method as the transformers baseline, then shut the server down.
#
#   bash scripts/run_sglang_bench.sh [--port 30000] [--mem-frac 0.85] [--ctx 8192]
#                                    [--max-running-requests 16] [--tag sglang_base]
#                                    [--concurrency 1 4 8]
#
# The server is started in its own process group so it can be stopped reliably
# without touching any other process on the node.
# NOTE: do not use 'set -u' here - /opt/dtk/env.sh references unset vars and would abort.
set -o pipefail
cd "$(dirname "$0")/.."
source ./env.sh

PORT=30000
MEM_FRAC=0.85
CTX=8192
MAX_RUNNING=16
TAG=sglang_base
CONC="1 4 8"
EXTRA_ARGS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --port)                 PORT="$2"; shift 2 ;;
    --mem-frac)             MEM_FRAC="$2"; shift 2 ;;
    --ctx)                  CTX="$2"; shift 2 ;;
    --max-running-requests) MAX_RUNNING="$2"; shift 2 ;;
    --tag)                  TAG="$2"; shift 2 ;;
    --concurrency)          CONC="$2"; shift 2 ;;
    --extra)                EXTRA_ARGS="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

TS="$(date +%Y%m%d_%H%M%S)"
SRV_LOG="${WORK_DIR}/logs/sglang_server_${TS}.log"
BENCH_LOG="${WORK_DIR}/logs/${TAG}_${TS}.log"
mkdir -p "${WORK_DIR}/logs" "${WORK_DIR}/outputs/phase4"

cleanup() {
  echo "--- stopping sglang (pgid ${SRV_PID:-?}) ---"
  if [ -n "${SRV_PID:-}" ]; then
    kill -TERM -- "-${SRV_PID}" 2>/dev/null || kill -TERM "${SRV_PID}" 2>/dev/null
    sleep 8
    kill -KILL -- "-${SRV_PID}" 2>/dev/null || kill -KILL "${SRV_PID}" 2>/dev/null
  fi
  sleep 3
  echo "--- remaining sglang processes (should be none of ours) ---"
  ps -eo pid,cmd | grep "launch_server" | grep -v grep || echo "none"
}
trap cleanup EXIT

echo "server log : ${SRV_LOG}"
echo "bench log  : ${BENCH_LOG}"
echo "port=${PORT} mem_frac=${MEM_FRAC} ctx=${CTX} max_running=${MAX_RUNNING} conc=${CONC}"

setsid sglang serve \
  --model-path "${MODEL_PATH}" \
  --served-model-name robobrain2.5-8b-nv \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tp-size 1 \
  --dtype bfloat16 \
  --mem-fraction-static "${MEM_FRAC}" \
  --context-length "${CTX}" \
  --max-running-requests "${MAX_RUNNING}" \
  --limit-mm-data-per-request '{"image": 8}' \
  --trust-remote-code \
  ${EXTRA_ARGS} > "${SRV_LOG}" 2>&1 &
SRV_PID=$!
echo "sglang pgid=${SRV_PID}"

# give the server a chance to come up before the client starts polling
sleep 20
if ! kill -0 "${SRV_PID}" 2>/dev/null; then
  echo "sglang died during startup; last 80 log lines:"
  tail -80 "${SRV_LOG}"
  exit 1
fi

python3 scripts/bench_http.py \
  --backend sglang --port "${PORT}" \
  --tasks samples/tasks_official.json \
  --image-dir samples/images \
  --repeat 3 --concurrency ${CONC} \
  --out-dir outputs/phase4 2>&1 | tee "${BENCH_LOG}"
RC=${PIPESTATUS[0]}
echo "bench exit=${RC}"
exit ${RC}
