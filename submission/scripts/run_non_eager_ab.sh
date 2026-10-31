#!/bin/bash
# =============================================================================
# Non-eager A/B: quantify the throughput cost of --enforce-eager.
#
# Why a watchdog
# --------------
# The first attempt to start vLLM WITHOUT --enforce-eager (logs/vllm_try3_*)
# hit the container memory ceiling (115 GiB cgroup limit) during inductor
# compilation and the OOM killer destroyed the container -> the Pod was
# recreated and the node was unreachable for five hours.
#
# This script therefore never lets the server exceed a configurable cgroup
# usage threshold (default 70 GiB): a watchdog polls the cgroup every 3 s and
# SIGKILLs the server before the kernel OOM killer can act. An aborted run is
# a legitimate, reportable outcome ("non-eager does not fit this container"),
# not a failure of the harness.
#
# Usage:
#   bash scripts/run_non_eager_ab.sh [--mem-limit-gib 70] [--mem-frac 0.85]
#                                     [--ctx 16384] [--port 8000]
# =============================================================================
set -o pipefail

WORK_DIR="${WORK_DIR:-/root/private_data/workspace/robo}"
MODEL_PATH="${MODEL_PATH:-/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV}"
MEM_LIMIT_GIB=70
MEM_FRAC=0.85
CTX=16384
PORT=8000
MODEL_NAME=robobrain2.5-8b-nv
TAG="noneager_$(date +%Y%m%d_%H%M%S)"

while [ $# -gt 0 ]; do
  case "$1" in
    --mem-limit-gib) MEM_LIMIT_GIB="$2"; shift 2 ;;
    --mem-frac) MEM_FRAC="$2"; shift 2 ;;
    --ctx) CTX="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

cd "$WORK_DIR" || exit 1
mkdir -p logs outputs

export WORK_DIR MODEL_PATH
export HF_HOME="$WORK_DIR/cache/hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$WORK_DIR/envs/pylibs:${PYTHONPATH:-}"
export VLLM_LOGGING_LEVEL=WARNING
source /opt/dtk/env.sh

LOG="logs/vllm_${TAG}.log"
LIMIT_BYTES=$((MEM_LIMIT_GIB * 1024 * 1024 * 1024))
CGROUP_USAGE=/sys/fs/cgroup/memory/memory.usage_in_bytes
CGROUP_LIMIT=/sys/fs/cgroup/memory/memory.limit_in_bytes

echo "=== non-eager A/B ${TAG} ==="
echo "cgroup limit: $(cat "$CGROUP_LIMIT" 2>/dev/null) bytes"
echo "watchdog threshold: ${LIMIT_BYTES} bytes (${MEM_LIMIT_GIB} GiB)"

pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null
pkill -f "serve_vllm_patched.py" 2>/dev/null
sleep 5

nohup python3 scripts/serve_vllm_patched.py \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization "$MEM_FRAC" \
  --max-model-len "$CTX" \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 32 \
  --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"image": 1}' \
  > "$LOG" 2>&1 &
SERVER_PID=$!
echo "[ab] server pid=${SERVER_PID} (NO --enforce-eager)"

WATCHDOG_PID=""
(
  PEAK=0
  while kill -0 "$SERVER_PID" 2>/dev/null; do
    U=$(cat "$CGROUP_USAGE" 2>/dev/null || echo 0)
    if [ "$U" -gt "$PEAK" ]; then PEAK=$U; fi
    if [ "$U" -gt "$LIMIT_BYTES" ]; then
      echo "[watchdog] cgroup usage ${U} exceeded ${LIMIT_BYTES}; SIGKILL server" >> "$LOG"
      kill -9 "$SERVER_PID" 2>/dev/null
      break
    fi
    sleep 3
  done
  echo "[watchdog] peak_cgroup_bytes=${PEAK}" >> "$LOG"
) &
WATCHDOG_PID=$!

READY=0
for i in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    READY=1
    echo "[ab] server ready after $((i*10))s"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[ab] SERVER EXITED before becoming ready"
    break
  fi
  sleep 10
done

if [ "$READY" = "1" ]; then
  echo "[ab] running the same 6-sample probe used for the eager measurement"
  python3 scripts/probe_http_request.py --backend vllm --port "$PORT" \
    --tasks samples/tasks_official.json --image-dir samples/images \
    --repeat 1 --override-max-tokens 768 \
    --out-dir "outputs/phase5_probe_${TAG}" 2>&1 | tail -25
  python3 scripts/compare_backends.py \
    --baseline outputs/phase3_baseline/baseline_raw.json \
    --candidate "outputs/phase5_probe_${TAG}/probe_result.json" \
    --out "outputs/phase5_probe_${TAG}/agreement.md" 2>&1 | tail -20
else
  echo "[ab] RESULT: non-eager did NOT fit the container"
  tail -30 "$LOG"
fi

kill "$WATCHDOG_PID" 2>/dev/null
if [ "$READY" = "1" ]; then
  kill "$SERVER_PID" 2>/dev/null
  for _ in $(seq 1 30); do
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 1
  done
  kill -9 "$SERVER_PID" 2>/dev/null
fi
echo "=== non-eager A/B ${TAG} finished (ready=${READY}) ==="
