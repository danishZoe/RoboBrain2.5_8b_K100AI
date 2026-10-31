#!/bin/bash
# =============================================================================
# Phase 5 driver: canonicalisation A/B + service smoke + six-tier load bench.
#
#   bash scripts/run_phase5.sh [--mem-frac 0.85] [--ctx 20480] [--skip-bench]
#
# Steps
#   0. stop any leftover vLLM server, start a fresh one with the recorded flags
#   1. probe with the official qwen_vl_utils preprocessing applied client side
#   2. probe with the original image uploaded (the pre-fix behaviour)
#   3. compare both against the transformers baseline
#   4. service smoke (4 task heads, multi-image, streaming, multi-turn, bad input)
#   5. six-tier load bench (4096/16384 input tokens x concurrency 1/8/64)
#
# Everything is written under $WORK_DIR. The script is re-runnable.
# =============================================================================
set -o pipefail

WORK_DIR="${WORK_DIR:-/root/private_data/workspace/robo}"
MODEL_PATH="${MODEL_PATH:-/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV}"
MEM_FRAC=0.85
CTX=20480
SKIP_BENCH=0
BENCH_ONLY=0
SERVE_ONLY=0
KEEP_SERVER=0
BENCH_TIERS=""
PORT=8000
MODEL_NAME=robobrain2.5-8b-nv
TAG="p5_$(date +%Y%m%d_%H%M%S)"

while [ $# -gt 0 ]; do
  case "$1" in
    --mem-frac) MEM_FRAC="$2"; shift 2 ;;
    --ctx) CTX="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --skip-bench) SKIP_BENCH=1; shift ;;
    --bench-only) BENCH_ONLY=1; shift ;;
    --serve-only) SERVE_ONLY=1; shift ;;
    --keep-server) KEEP_SERVER=1; shift ;;
    --tiers) BENCH_TIERS="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

cd "$WORK_DIR" || exit 1
mkdir -p logs outputs tmp

export WORK_DIR MODEL_PATH
export HF_HOME="$WORK_DIR/cache/hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$WORK_DIR/envs/pylibs:${PYTHONPATH:-}"
export VLLM_LOGGING_LEVEL=WARNING
# DTK env.sh must be sourced without `set -u` (it references unset variables).
source /opt/dtk/env.sh

SERVER_LOG="logs/vllm_${TAG}.log"
SERVER_PID=""

echo "=== phase5 ${TAG} | mem-frac=${MEM_FRAC} ctx=${CTX} port=${PORT} ==="

cleanup() {
  if [ "$KEEP_SERVER" = "1" ]; then
    echo "[phase5] keeping server pid=${SERVER_PID} alive (--keep-server)"
    return
  fi
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[phase5] stopping server pid=${SERVER_PID}"
    kill "$SERVER_PID" 2>/dev/null
    for _ in $(seq 1 30); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$SERVER_PID" 2>/dev/null
  fi
}
trap cleanup EXIT

# --- 0. fresh server -------------------------------------------------------
if [ "$BENCH_ONLY" = "0" ]; then
echo "[phase5] killing leftover servers"
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null
pkill -f "serve_vllm_patched.py" 2>/dev/null
sleep 5

echo "[phase5] starting vLLM (log: ${SERVER_LOG})"
nohup python3 scripts/serve_vllm_patched.py \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization "$MEM_FRAC" \
  --max-model-len "$CTX" \
  --enforce-eager \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 64 \
  --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"image": 8}' \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[phase5] server pid=${SERVER_PID}"

READY=0
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    READY=1
    echo "[phase5] server ready after ${i}0s"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[phase5] SERVER DIED, tail of log:"
    tail -60 "$SERVER_LOG"
    exit 1
  fi
  sleep 10
done
if [ "$READY" != "1" ]; then
  echo "[phase5] server NOT ready, tail of log:"
  tail -60 "$SERVER_LOG"
  exit 1
fi

curl -s "http://127.0.0.1:${PORT}/v1/models" | head -c 400; echo
grep -iE "KV cache size|GPU KV cache size|maximum concurrency" "$SERVER_LOG" | tail -5
fi

if [ "$SERVE_ONLY" = "1" ]; then
  echo "[phase5] serve-only mode: server left running (pid=${SERVER_PID})"
  exit 0
fi

# --- 1/2. canonicalisation A/B ---------------------------------------------
echo "[phase5] probe WITH canonical preprocessing"
python3 scripts/probe_http_request.py --backend vllm --port "$PORT" \
  --tasks samples/tasks_official.json --image-dir samples/images \
  --repeat 1 --override-max-tokens 768 \
  --out-dir outputs/phase5_probe_canonical 2>&1 | tail -20

echo "[phase5] probe WITHOUT canonical preprocessing (pre-fix behaviour)"
python3 scripts/probe_http_request.py --backend vllm --port "$PORT" \
  --tasks samples/tasks_official.json --image-dir samples/images \
  --repeat 1 --override-max-tokens 768 --no-canonical \
  --out-dir outputs/phase5_probe_nocanonical 2>&1 | tail -20

# --- 3. compare against the transformers baseline ---------------------------
for d in outputs/phase5_probe_canonical outputs/phase5_probe_nocanonical; do
  if [ -f "$d/probe_result.json" ]; then
    python3 scripts/compare_backends.py \
      --baseline outputs/phase3_baseline/baseline_raw.json \
      --candidate "$d/probe_result.json" \
      --out "$d/agreement.md" 2>&1 | tail -25
  fi
done

# --- 4. service smoke -------------------------------------------------------
echo "[phase5] service smoke"
python3 scripts/smoke_service.py --url "http://127.0.0.1:${PORT}" \
  --model "$MODEL_NAME" --out-dir outputs/phase5_smoke 2>&1 | tail -40

# --- 5. six-tier load bench -------------------------------------------------
if [ "$BENCH_ONLY" = "1" ]; then
  echo "[phase5] bench-only mode, tiers=${BENCH_TIERS:-default}"
  TIERS_ARG=""
  if [ -n "$BENCH_TIERS" ]; then
    TIERS_ARG="--tiers $BENCH_TIERS"
  fi
  python3 scripts/bench_service.py --url "http://127.0.0.1:${PORT}/v1/chat/completions" \
    --model "$MODEL_NAME" --rounds 2 $TIERS_ARG \
    --out-dir outputs/phase5_bench 2>&1 | tail -80
  echo "=== phase5 ${TAG} bench finished ==="
  exit 0
fi

if [ "$SKIP_BENCH" = "0" ]; then
  echo "[phase5] six-tier load bench (this takes a while)"
  python3 scripts/bench_service.py --url "http://127.0.0.1:${PORT}/v1/chat/completions" \
    --model "$MODEL_NAME" --rounds 2 --out-dir outputs/phase5_bench 2>&1 | tail -60
fi

echo "=== phase5 ${TAG} finished ==="
