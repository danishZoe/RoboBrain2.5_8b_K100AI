#!/usr/bin/env bash
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
# SIGKILLs the **whole process group** (the launcher alone is not enough: the
# EngineCore child keeps allocating) before the kernel OOM killer can act.
# An aborted run is a legitimate, reportable outcome ("non-eager does not fit
# this container"), not a failure of the harness.
#
# Strict single-variable A/B: every server flag except --enforce-eager is
# identical to the eager baseline recorded in submission/bench/run_config.txt.
#
# Usage:
#   bash scripts/run_non_eager_ab.sh [--mem-limit-gib 70] [--mem-frac 0.85]
#                                     [--ctx 32768] [--port 8000]
#                                     [--tiers "4096:1,4096:8"]
# =============================================================================
set -o pipefail

WORK_DIR="${WORK_DIR:-/root/private_data/workspace/robo}"
MODEL_PATH="${MODEL_PATH:-/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV}"
MEM_LIMIT_GIB=70
MEM_FRAC=0.85
CTX=32768
PORT=8000
MODEL_NAME=robobrain2.5-8b-nv
MAX_BATCHED_TOKENS=4096
MAX_NUM_SEQS=64
TIERS="4096:1,4096:8"
TAG="noneager_$(date +%Y%m%d_%H%M%S)"

while [ $# -gt 0 ]; do
  case "$1" in
    --mem-limit-gib) MEM_LIMIT_GIB="$2"; shift 2 ;;
    --mem-frac) MEM_FRAC="$2"; shift 2 ;;
    --ctx) CTX="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --tiers) TIERS="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

cd "$WORK_DIR" || exit 1
mkdir -p logs outputs/phase5_ab_noneager

export WORK_DIR MODEL_PATH
export HF_HOME="$WORK_DIR/cache/hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$WORK_DIR/envs/pylibs:${PYTHONPATH:-}"
export VLLM_LOGGING_LEVEL=WARNING
set +u
source /opt/dtk/env.sh
set -u

LOG="logs/vllm_${TAG}.log"
LIMIT_BYTES=$((MEM_LIMIT_GIB * 1024 * 1024 * 1024))
CGROUP_USAGE=/sys/fs/cgroup/memory/memory.usage_in_bytes
CGROUP_LIMIT=/sys/fs/cgroup/memory/memory.limit_in_bytes
OUTDIR="outputs/phase5_ab_noneager/${TAG}"

echo "=== non-eager A/B ${TAG} ==="
echo "cgroup limit:      $(cat "$CGROUP_LIMIT" 2>/dev/null) bytes"
echo "watchdog threshold:${LIMIT_BYTES} bytes (${MEM_LIMIT_GIB} GiB)"
echo "server flags: identical to eager baseline EXCEPT --enforce-eager is absent"
echo "  ctx=${CTX} mem_frac=${MEM_FRAC} max_batched_tokens=${MAX_BATCHED_TOKENS} max_seqs=${MAX_NUM_SEQS} port=${PORT}"

# 安全闸：官方六档压测还在跑时禁止启动本脚本（它会杀掉服务，毁掉基准数据）
if pgrep -f 'bash bench.sh' >/dev/null 2>&1 || pgrep -f 'bench_serving' >/dev/null 2>&1; then
  echo "[ab] ABORT: 官方压测仍在运行，拒绝启动（避免杀掉服务毁掉基准数据）" >&2
  exit 3
fi

# 只清本任务的进程
pkill -f "scripts/serve_vllm_patched.py" 2>/dev/null
pkill -f "VLLM::EngineCore" 2>/dev/null
sleep 6

# setsid: 服务自成进程组，看门狗可以整组杀掉
setsid nohup python3 scripts/serve_vllm_patched.py \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --dtype bfloat16 --trust-remote-code \
  --gpu-memory-utilization "$MEM_FRAC" \
  --max-model-len "$CTX" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"image": 8}' \
  > "$LOG" 2>&1 &
SERVER_PID=$!
PGID=$(ps -o pgid= -p "$SERVER_PID" 2>/dev/null | tr -d ' ')
echo "[ab] server pid=${SERVER_PID} pgid=${PGID} (NO --enforce-eager)"

(
  PEAK=0
  while kill -0 "$SERVER_PID" 2>/dev/null; do
    U=$(cat "$CGROUP_USAGE" 2>/dev/null || echo 0)
    if [ "$U" -gt "$PEAK" ]; then PEAK=$U; fi
    if [ "$U" -gt "$LIMIT_BYTES" ]; then
      echo "[watchdog] cgroup usage ${U} > ${LIMIT_BYTES}; SIGKILL process group ${PGID}" >> "$LOG"
      kill -9 -"$PGID" 2>/dev/null
      break
    fi
    sleep 3
  done
  echo "[watchdog] peak_cgroup_bytes=${PEAK}" >> "$LOG"
) &
WATCHDOG_PID=$!

READY=0
START=$(date +%s)
for i in $(seq 1 120); do
  if curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    READY=1
    echo "[ab] server ready after $(( $(date +%s) - START ))s"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[ab] SERVER EXITED before becoming ready"
    break
  fi
  sleep 10
done

if [ "$READY" = "1" ]; then
  echo "[ab] server up; running official-format client on tiers: ${TIERS}"
  mkdir -p "$OUTDIR"
  {
    echo "# non-eager A/B ${TAG}"
    echo "# server: no --enforce-eager, ctx=${CTX} mem_frac=${MEM_FRAC} max_batched=${MAX_BATCHED_TOKENS} max_seqs=${MAX_NUM_SEQS}"
    echo "# ready_after_s=$(( $(date +%s) - START ))"
  } > "$OUTDIR/run_config.txt"

  IFS=',' read -ra TIER_ARR <<< "$TIERS"
  for T in "${TIER_ARR[@]}"; do
    ILEN="${T%%:*}"; C="${T##*:}"
    N=$(( C * 2 > 8 ? C * 2 : 8 ))
    echo "[ab] tier in=${ILEN} conc=${C} prompts=${N}"
    python3 scripts/bench_serving_vllm_image.py \
      --backend vllm --host 127.0.0.1 --port "$PORT" \
      --model "$MODEL_NAME" --tokenizer "$MODEL_PATH" \
      --dataset-name image --image-count 1 --image-resolution 1080p \
      --random-input-len "$ILEN" --random-output-len 1024 --random-range-ratio 1 \
      --max-concurrency "$C" --num-prompts "$N" --seed 1 \
      --output-file "$OUTDIR/ab_${ILEN}_c${C}.jsonl" \
      > "$OUTDIR/ab_${ILEN}_c${C}.log" 2>&1
    grep -E 'Successful requests|Failed requests|Benchmark duration|Total generated tokens|Output token throughput|Mean TTFT|Mean TPOT|Mean E2E' \
      "$OUTDIR/ab_${ILEN}_c${C}.log" | sed 's/^/    /'
  done

  # 同口径一致性抽查（6 样例 / 768 token）
  python3 scripts/probe_http_request.py --backend vllm --port "$PORT" \
    --tasks samples/tasks_official.json --image-dir samples/images \
    --repeat 1 --override-max-tokens 768 \
    --out-dir "$OUTDIR/probe" 2>&1 | tail -15
  python3 scripts/compare_backends.py \
    --baseline outputs/phase3_baseline/baseline_raw.json \
    --candidate "$OUTDIR/probe/probe_result.json" \
    --out "$OUTDIR/agreement.md" 2>&1 | tail -15
else
  echo "[ab] RESULT: non-eager did NOT fit the container (watchdog threshold ${MEM_LIMIT_GIB} GiB)"
  tail -40 "$LOG"
fi

kill "$WATCHDOG_PID" 2>/dev/null
if [ "$READY" = "1" ]; then
  kill -"$PGID" 2>/dev/null
  sleep 3
  kill -9 -"$PGID" 2>/dev/null
fi
pkill -f "scripts/serve_vllm_patched.py" 2>/dev/null
pkill -f "VLLM::EngineCore" 2>/dev/null
echo "=== non-eager A/B ${TAG} finished (ready=${READY}) ==="
echo "=== peak cgroup usage ==="
grep 'peak_cgroup_bytes' "$LOG" || true
