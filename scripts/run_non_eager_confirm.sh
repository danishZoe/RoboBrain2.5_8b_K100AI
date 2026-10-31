#!/usr/bin/env bash
# =============================================================================
# E1 确认实验：非 eager 配置的复跑与口径校验
#
# 背景：E1 首测发现非 eager（torch.compile + graph capture）在 **单并发**档
#       输出吞吐 +30.5%（TPOT −23.8%），但在 **并发 8** 档只有 +0.2%（落在波动内）。
#       按实验前登记的判定标准，单并发档需要「重跑确认」。
#
# 本脚本做四件事：
#   1) 用与 E1 完全相同的参数启动非 eager 服务（带 70 GiB 内存看门狗）
#   2) 在 4096/1、4096/8、16384/1、16384/8 四档上各跑一轮官方格式请求
#      （16384/c64 档单轮就要 ~2.2 h，不在本确认实验内，避免与后续流水线抢卡）
#   3) 跑**服务级 Smoke**（12 项）确认功能未因编译/graph 捕获而变化
#   4) 跑同口径探针并与 transformers 基线逐坐标比对，确认**输出逐字一致**
#
# 全程只动本任务自己的进程；结束后恢复基线服务（eager / 32768 / 30000）。
#
# 用法: bash scripts/run_non_eager_confirm.sh [--port 8000] [--tiers "4096:1,4096:8,16384:1,16384:8"]
# =============================================================================
set -uo pipefail

WORK_DIR="${WORK_DIR:-/root/private_data/workspace/robo}"
MODEL_PATH="${MODEL_PATH:-/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV}"
MEM_LIMIT_GIB=70
MEM_FRAC=0.85
CTX=32768
PORT=8000
MODEL_NAME=robobrain2.5-8b-nv
MAX_BATCHED_TOKENS=4096
MAX_NUM_SEQS=64
TIERS="4096:1,4096:8,16384:1,16384:8"
TAG="noneager_confirm_$(date +%Y%m%d_%H%M%S)"

while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --tiers) TIERS="$2"; shift 2 ;;
    --mem-limit-gib) MEM_LIMIT_GIB="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

cd "$WORK_DIR" || exit 1
OUTDIR="outputs/phase5_ab_noneager/${TAG}"
mkdir -p "$OUTDIR" logs

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

echo "=== E1 确认实验 ${TAG} ==="
echo "tiers: ${TIERS}"
echo "watchdog threshold: ${MEM_LIMIT_GIB} GiB"

# 安全闸：官方六档压测还在跑时禁止启动
if pgrep -f 'bash bench.sh' >/dev/null 2>&1 || pgrep -f 'bench_serving' >/dev/null 2>&1; then
  echo "[confirm] ABORT: 官方压测仍在运行，拒绝启动" >&2
  exit 3
fi

pkill -f "scripts/serve_vllm_patched.py" 2>/dev/null
pkill -f "VLLM::EngineCore" 2>/dev/null
sleep 6

setsid nohup python3 scripts/serve_vllm_patched.py \
  --model "$MODEL_PATH" --served-model-name "$MODEL_NAME" \
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
    echo "[confirm] server ready after $(( $(date +%s) - START ))s"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[confirm] SERVER EXITED before becoming ready"; break
  fi
  sleep 10
done

if [ "$READY" != "1" ]; then
  echo "[confirm] RESULT: 非 eager 未能就绪（见 $LOG）"
  kill "$WATCHDOG_PID" 2>/dev/null
  exit 1
fi

{
  echo "# non-eager 确认实验 ${TAG}"
  echo "# server: NO --enforce-eager, ctx=${CTX} mem_frac=${MEM_FRAC} max_batched=${MAX_BATCHED_TOKENS} max_seqs=${MAX_NUM_SEQS}"
  echo "# ready_after_s=$(( $(date +%s) - START ))"
} > "$OUTDIR/run_config.txt"

# --- 四档官方格式请求 --------------------------------------------------------
IFS=',' read -ra TIER_ARR <<< "$TIERS"
for T in "${TIER_ARR[@]}"; do
  ILEN="${T%%:*}"; C="${T##*:}"
  N=$(( C * 2 > 8 ? C * 2 : 8 ))
  echo "[confirm] tier in=${ILEN} conc=${C} prompts=${N}"
  python3 scripts/bench_serving_vllm_image.py \
    --backend vllm --host 127.0.0.1 --port "$PORT" \
    --model "$MODEL_NAME" --tokenizer "$MODEL_PATH" \
    --dataset-name image --image-count 1 --image-resolution 1080p \
    --random-input-len "$ILEN" --random-output-len 1024 --random-range-ratio 1 \
    --max-concurrency "$C" --num-prompts "$N" --seed 1 \
    --output-file "$OUTDIR/ab_${ILEN}_c${C}.jsonl" \
    > "$OUTDIR/ab_${ILEN}_c${C}.log" 2>&1
  grep -E 'Successful requests|Failed requests|Benchmark duration|Output token throughput|Mean TTFT|Mean TPOT' \
    "$OUTDIR/ab_${ILEN}_c${C}.log" | sed 's/^/      /'
done

# --- 服务级 Smoke（12 项）----------------------------------------------------
echo
echo "[confirm] 服务级 Smoke（12 项）"
python3 scripts/smoke_service.py --url "http://127.0.0.1:${PORT}" \
  --model "$MODEL_NAME" --out-dir "$OUTDIR/smoke" 2>&1 | tail -22

# --- 同口径探针 + 与 transformers 基线逐坐标比对 ------------------------------
echo
echo "[confirm] 同口径探针 + 与 transformers 基线比对"
python3 scripts/probe_http_request.py --backend vllm --port "$PORT" \
  --tasks samples/tasks_official.json --image-dir samples/images \
  --repeat 1 --override-max-tokens 768 \
  --out-dir "$OUTDIR/probe" 2>&1 | tail -12
python3 scripts/compare_backends.py \
  --baseline outputs/phase3_baseline/baseline_raw.json \
  --candidate "$OUTDIR/probe/probe_result.json" \
  --out "$OUTDIR/agreement.md" 2>&1 | tail -12

# --- 收尾：停服务，恢复基线服务 ----------------------------------------------
kill "$WATCHDOG_PID" 2>/dev/null
kill -"$PGID" 2>/dev/null
sleep 3
kill -9 -"$PGID" 2>/dev/null
pkill -f "scripts/serve_vllm_patched.py" 2>/dev/null
pkill -f "VLLM::EngineCore" 2>/dev/null
sleep 3

echo
echo "[confirm] 恢复基线服务（eager / 32768 / 30000）"
cd "$WORK_DIR/submission" || exit 1
set +u
source ./env.sh
set -u
setsid nohup bash run.sh > logs/run_stdout_restore.log 2>&1 &
for i in $(seq 1 90); do
  if curl -sf --noproxy '*' "http://127.0.0.1:${PORT_BASE:-30000}/health" >/dev/null 2>&1; then
    echo "[confirm] baseline service restored"
    break
  fi
  sleep 10
done

echo
echo "=== 确认实验产物 ==="
ls -l "$OUTDIR" | sed 's/^/  /'
grep 'peak_cgroup_bytes' "$LOG" || true
echo "=== done ==="
