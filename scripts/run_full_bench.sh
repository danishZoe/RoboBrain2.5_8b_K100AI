#!/usr/bin/env bash
# =============================================================================
# 完整压测驱动：官方六档 × N 轮，单服务实例内连续完成
#
# 口径（与 submission/bench、10_phase5_bench.md 一致）：
#   单卡 K100AI（gfx928 / 63.98 GiB）/ vLLM 0.15.1+das.opt1.alpha.dtk2604 / BF16 / TP=1
#   --gpu-memory-utilization 0.85 --max-model-len 32768
#   --max-num-batched-tokens 4096 --max-num-seqs 64
#   --no-enable-prefix-caching --limit-mm-per-prompt '{"image": 8}'
#   graph 模式 = 不加 --enforce-eager（torch.compile + CUDA/HIP graph capture）
#   客户端 = submission/scripts/bench_serving_vllm_image.py 适配的官方 bench_serving
#   负载 = image 数据集，1×1080p 图/请求，--random-output-len 1024，--seed 1
#   请求数 = max(8, 2×并发)
#
# 配置：
#   candidate = graph + GEMM 调优库(ROCBLAS_TENSILE_LIBPATH) + ROCM_AITER_FA
#   baseline  = graph + stock rocBLAS 库（graph 基线，不加任何调优开关）
#
# 用法：
#   bash scripts/run_full_bench.sh --config candidate --tiers "4096:1:2,4096:8:2" --rounds 2
#   tiers 语法 = 输入长度:并发:轮数（轮数省略时取 --rounds）
# =============================================================================
set -uo pipefail

WS="${WS:-/root/private_data/workspace/robo}"
MODEL_PATH="${MODEL_PATH:-/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV}"
MODEL_NAME=robobrain2.5-8b-nv
PORT="${PORT:-30000}"
CONFIG=candidate
ROUNDS=2
TIERS="4096:1:2,4096:8:2,16384:1:2,16384:8:2,4096:64:2,16384:64:1"
SKIP_SMOKE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --tiers) TIERS="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

cd "$WS" || exit 1
RUN_ID="fullbench_${CONFIG}_$(date +%Y%m%d_%H%M%S)"
OUT="$WS/outputs/$RUN_ID"
mkdir -p "$OUT"

# --- DTK 运行时（必须先加载：压测客户端要 import torch，缺 libgalaxyhip 会 ImportError) ---
# /opt/dtk/env.sh 引用了未定义变量，不能在 set -u 下 source
set +u
# shellcheck disable=SC1091
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

export HF_HOME="$WS/cache/hf"
export HUGGINGFACE_HUB_CACHE="$WS/cache/hf/hub"
export TRANSFORMERS_CACHE="$WS/cache/hf/transformers"
export TORCH_HOME="$WS/cache/torch"
export XDG_CACHE_HOME="$WS/cache/xdg"
export PIP_CACHE_DIR="$WS/cache/pip"
export TMPDIR="$WS/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$WS/envs/pylibs:${PYTHONPATH:-}"
export VLLM_LOGGING_LEVEL=WARNING

echo "=== 完整压测 ${RUN_ID} ==="
echo "config=${CONFIG} tiers=${TIERS} rounds=${ROUNDS} port=${PORT}"
echo "started_at=$(date -Is)"

# --- 安全闸：本任务不与他人抢卡 -------------------------------------------------
if pgrep -f "bench_serving" >/dev/null 2>&1; then
  echo "[guard] 已有压测进程运行，拒绝启动" >&2
  exit 3
fi
if pgrep -f "$WS/submission/scripts/serve_vllm_patched.py" >/dev/null 2>&1; then
  echo "[guard] 已有服务进程运行，拒绝启动" >&2
  exit 4
fi

# --- 记录配置指纹 --------------------------------------------------------------
{
  echo "run_id=${RUN_ID}"
  echo "config=${CONFIG}"
  echo "port=${PORT}"
  echo "started_at=$(date -Is)"
  echo "tiers=${TIERS}"
  echo "rounds=${ROUNDS}"
  echo "model=${MODEL_PATH}"
  echo "git_revision=$(git -C "$WS" rev-parse HEAD 2>/dev/null)"
  echo "vllm_version=$(python3 -c 'import vllm;print(vllm.__version__)' 2>/dev/null)"
  echo "rocblas_stock_ai_dat=$(sha256sum /opt/dtk/lib/rocblas/library_gpu4/GemmGenerality_AI_Eco.dat 2>/dev/null | cut -d' ' -f1)"
  echo "candidate_ai_dat=$(sha256sum "$WS/tools/auto_select_tools/optimization_configs/new/config/library_gpu4/GemmGenerality_AI_Eco.dat" 2>/dev/null | cut -d' ' -f1)"
  echo "candidate_add_dat=$(sha256sum "$WS/tools/auto_select_tools/optimization_configs/new/config/library_gpu4/GemmGeneralityAdd_Eco.dat" 2>/dev/null | cut -d' ' -f1)"
} > "$OUT/run_config.txt"

# --- 启动服务 ------------------------------------------------------------------
case "$CONFIG" in
  candidate)
    # GEMM 调优库 + ROCM_AITER_FA + graph（不加 --enforce-eager）
    bash "$WS/scripts/start_aiterfa_service.sh" 2>&1 | tee "$OUT/service_start.log"
    ;;
  baseline)
    # graph 基线：stock rocBLAS 库，无任何调优开关
    bash "$WS/tools/start_bench_service.sh" graph "$PORT" 2>&1 | tee "$OUT/service_start.log"
    ;;
  *)
    echo "unknown config: $CONFIG" >&2
    exit 2
    ;;
esac
if ! curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "[fullbench] 服务未就绪，终止" >&2
  exit 5
fi
echo "[fullbench] service ready"

# 记录 worker 实际环境（调优开关是否真正进入推理进程）
{
  echo "--- worker env (ROCBLAS/VLLM) ---"
  for pid in $(pgrep -f "serve_vllm_patched.py"); do
    echo "pid=${pid}"
    tr '\0' '\n' < "/proc/${pid}/environ" 2>/dev/null | grep -E '^(ROCBLAS|VLLM_ATTENTION|VLLM_ROCM|HIP_|VLLM_LOGGING)' || true
  done
} > "$OUT/worker_env.txt"

# --- 六档 × N 轮 ---------------------------------------------------------------
IFS=',' read -ra TIER_ARR <<< "$TIERS"
for T in "${TIER_ARR[@]}"; do
  ILEN="${T%%:*}"; REST="${T#*:}"; C="${REST%%:*}"
  NR="${REST##*:}"; [ "$NR" = "$C" ] && NR="$ROUNDS"
  N=$(( C * 2 > 8 ? C * 2 : 8 ))
  for R in $(seq 1 "$NR"); do
    TAG="round${R}_${ILEN}_1024_c${C}"
    echo "[fullbench] $(date -Is)] ${TAG} prompts=${N}"
    python3 "$WS/submission/scripts/bench_serving_vllm_image.py" \
      --backend vllm --host 127.0.0.1 --port "$PORT" \
      --model "$MODEL_NAME" --tokenizer "$MODEL_PATH" \
      --dataset-name image --image-count 1 --image-resolution 1080p \
      --random-input-len "$ILEN" --random-output-len 1024 --random-range-ratio 1 \
      --max-concurrency "$C" --num-prompts "$N" --seed 1 \
      --output-file "$OUT/${TAG}.jsonl" \
      > "$OUT/${TAG}.log" 2>&1
    RC=$?
    echo "[fullbench] $(date -Is)] ${TAG} exit=${RC}"
    grep -E 'Successful requests|Failed requests|Benchmark duration|Output token throughput|Mean TTFT|Mean TPOT|Mean E2E' \
      "$OUT/${TAG}.log" | sed 's/^/      /'
  done
done

# --- 功能冒烟（12 项）+ 同口径探针 ---------------------------------------------
if [ "$SKIP_SMOKE" != "1" ]; then
  echo "[fullbench] 服务级 Smoke"
  python3 "$WS/scripts/smoke_service.py" --url "http://127.0.0.1:${PORT}" \
    --model "$MODEL_NAME" --out-dir "$OUT/smoke" > "$OUT/smoke.log" 2>&1
  tail -20 "$OUT/smoke.log"
  python3 "$WS/scripts/probe_http_request.py" --backend vllm --port "$PORT" \
    --tasks "$WS/samples/tasks_official.json" --image-dir "$WS/samples/images" \
    --repeat 1 --override-max-tokens 768 \
    --out-dir "$OUT/probe" > "$OUT/probe.log" 2>&1
  tail -8 "$OUT/probe.log"
  python3 "$WS/scripts/compare_backends.py" \
    --baseline "$WS/outputs/phase3_baseline/baseline_raw.json" \
    --candidate "$OUT/probe/probe_result.json" \
    --out "$OUT/agreement.md" >> "$OUT/probe.log" 2>&1
fi

# --- 停服务 -------------------------------------------------------------------
echo "[fullbench] 停止服务"
pkill -f "$WS/submission/scripts/serve_vllm_patched.py" 2>/dev/null
pkill -f "VLLM::EngineCore" 2>/dev/null
sleep 8
pkill -9 -f "$WS/submission/scripts/serve_vllm_patched.py" 2>/dev/null

# --- 汇总 ---------------------------------------------------------------------
python3 "$WS/scripts/summarize_full_bench.py" --run-dir "$OUT" --config "$CONFIG" \
  > "$OUT/summary.md" 2>&1
echo "finished_at=$(date -Is)" >> "$OUT/run_config.txt"
cat "$OUT/summary.md"
echo "=== done: $OUT ==="
