#!/usr/bin/env bash
# 启动「最优版」服务：graph(非 eager) + GEMM 算子调优库 + ROCM_AITER_FA
#
# 用法：bash scripts/start_aiterfa_service.sh [port]
#   默认端口 30000（与 tools/start_bench_service.sh 一致）
#
# ★ 调优库路径必须用**交付包组装出来的** $WS/rocblas_tuned/library_gpu4：
#   旧写法指向 $WS/tools/auto_select_tools/optimization_configs/new/config/library_gpu4，
#   那是调优现场目录（8 GB，已被 .gitignore 排除），换节点/新容器里**不一定存在**。
#   路径不存在时 rocBLAS 会静默回退到默认 kernel，最优版会退化成 graph 基线，
#   压测结论直接失真。因此这里：先组装 → 再校验 6 个差异文件的 SHA256 → 不符即退出。
set -o pipefail
WS="${WS:-/root/private_data/workspace/robo}"
PORT="${1:-30000}"
cd "$WS" || exit 1

MODEL=/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV

export HF_HOME="$WS/cache/hf"
export HUGGINGFACE_HUB_CACHE="$WS/cache/hf/hub"
export TRANSFORMERS_CACHE="$WS/cache/hf/transformers"
export TORCH_HOME="$WS/cache/torch"
export XDG_CACHE_HOME="$WS/cache/xdg"
export PIP_CACHE_DIR="$WS/cache/pip"
export TMPDIR="$WS/tmp"
export VLLM_LOGGING_LEVEL=WARNING
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$WS/tmp" "$WS/logs" "$WS/outputs"

set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

# --- 1) 组装调优库（幂等；失败即退出，不允许带病启动）---------------------------
if [ -f "$WS/submission/scripts/make_tuned_rocblas_lib.sh" ]; then
  if ! bash "$WS/submission/scripts/make_tuned_rocblas_lib.sh" \
        > "$WS/logs/make_tuned_rocblas_lib.log" 2>&1; then
    echo "[start] ERROR: 调优库组装/校验失败，拒绝启动（详见 logs/make_tuned_rocblas_lib.log）" >&2
    tail -n 20 "$WS/logs/make_tuned_rocblas_lib.log" >&2
    exit 1
  fi
  echo "[start] 调优库组装/校验通过"
fi

# --- 2) 选定库目录 -------------------------------------------------------------
NEWLIB="${TUNED_ROCBLAS_DIR:-$WS/rocblas_tuned/library_gpu4}"
if [ ! -d "$NEWLIB" ]; then
  NEWLIB="$WS/tools/auto_select_tools/optimization_configs/new/config/library_gpu4"
fi
if [ ! -d "$NEWLIB" ]; then
  echo "[start] ERROR: 找不到调优库目录" >&2
  echo "        期望：$WS/rocblas_tuned/library_gpu4" >&2
  echo "        或  ：$WS/tools/auto_select_tools/optimization_configs/new/config/library_gpu4" >&2
  echo "        请先执行：bash submission/scripts/make_tuned_rocblas_lib.sh" >&2
  exit 1
fi

# --- 3) 校验 6 个差异文件（防止库被改坏 / 路径指错）---------------------------
EXPECTED="GemmGeneralityAdd_Eco.dat 23e30c02a186d0c9a0cc851c2874752f1fc3f7fdd89ba77bf2cd5c5806324f58
GemmGeneralityAdd_SolutionName_Eco.txt 8b5b348f95a93ab7f852a816389cc2e34f48806a96dd02b5a4842c035082ccd9
GemmGenerality_AI_Eco.dat ad891f317f5c097776e014d738f17e84d2e4af7dfc7ad352f01f0d60d28025f3
GemmGenerality_SolutionName_AI_Eco.txt 5509bf83b23f961c17b9a60021a2be070f431835b8920f34cedeac079697b1c3
solutionIndexAdd_Eco.txt 3639767ab465c78e729e56850ce0ccbe2229d84f1dfaa7bb059bc8c62b43da79
solutionIndex_AI_Eco.txt 6ec0a9d84da70e5a2eeb0fa1dffe06c9f70feaf8defdc1d8209d9c09aa2d736c"
rc=0
while read -r fname fsha; do
  [ -z "${fname:-}" ] && continue
  if [ ! -f "$NEWLIB/$fname" ]; then
    echo "[start] ERROR: 调优库缺少 $fname" >&2; rc=1; continue
  fi
  cur=$(sha256sum "$NEWLIB/$fname" | cut -d' ' -f1)
  if [ "$cur" != "$fsha" ]; then
    echo "[start] ERROR: 调优库哈希不符 $fname 期望=$fsha 实际=$cur" >&2; rc=1
  else
    echo "[start] OK $fname $cur"
  fi
done <<< "$EXPECTED"
if [ "$rc" != "0" ]; then
  echo "[start] 调优库校验失败，拒绝启动" >&2
  exit 1
fi

# --- 4) 生效开关 ---------------------------------------------------------------
# GEMM 调优产物：只在工作目录内，不触碰 /opt/dtk 原库
export ROCBLAS_TENSILE_LIBPATH="$NEWLIB"
export VLLM_ATTENTION_BACKEND=ROCM_AITER_FA
# 注意：不要设置 VLLM_ROCM_USE_AITER=1（本镜像下会在 ViT 的
# mm_encoder_attention flash op 崩溃，见 reports/23 §2 E8）。

LOG="$WS/logs/serve_graph_aiterfa_$(date +%Y%m%d-%H%M%S).log"
echo "[start] tuned lib=$NEWLIB (files=$(ls -1 "$NEWLIB" | wc -l))"
echo "[start] ROCBLAS_TENSILE_LIBPATH=$ROCBLAS_TENSILE_LIBPATH"
echo "[start] VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND"
echo "[start] port=$PORT log=$LOG"

setsid nohup python3 "$WS/submission/scripts/serve_vllm_patched.py" \
  --model "$MODEL" \
  --served-model-name robobrain2.5-8b-nv \
  --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --dtype bfloat16 --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 64 \
  --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"image": 8}' \
  > "$LOG" 2>&1 < /dev/null &

disown || true
sleep 2
echo $! > "$WS/outputs/.bench_server_tunedlib.pid"
echo "[start] pid=$(cat "$WS/outputs/.bench_server_tunedlib.pid")"

for i in $(seq 1 90); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 \
        "http://127.0.0.1:${PORT}/health" || echo 000)
  if [ "$code" = "200" ]; then
    echo "[start] READY after ${i}0s"
    exit 0
  fi
  sleep 10
done
echo "[start] TIMEOUT; tail log:"; tail -n 20 "$LOG"
exit 1
