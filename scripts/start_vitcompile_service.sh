#!/usr/bin/env bash
# 启动使用 GEMM 调优后 rocBLAS 库的 vLLM graph 服务（AB 验证用）
set -o pipefail
WS=/root/private_data/workspace/robo
PORT=30000
cd "$WS" || exit 1

MODEL=/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV
NEWLIB="$WS/tools/auto_select_tools/optimization_configs/new/config/library_gpu4"

export HF_HOME="$WS/cache/hf"
export HUGGINGFACE_HUB_CACHE="$WS/cache/hf/hub"
export TRANSFORMERS_CACHE="$WS/cache/hf/transformers"
export TORCH_HOME="$WS/cache/torch"
export XDG_CACHE_HOME="$WS/cache/xdg"
export PIP_CACHE_DIR="$WS/cache/pip"
export TMPDIR="$WS/tmp"
export VLLM_LOGGING_LEVEL=WARNING

# GEMM 调优产物：只在工作目录内，不触碰 /opt/dtk 原库
export ROCBLAS_TENSILE_LIBPATH="$NEWLIB"
export VLLM_ATTENTION_BACKEND=ROCM_AITER_FA

set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

LOG="$WS/logs/serve_graph_vitcompile_$(date +%Y%m%d-%H%M%S).log"
echo "[start] tuned lib=$NEWLIB log=$LOG"

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
  --compilation-config '{"compile_mm_encoder": true}' --limit-mm-per-prompt '{"image": 8}' \
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
