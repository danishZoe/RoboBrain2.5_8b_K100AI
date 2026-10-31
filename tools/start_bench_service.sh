#!/usr/bin/env bash
# 启动 vLLM 服务（公开基准评测用）
# 用法: start_bench_service.sh <eager|graph> <port>
#   eager : 首次提交基线配置（--enforce-eager），端口默认 30000
#   graph : E1 候选配置（不加 --enforce-eager），用于精度回归对照
set -o pipefail
WS=/root/private_data/workspace/robo
MODE="${1:-eager}"
PORT="${2:-30000}"
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
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$WS/logs" "$WS/tmp"

set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

EAGER_ARG=""
if [ "$MODE" = "eager" ]; then
  EAGER_ARG="--enforce-eager"
fi

LOG="$WS/logs/serve_${MODE}_$(date +%Y%m%d-%H%M%S).log"
echo "[start] mode=$MODE port=$PORT log=$LOG"

# setsid：完全脱离当前进程组，避免启动方的会话/进程组被终止时带走服务
# （nohup 只能挡 SIGHUP，挡不住进程组 SIGTERM）
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
  $EAGER_ARG \
  --limit-mm-per-prompt '{"image": 8}' \
  > "$LOG" 2>&1 < /dev/null &

disown || true
sleep 2
echo $! > "$WS/outputs/.bench_server_${MODE}.pid"
echo "[start] pid=$(cat "$WS/outputs/.bench_server_${MODE}.pid")"

for i in $(seq 1 90); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 \
        "http://127.0.0.1:${PORT}/health" || echo 000)
  if [ "$code" = "200" ]; then
    echo "[start] READY after ${i}0s  http=$PORT"
    curl -s --max-time 10 "http://127.0.0.1:${PORT}/v1/models"; echo
    exit 0
  fi
  sleep 10
done
echo "[start] TIMEOUT waiting for health; tail log:"; tail -n 40 "$LOG"
exit 1
