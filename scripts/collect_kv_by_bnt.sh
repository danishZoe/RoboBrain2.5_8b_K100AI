#!/usr/bin/env bash
# =============================================================================
# collect_kv_by_bnt.sh - 采集不同 --max-num-batched-tokens 取值下的 KV cache 容量
#
# 用途：E2 的假设是「放大 token 预算 → 激活内存上升 → KV cache 变小 → 真实并发下降」。
#       只看吞吐无法证明这条机制链，因此单独把「KV cache token 数」按取值测出来。
#
# 方法：不依赖 vLLM 的 INFO 日志（E2 运行时把 VLLM_LOGGING_LEVEL 压到 WARNING，
#       "GPU KV cache size: N tokens" 那行被吞掉），改为读运行中服务的 /metrics：
#       vllm:cache_config_info 里带 num_gpu_blocks 与 block_size，
#       KV token 数 = num_gpu_blocks × block_size（基线值 3958 × 64 = 253,312，
#       与首提基线一致，可交叉验证口径）。
#
# 与 E2 保持同参数：eager / ctx 32768 / mem_frac 0.85 / max_num_seqs 64 /
#                   no-prefix-caching / limit-mm-per-prompt {"image":8}
#
# 用法: bash scripts/collect_kv_by_bnt.sh [--values "4096,8192,16384"] [--port 8000]
# =============================================================================
set -uo pipefail

WORK_DIR="${WORK_DIR:-/root/private_data/workspace/robo}"
MODEL_PATH="${MODEL_PATH:-/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV}"
MEM_FRAC=0.85
CTX=32768
PORT=8000
MODEL_NAME=robobrain2.5-8b-nv
MAX_NUM_SEQS=64
VALUES="4096,8192,16384"
TAG="kv_$(date +%Y%m%d_%H%M%S)"

while [ $# -gt 0 ]; do
  case "$1" in
    --values) VALUES="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

cd "$WORK_DIR" || exit 1
mkdir -p logs outputs/phase5_kv_bnb
OUTROOT="outputs/phase5_kv_bnb/${TAG}"
mkdir -p "$OUTROOT"

export WORK_DIR MODEL_PATH
export HF_HOME="$WORK_DIR/cache/hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$WORK_DIR/envs/pylibs:${PYTHONPATH:-}"
export VLLM_LOGGING_LEVEL=WARNING
set +u
source /opt/dtk/env.sh
set -u

stop_server() {
  pkill -f "scripts/serve_vllm_patched.py" 2>/dev/null
  pkill -f "VLLM::EngineCore" 2>/dev/null
  sleep 8
}

# 安全闸：官方六档压测还在跑时禁止启动
if pgrep -f 'bash bench.sh' >/dev/null 2>&1 || pgrep -f 'bench_serving' >/dev/null 2>&1; then
  echo "[kv] ABORT: 官方压测仍在运行，拒绝启动" >&2
  exit 3
fi

TSV="$OUTROOT/kv_by_bnt.tsv"
printf 'max_num_batched_tokens\tnum_gpu_blocks\tblock_size\tkv_cache_tokens\tkv_cache_gib\tmax_conc_4096\tmax_conc_16384\tready_s\n' > "$TSV"

IFS=',' read -ra VAL_ARR <<< "$VALUES"
for V in "${VAL_ARR[@]}"; do
  echo "=== bnt=${V} ==="
  stop_server
  LOG="logs/vllm_kv${V}_${TAG}.log"
  setsid nohup python3 scripts/serve_vllm_patched.py \
    --model "$MODEL_PATH" --served-model-name "$MODEL_NAME" \
    --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size 1 --pipeline-parallel-size 1 --dtype bfloat16 \
    --trust-remote-code \
    --gpu-memory-utilization "$MEM_FRAC" --max-model-len "$CTX" \
    --max-num-batched-tokens "$V" --max-num-seqs "$MAX_NUM_SEQS" \
    --enforce-eager --no-enable-prefix-caching \
    --limit-mm-per-prompt '{"image": 8}' \
    > "$LOG" 2>&1 &
  SPID=$!
  T0=$(date +%s)
  READY=0
  for i in $(seq 1 90); do
    if curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      READY=1; break
    fi
    if ! kill -0 "$SPID" 2>/dev/null; then echo "  server exited early"; break; fi
    sleep 10
  done
  RS=$(( $(date +%s) - T0 ))
  if [ "$READY" != "1" ]; then
    echo "  SKIP: bnt=${V} 服务未就绪"
    printf '%s\tNOT_READY\t-\t-\t-\t-\t-\t%s\n' "$V" "$RS" >> "$TSV"
    continue
  fi
  M=$(curl -s --noproxy '*' "http://127.0.0.1:${PORT}/metrics" 2>/dev/null | grep 'cache_config_info' | head -1)
  NGB=$(printf '%s' "$M" | grep -o 'num_gpu_blocks="[0-9]*"' | grep -o '[0-9]*' | head -1)
  BS=$(printf '%s' "$M" | grep -o 'block_size="[0-9]*"' | grep -o '[0-9]*' | head -1)
  [ -z "$NGB" ] && NGB=0
  [ -z "$BS" ] && BS=64
  TOK=$(( NGB * BS ))
  GIB=$(awk -v t="$TOK" 'BEGIN{printf "%.2f", t*36*2*2/1024/1024/1024}')
  MC4=$(awk -v t="$TOK" 'BEGIN{printf "%.1f", t/4096}')
  MC16=$(awk -v t="$TOK" 'BEGIN{printf "%.1f", t/16384}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$V" "$NGB" "$BS" "$TOK" "$GIB" "$MC4" "$MC16" "$RS" >> "$TSV"
  echo "  bnt=${V}: blocks=${NGB} block_size=${BS} kv_tokens=${TOK} (~${GIB} GiB) ready=${RS}s"
  stop_server
done

echo
echo "=== KV cache 汇总 ==="
cat "$TSV"
echo
echo "=== 基线交叉验证：4096 档应与首提基线 253,312 token 一致 ==="
grep -P '^4096\t' "$TSV" || true
echo "=== done ==="
