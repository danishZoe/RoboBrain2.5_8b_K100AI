#!/usr/bin/env bash
# =============================================================================
# Single-variable A/B: --max-num-batched-tokens
#
# Hypothesis
# ----------
# With chunked prefill and --max-num-batched-tokens 4096, a 16384-input request
# (text ~16.4K + vision ~2.0K) is prefilled in ~5 chunks, i.e. 5 separate
# forward passes. A larger token budget should reduce the number of chunks and
# therefore the prefill latency (TTFT) of the 16384 tiers, at the cost of
# activation memory (which may shrink the KV cache and thus the real
# concurrency of the c64 tier).
#
# Everything else is identical to the baseline server (see
# submission/bench/run_config.txt): same weights, same --enforce-eager, same
# mem-fraction, same --max-num-seqs, same no-prefix-caching, same image
# dataset, same seed. Only --max-num-batched-tokens changes.
#
# Usage:
#   bash scripts/run_batched_tokens_ab.sh [--values "4096,8192,16384"]
#                                        [--tiers "16384:1,16384:8"] [--port 8000]
# =============================================================================
set -o pipefail

WORK_DIR="${WORK_DIR:-/root/private_data/workspace/robo}"
MODEL_PATH="${MODEL_PATH:-/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV}"
MEM_FRAC=0.85
CTX=32768
PORT=8000
MODEL_NAME=robobrain2.5-8b-nv
MAX_NUM_SEQS=64
VALUES="4096,8192,16384"
TIERS="16384:1,16384:8"
TAG="bnb_$(date +%Y%m%d_%H%M%S)"

while [ $# -gt 0 ]; do
  case "$1" in
    --values) VALUES="$2"; shift 2 ;;
    --tiers) TIERS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --ctx) CTX="$2"; shift 2 ;;
    --mem-frac) MEM_FRAC="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

cd "$WORK_DIR" || exit 1
mkdir -p logs outputs/phase5_ab_bnb
OUTROOT="outputs/phase5_ab_bnb/${TAG}"
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

start_server() {
  local bnt="$1"
  local log="$2"
  setsid nohup python3 scripts/serve_vllm_patched.py \
    --model "$MODEL_PATH" --served-model-name "$MODEL_NAME" \
    --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size 1 --pipeline-parallel-size 1 --dtype bfloat16 \
    --trust-remote-code \
    --gpu-memory-utilization "$MEM_FRAC" --max-model-len "$CTX" \
    --max-num-batched-tokens "$bnt" --max-num-seqs "$MAX_NUM_SEQS" \
    --enforce-eager --no-enable-prefix-caching \
    --limit-mm-per-prompt '{"image": 8}' \
    > "$log" 2>&1 &
  for i in $(seq 1 90); do
    if curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      echo "  server ready (${bnt} tokens) after ~$((i*10))s"
      return 0
    fi
    sleep 10
  done
  echo "  server NOT ready (${bnt} tokens)"
  return 1
}

IFS=',' read -ra VAL_ARR <<< "$VALUES"
IFS=',' read -ra TIER_ARR <<< "$TIERS"

# 安全闸：官方六档压测还在跑时禁止启动本脚本。
# 本脚本会 pkill 服务进程，在压测期间执行会直接毁掉最后一档数据。
if pgrep -f 'bash bench.sh' >/dev/null 2>&1 || pgrep -f 'bench_serving' >/dev/null 2>&1; then
  echo "[bnb] ABORT: 官方压测仍在运行，拒绝启动（避免杀掉服务毁掉基准数据）" >&2
  exit 3
fi

for V in "${VAL_ARR[@]}"; do
  echo "=============================================="
  echo "=== max-num-batched-tokens = ${V} ==="
  stop_server
  LOG="logs/vllm_bnb${V}_${TAG}.log"
  if ! start_server "$V" "$LOG"; then
    echo "  SKIP: server failed to start for ${V}"
    tail -20 "$LOG"
    continue
  fi
  KV=$(grep -o 'GPU KV cache size: [0-9,]* tokens' "$LOG" | tail -1)
  CONC=$(grep -o 'Maximum concurrency for [0-9,]* tokens per request: [0-9.]*x' "$LOG" | tail -1)
  echo "  ${KV:-KV n/a} | ${CONC:-conc n/a}"
  {
    echo "# max-num-batched-tokens=${V}  tag=${TAG}"
    echo "# ${KV:-KV n/a}"
    echo "# ${CONC:-conc n/a}"
  } >> "$OUTROOT/run_config.txt"

  for T in "${TIER_ARR[@]}"; do
    ILEN="${T%%:*}"; C="${T##*:}"
    N=$(( C * 2 > 8 ? C * 2 : 8 ))
    echo "  tier in=${ILEN} conc=${C} prompts=${N}"
    python3 scripts/bench_serving_vllm_image.py \
      --backend vllm --host 127.0.0.1 --port "$PORT" \
      --model "$MODEL_NAME" --tokenizer "$MODEL_PATH" \
      --dataset-name image --image-count 1 --image-resolution 1080p \
      --random-input-len "$ILEN" --random-output-len 1024 --random-range-ratio 1 \
      --max-concurrency "$C" --num-prompts "$N" --seed 1 \
      --output-file "$OUTROOT/bnb${V}_${ILEN}_c${C}.jsonl" \
      > "$OUTROOT/bnb${V}_${ILEN}_c${C}.log" 2>&1
    grep -E 'Successful requests|Failed requests|Benchmark duration|Total input tokens|Total generated tokens|Output token throughput|Mean TTFT|Mean TPOT|Mean E2E' \
      "$OUTROOT/bnb${V}_${ILEN}_c${C}.log" | sed 's/^/      /'
  done
done

stop_server

echo
echo "=============================================="
echo "=== A/B summary (${OUTROOT}) ==="
python3 - "$OUTROOT" <<'PY'
import json, os, sys, glob
root = sys.argv[1]
rows = []
for p in sorted(glob.glob(os.path.join(root, "bnb*.jsonl"))):
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        name = os.path.basename(p)[:-len(".jsonl")]
        bnt = name.split("_")[0].replace("bnb", "")
        rows.append((int(bnt), r.get("random_input_len"), r.get("max_concurrency"),
                     r.get("completed"), r.get("output_throughput"),
                     r.get("mean_ttft_ms"), r.get("mean_tpot_ms"),
                     r.get("mean_e2e_latency_ms")))
print("| max_num_batched_tokens | 输入档 | 并发 | completed | 输出吞吐 tok/s | 平均TTFT ms | 平均TPOT ms | 平均端到端 ms |")
print("| --- | --- | --- | --- | --- | --- | --- | --- |")
for bnt, ilen, c, comp, tps, ttft, tpot, e2e in sorted(rows):
    print("| %d | %s | %s | %s | %.2f | %.2f | %.2f | %.2f |" %
          (bnt, ilen, c, comp, tps or 0, ttft or 0, tpot or 0, e2e or 0))
PY
echo "=== done ==="
