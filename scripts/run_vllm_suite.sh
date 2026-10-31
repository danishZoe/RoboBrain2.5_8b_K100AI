#!/usr/bin/env bash
# vLLM serving 全套验证（一次启动跑完所有实验）
#
#   bash scripts/run_vllm_suite.sh [--mem-frac 0.70] [--ctx 8192] [--port 8000]
#
# 依次执行：
#   A. 探针（max_tokens=768，官方 inference.py 默认值） + 与 transformers 基线逐坐标比对
#   B. 探针（沿用基线各自的 max_new_tokens 预算） + 逐坐标比对
#   C. 多图验证（T02 要求「单图/多图均正常」）
#   D. 并发扫描（concurrency 1/4/8），产出 serving 口径性能基线
set -o pipefail
cd "$(dirname "$0")/.."
source ./env.sh

PORT=8000
MEM_FRAC=0.70
CTX=8192
EXTRA_ARGS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --port)      PORT="$2"; shift 2 ;;
    --mem-frac)  MEM_FRAC="$2"; shift 2 ;;
    --ctx)       CTX="$2"; shift 2 ;;
    --extra)     EXTRA_ARGS="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

TS="$(date +%Y%m%d_%H%M%S)"
SRV_LOG="${WORK_DIR}/logs/vllm_suite_server_${TS}.log"
mkdir -p "${WORK_DIR}/logs" "${WORK_DIR}/outputs"

cleanup() {
  echo "--- stopping vllm ---"
  if [ -n "${SRV_PID:-}" ]; then
    kill -TERM -- "-${SRV_PID}" 2>/dev/null || kill -TERM "${SRV_PID}" 2>/dev/null
    sleep 8
    kill -KILL -- "-${SRV_PID}" 2>/dev/null || kill -KILL "${SRV_PID}" 2>/dev/null
  fi
  sleep 3
  ps -eo pid,cmd | grep -E "serve_vllm_patched|vllm.entrypoints" | grep -v grep || echo "no leftover proc"
}
trap cleanup EXIT

echo "=== 启动 vLLM (eager, mem_frac=${MEM_FRAC}, ctx=${CTX}) ==="
setsid python3 scripts/serve_vllm_patched.py \
  --model "${MODEL_PATH}" \
  --served-model-name robobrain2.5-8b-nv \
  --host 127.0.0.1 --port "${PORT}" \
  --tensor-parallel-size 1 --dtype bfloat16 \
  --gpu-memory-utilization "${MEM_FRAC}" --max-model-len "${CTX}" \
  --enforce-eager --max-num-batched-tokens 4096 \
  --limit-mm-per-prompt '{"image": 8}' \
  ${EXTRA_ARGS} > "${SRV_LOG}" 2>&1 < /dev/null &
SRV_PID=$!

for i in $(seq 1 90); do
  code=$(curl -s -m 4 -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/health" 2>/dev/null)
  if [ "$code" = "200" ]; then echo "READY after ~$((i*10))s"; break; fi
  if ! kill -0 "${SRV_PID}" 2>/dev/null; then
    echo "SERVER DIED at ~$((i*10))s"; tail -25 "${SRV_LOG}"; exit 1
  fi
  sleep 10
done

echo
echo "############ A. 探针：max_tokens=768（官方默认值） ############"
python3 scripts/probe_http_request.py --backend vllm --port "${PORT}" \
  --tasks samples/tasks_official.json --image-dir samples/images \
  --repeat 1 --override-max-tokens 768 --out-dir outputs/vllm_mt768 \
  2>&1 | tee "${WORK_DIR}/logs/vllm_suite_probeA_${TS}.log" | tail -22
python3 scripts/compare_backends.py \
  --baseline outputs/phase3_baseline/baseline_raw.json \
  --candidate outputs/vllm_mt768/probe_result.json \
  --out outputs/vllm_mt768/agreement.md 2>&1 | tail -18

echo
echo "############ B. 探针：沿用基线各自的 max_new_tokens 预算 ############"
python3 scripts/probe_http_request.py --backend vllm --port "${PORT}" \
  --tasks samples/tasks_official.json --image-dir samples/images \
  --repeat 1 --out-dir outputs/vllm_mtbudget \
  2>&1 | tee "${WORK_DIR}/logs/vllm_suite_probeB_${TS}.log" | tail -22
python3 scripts/compare_backends.py \
  --baseline outputs/phase3_baseline/baseline_raw.json \
  --candidate outputs/vllm_mtbudget/probe_result.json \
  --out outputs/vllm_mtbudget/agreement.md 2>&1 | tail -18

echo
echo "############ C. 多图验证 ############"
python3 scripts/probe_http_request.py --backend vllm --port "${PORT}" \
  --tasks samples/tasks_multi_image.json --image-dir samples/images \
  --repeat 1 --override-max-tokens 256 --out-dir outputs/vllm_multi_image \
  2>&1 | tee "${WORK_DIR}/logs/vllm_suite_probeC_${TS}.log" | tail -20

echo
echo "############ D. 并发扫描（serving 口径性能基线） ############"
python3 scripts/bench_http.py --backend vllm --port "${PORT}" --tag vllm_base \
  --tasks samples/tasks_official.json --image-dir samples/images \
  --repeat 3 --concurrency 1 4 8 --out-dir outputs/phase4 \
  2>&1 | tee "${WORK_DIR}/logs/vllm_suite_bench_${TS}.log" | tail -30

echo
echo "=== server 关键事件 ==="
grep -nE "Application startup complete|GPU KV cache size|maximum concurrency" "${SRV_LOG}" | tail -5
