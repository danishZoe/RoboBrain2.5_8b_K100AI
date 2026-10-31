#!/usr/bin/env bash
# vLLM serving 复跑：启动 -> 内存采样 -> 就绪 -> 6样例探针 -> 跨后端比对 -> 收尾
#
#   bash scripts/run_vllm_probe.sh [--eager] [--mem-frac 0.70] [--ctx 8192]
#                                 [--port 8000] [--tag vllm_eager]
#
# 关键约束（2026-10-27 实测）：
#   容器内存硬上限 115 GiB（/sys/fs/cgroup/memory/memory.limit_in_bytes）
#   vLLM 的 torch.compile / inductor 阶段会触顶 OOM，导致 Pod 被重建
#   => 首跑一律带 --enforce-eager 跳过编译；非 eager 版本单独一次实验再评估
# NOTE: do not use 'set -u' here - /opt/dtk/env.sh references unset vars and would abort.
set -o pipefail
cd "$(dirname "$0")/.."
source ./env.sh

PORT=8000
MEM_FRAC=0.70
CTX=8192
TAG=vllm_eager
EAGER=1
MAXTOK=4096
EXTRA_ARGS=""

while [ $# -gt 0 ]; do
  case "$1" in
    --port)      PORT="$2"; shift 2 ;;
    --mem-frac)  MEM_FRAC="$2"; shift 2 ;;
    --ctx)       CTX="$2"; shift 2 ;;
    --tag)       TAG="$2"; shift 2 ;;
    --max-tokens) MAXTOK="$2"; shift 2 ;;
    --eager)     EAGER=1; shift ;;
    --no-eager)  EAGER=0; shift ;;
    --extra)     EXTRA_ARGS="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

TS="$(date +%Y%m%d_%H%M%S)"
SRV_LOG="${WORK_DIR}/logs/${TAG}_server_${TS}.log"
MEM_LOG="${WORK_DIR}/logs/${TAG}_mem_${TS}.log"
PROBE_DIR="${WORK_DIR}/outputs/${TAG}"
mkdir -p "${WORK_DIR}/logs" "${PROBE_DIR}"

CG=/sys/fs/cgroup/memory/memory.usage_in_bytes
LIMIT=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo 0)

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

echo "server log : ${SRV_LOG}"
echo "mem log    : ${MEM_LOG}"
echo "probe dir  : ${PROBE_DIR}"
echo "port=${PORT} mem_frac=${MEM_FRAC} ctx=${CTX} eager=${EAGER} max_batched_tokens=${MAXTOK}"
echo "cgroup memory limit = ${LIMIT} bytes ($(awk "BEGIN{printf \"%.1f\", ${LIMIT}/1024/1024/1024}") GiB)"

EAGER_FLAG=""
if [ "$EAGER" = "1" ]; then EAGER_FLAG="--enforce-eager"; fi

setsid python3 scripts/serve_vllm_patched.py \
  --model "${MODEL_PATH}" \
  --served-model-name robobrain2.5-8b-nv \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization "${MEM_FRAC}" \
  --max-model-len "${CTX}" \
  --max-num-batched-tokens "${MAXTOK}" \
  ${EAGER_FLAG} \
  --limit-mm-per-prompt '{"image": 8}' \
  > "${SRV_LOG}" 2>&1 < /dev/null &
SRV_PID=$!
echo "vllm pgid=${SRV_PID}"

# 内存采样：整个启动期的容器内存占用
(
  PEAK=0
  while kill -0 "${SRV_PID}" 2>/dev/null; do
    U=$(cat "${CG}" 2>/dev/null || echo 0)
    P=$(cat /sys/fs/cgroup/memory/memory.peak 2>/dev/null || echo 0)
    [ "${U}" -gt "${PEAK}" ] 2>/dev/null && PEAK="${U}"
    echo "$(date +%H:%M:%S) usage=${U} peak_cgroup=${P}" >> "${MEM_LOG}"
    sleep 2
  done
  echo "PEAK_USAGE_BYTES=${PEAK}" >> "${MEM_LOG}"
) &
MEM_PID=$!

# 等待就绪
READY=0
for i in $(seq 1 90); do
  code=$(curl -s -m 4 -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/health" 2>/dev/null)
  if [ "$code" = "200" ]; then echo "READY after ~$((i*10))s"; READY=1; break; fi
  if ! kill -0 "${SRV_PID}" 2>/dev/null; then echo "SERVER DIED at ~$((i*10))s"; break; fi
  sleep 10
done

if [ "${READY}" != "1" ]; then
  echo "=== 未就绪：最后 30 行 ==="
  tail -30 "${SRV_LOG}"
  echo "=== 内存采样峰值 ==="
  tail -3 "${MEM_LOG}"
  exit 1
fi

echo "=== 就绪，开始探针 ==="
python3 scripts/probe_http_request.py --backend vllm --port "${PORT}" \
  --tasks samples/tasks_official.json --image-dir samples/images \
  --repeat 2 --out-dir "${PROBE_DIR}" 2>&1 | tee "${WORK_DIR}/logs/${TAG}_probe_${TS}.log" | tail -40

echo "=== 与 transformers 基线逐坐标比对 ==="
python3 scripts/compare_backends.py \
  --baseline outputs/phase3_baseline/baseline_raw.json \
  --candidate "${PROBE_DIR}/probe_result.json" \
  --out "${PROBE_DIR}/agreement.md" 2>&1 | tail -30

echo "=== 内存采样峰值 ==="
tail -3 "${MEM_LOG}"
echo "=== server log 关键事件 ==="
grep -nE "Application startup complete|Uvicorn running|Using cache directory|KV cache size|GPU KV cache size|maximum concurrency" "${SRV_LOG}" | tail -8
