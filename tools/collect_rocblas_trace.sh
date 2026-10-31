#!/usr/bin/env bash
# =============================================================================
# 采集 rocBLAS layer3 trace（诊断用，非调优）
#
# 目的：E3 的替代产出 —— 既然本节点没有厂商调优库与 shape 表，
#       至少把「实际跑了哪些 GEMM shape、各占多少时间」采下来，
#       作为「为何不做 GEMM 调优也能定位瓶颈」的一手证据。
#
# 做法：以 ROCBLAS_TRACE=3 / ROCBLAS_LOG_TRACE_PATH 启动一次服务，
#       打一发图文请求，然后停服务并解析 trace。
# 注意：开启 tracing 会有额外开销，本脚本的结果**不能**用作性能数据。
#
# 用法: bash tools/collect_rocblas_trace.sh [--port 8000] [--requests 1]
# =============================================================================
set -uo pipefail

WORK_DIR="${WORK_DIR:-/root/private_data/workspace/robo}"
MODEL_PATH="${MODEL_PATH:-/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV}"
PORT=8000
REQUESTS=1
TAG="trace_$(date +%Y%m%d_%H%M%S)"

while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --requests) REQUESTS="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

cd "$WORK_DIR" || exit 1
mkdir -p logs outputs/phase5_gemm_trace
OUT="outputs/phase5_gemm_trace/${TAG}"
mkdir -p "$OUT"

export WORK_DIR MODEL_PATH
export HF_HOME="$WORK_DIR/cache/hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$WORK_DIR/envs/pylibs:${PYTHONPATH:-}"
export VLLM_LOGGING_LEVEL=WARNING
set +u
source /opt/dtk/env.sh
set -u

# --- 1) 起带 tracing 的服务 ----------------------------------------------------
# 安全闸：官方六档压测还在跑时禁止启动（本脚本会 pkill 服务进程）。
if pgrep -f 'bash bench.sh' >/dev/null 2>&1 || pgrep -f 'bench_serving' >/dev/null 2>&1; then
  echo "[trace] ABORT: 官方压测仍在运行，拒绝启动（避免杀掉服务毁掉基准数据）" >&2
  exit 3
fi

export ROCBLAS_TRACE=3
export ROCBLAS_LOG_TRACE_PATH="$OUT/rocblas_layer3.json"
export ROCBLAS_LOG_BENCHMARK=1

echo "[trace] starting server with ROCBLAS_TRACE=3 ..."
setsid nohup python3 scripts/serve_vllm_patched.py \
  --model "$MODEL_PATH" --served-model-name robobrain2.5-8b-nv \
  --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 1 --dtype bfloat16 \
  --gpu-memory-utilization 0.85 --max-model-len 32768 \
  --max-num-batched-tokens 4096 --max-num-seqs 64 \
  --enforce-eager --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"image": 8}' \
  > "logs/vllm_${TAG}.log" 2>&1 &
SPID=$!

READY=0
for i in $(seq 1 90); do
  if curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    READY=1; echo "[trace] server ready after ~$((i*10))s"; break
  fi
  sleep 10
done

if [ "$READY" = "1" ]; then
  echo "[trace] sending ${REQUESTS} image request(s) ..."
  python3 - "$PORT" "$REQUESTS" <<'PY'
import base64, json, sys, urllib.request
port, n = sys.argv[1], int(sys.argv[2])
img = "/root/private_data/workspace/robo/samples/images/official_affordance.jpg"
with open(img, "rb") as f:
    url = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
payload = {
    "model": "robobrain2.5-8b-nv",
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": "Describe the image in exhaustive detail."}]}],
    "temperature": 0, "max_tokens": 256, "stream": False,
}
for i in range(n):
    req = urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % port,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read().decode())
    print("[trace] request %d ok, completion_tokens=%s" %
          (i + 1, d["usage"]["completion_tokens"]))
PY
else
  echo "[trace] server failed to start"
fi

# --- 2) 停服务 -----------------------------------------------------------------
pkill -f "scripts/serve_vllm_patched.py" 2>/dev/null
pkill -f "VLLM::EngineCore" 2>/dev/null
sleep 6

# --- 3) 解析 trace -------------------------------------------------------------
echo
echo "[trace] trace files:"
ls -la "$OUT" | head -10

python3 - "$OUT" <<'PY'
import json, os, re, sys, collections
out = sys.argv[1]
log = os.path.join(out, "rocblas_layer3.json")
if not os.path.exists(log):
    # rocBLAS 也可能按进程写多个文件
    cands = [f for f in os.listdir(out) if f.startswith("rocblas_layer3")]
    if not cands:
        print("[trace] 未找到 rocBLAS trace 文件（可能该构建未启用 tracing）")
        raise SystemExit(0)
    log = os.path.join(out, cands[0])

shapes = collections.Counter()
counts = collections.Counter()
total = 0.0
pat = re.compile(r'"(?:m|n|k)":\s*(\d+)')
with open(log, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        if "args" not in line:
            continue
        m = re.search(r'"m":\s*(\d+).*?"n":\s*(\d+).*?"k":\s*(\d+)', line)
        if not m:
            continue
        key = "x".join(m.groups())
        counts[key] += 1
        t = re.search(r'"total_time"\s*:\s*([0-9.eE+-]+)', line)
        if t:
            total += float(t.group(1))
            shapes[key] += float(t.group(1))

print("[trace] distinct GEMM shapes: %d, traced lines with args: %d" %
      (len(counts), sum(counts.values())))
print("[trace] top 15 shapes by total_time (us):")
for k, v in shapes.most_common(15):
    print("   %-28s  %12.1f us   calls=%d" % (k, v, counts[k]))
print("[trace] top 10 shapes by call count:")
for k, v in counts.most_common(10):
    print("   %-28s  calls=%d" % (k, v))
PY

echo "[trace] done; artifacts under $OUT"
