#!/usr/bin/env bash
# 公开基准评测一键驱动：manifest -> RefSpatial -> TraceSpatial -> CV-Bench -> 离线重判分
# 用法: run_public_bench.sh <url> <run_id> [stage]
#   stage: all(默认) | manifest | refspatial | tracespatial | cvbench | rescore
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1

URL="${1:-http://127.0.0.1:30000/v1/chat/completions}"
RUN_ID="${2:-baseline-eager}"
STAGE="${3:-all}"

export HF_HOME="$WS/cache/hf"
export HUGGINGFACE_HUB_CACHE="$WS/cache/hf/hub"
export TRANSFORMERS_CACHE="$WS/cache/hf/transformers"
export TORCH_HOME="$WS/cache/torch"
export XDG_CACHE_HOME="$WS/cache/xdg"
export TMPDIR="$WS/tmp"
export PYTHONPATH="$WS/envs/pylibs:$WS/third_party/upstream_eval/FlagEvalMM:${PYTHONPATH:-}"
export PYTHONWARNINGS="ignore"
mkdir -p "$TMPDIR" "$WS/logs"

set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

LOGDIR="$WS/logs/public_bench_${RUN_ID}"
mkdir -p "$LOGDIR"
OUT="$WS/outputs/bench_public"

stamp() { date +"%Y-%m-%dT%H:%M:%S%z"; }
run_stage() {
  local name="$1"; shift
  echo ""
  echo "################ $(stamp) stage: $name ################"
  local log="$LOGDIR/${name}.log"
  "$@" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  echo "[stage $name] exit=$rc log=$log"
  return $rc
}

# ---- 0) 等服务 ----
if [ "$STAGE" = "all" ] || [ "$STAGE" = "refspatial" ] || [ "$STAGE" = "tracespatial" ] || [ "$STAGE" = "cvbench" ]; then
  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "${URL%/v1/chat/completions}/health" || echo 000)
    [ "$code" = "200" ] && break
    sleep 10
  done
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "${URL%/v1/chat/completions}/health" || echo 000)
  if [ "$code" != "200" ]; then
    echo "[run] 服务未就绪 ($code): $URL"; exit 1
  fi
  echo "[run] 服务就绪: $URL"
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "manifest" ]; then
  run_stage manifest python3 "$WS/submission/bench_public/manifest.py" \
    --out-dir "$OUT/manifest" || exit 2
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "refspatial" ]; then
  run_stage refspatial python3 "$WS/submission/bench_public/run_refspatial.py" \
    --url "$URL" --run-id "${RUN_ID}-refspatial" --out-dir "$OUT/refspatial" || exit 3
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "tracespatial" ]; then
  run_stage tracespatial python3 "$WS/submission/bench_public/run_tracespatial.py" \
    --url "$URL" --run-id "${RUN_ID}-tracespatial" --out-dir "$OUT/tracespatial" || exit 4
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "cvbench" ]; then
  run_stage cvbench python3 "$WS/submission/bench_public/run_cv_bench.py" \
    --url "$URL" --run-id "${RUN_ID}-cvbench" --out-dir "$OUT/cvbench" || exit 5
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "rescore" ]; then
  run_stage rescore python3 "$WS/submission/bench_public/rescore.py" \
    --bench-root "$OUT" || exit 6
fi

echo ""
echo "[run] 全部阶段完成 $(stamp)"
