#!/usr/bin/env bash
# Probe which SGLang attention backend combination actually works on this node.
#
#   bash scripts/probe_attn_backend.sh torch_native:sdpa wave:sdpa aiter:sdpa
#
# Each argument is "<main-backend>[:<mm-attention-backend>]".
#   main backend  -> --attention-backend        (choices: triton torch_native wave aiter ...)
#   mm backend    -> --mm-attention-backend     (choices: sdpa fa3 triton_attn aiter_attn ...)
#
# For each candidate: start the server, wait for /v1/models, issue one image+text
# request, print the answer, then stop the server. Servers that fail are reported
# with the decisive log lines (invalid memory access / missing symbol / etc.).
# NOTE: do not use 'set -u' here - /opt/dtk/env.sh references unset vars and would abort.
set -o pipefail
cd "$(dirname "$0")/.."
source ./env.sh

CANDIDATES=("$@")
if [ ${#CANDIDATES[@]} -eq 0 ]; then
  CANDIDATES=(torch_native:sdpa wave:sdpa aiter:sdpa)
fi

PORT=30010
mkdir -p "${WORK_DIR}/logs" "${WORK_DIR}/outputs/probe_attn"

for CAND in "${CANDIDATES[@]}"; do
  BE="${CAND%%:*}"
  REST="${CAND#*:}"
  MM="${REST%%:*}"
  if [ "${REST}" = "${CAND}" ]; then MM=""; fi
  MI="${REST#*:}"
  if [ "${MI}" = "${REST}" ]; then MI=""; fi

  TS="$(date +%Y%m%d_%H%M%S)"
  SRV_LOG="${WORK_DIR}/logs/probe_attn_${BE}${MM:+_${MM}}${MI:+_${MI}}_${TS}.log"
  echo ""
  echo "############################################################"
  echo "# attention-backend=${BE}  mm-attention-backend=${MM:-<default>}  model-impl=${MI:-<default>}"
  echo "############################################################"

  MM_ARG=""
  [ -n "${MM}" ] && MM_ARG="--mm-attention-backend ${MM}"
  MI_ARG=""
  [ -n "${MI}" ] && MI_ARG="--model-impl ${MI}"

  setsid sglang serve \
    --model-path "${MODEL_PATH}" \
    --served-model-name robobrain2.5-8b-nv \
    --host 127.0.0.1 --port "${PORT}" \
    --tp-size 1 --dtype bfloat16 \
    --mem-fraction-static 0.85 --context-length 8192 \
    --max-running-requests 16 \
    --limit-mm-data-per-request '{"image": 8}' \
    --trust-remote-code \
    --attention-backend "${BE}" ${MM_ARG} ${MI_ARG} > "${SRV_LOG}" 2>&1 &
  SRV_PID=$!

  READY=0
  for i in $(seq 1 90); do
    if ! kill -0 "${SRV_PID}" 2>/dev/null; then
      echo "  RESULT: server exited during startup"
      break
    fi
    if python3 - "$PORT" <<'PYEOF' 2>/dev/null
import sys, urllib.request
try:
    urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/v1/models", timeout=5).read()
except Exception:
    sys.exit(1)
sys.exit(0)
PYEOF
    then
      READY=1
      echo "  server ready after $((i * 10))s"
      break
    fi
    sleep 10
  done

  if [ "${READY}" = "1" ]; then
    python3 scripts/probe_one_request.py --port "${PORT}" \
      --image "${WORK_DIR}/samples/images/official_affordance.jpg" \
      --prompt "What is shown in this image?" \
      --out "${WORK_DIR}/outputs/probe_attn/${BE}${MM:+_${MM}}${MI:+_${MI}}.json" 2>&1 | tail -12
  else
    echo "  RESULT: NOT READY - decisive log lines:"
    grep -E "Invalid address|Error code|Fatal Python error|NameError|ModuleNotFound|Exception:|assert_pkg_version|scheduler died" "${SRV_LOG}" 2>/dev/null | head -6 | sed 's/^/    /'
  fi

  kill -TERM -- "-${SRV_PID}" 2>/dev/null || kill -TERM "${SRV_PID}" 2>/dev/null
  sleep 10
  kill -KILL -- "-${SRV_PID}" 2>/dev/null || kill -KILL "${SRV_PID}" 2>/dev/null
  sleep 5
done

echo ""
echo "=== leftover servers (should be none) ==="
ps -eo pid,cmd | grep "sglang serve" | grep -v grep || echo "none"
