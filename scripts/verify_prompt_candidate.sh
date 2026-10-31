#!/usr/bin/env bash
set -euo pipefail
WS=/root/private_data/workspace/robo
RUN_ID=${RUN_ID:?"set RUN_ID"}
ROOT="$WS/outputs/prompt_retune/$RUN_ID"
OUT="$ROOT/${VERIFY_OUT:-library_open_verification}"
LIB="${NEWLIB:-$WS/tools/auto_select_tools/optimization_configs/new/config/library_gpu4}"
mkdir -p "$OUT"
cd "$WS"

stop_service() {
  local f="$WS/outputs/.bench_server_tunedlib.pid" pid args
  [ -f "$f" ] || return 0
  pid=$(cat "$f")
  args=$(ps -p "$pid" -o args= 2>/dev/null || true)
  if [[ "$args" == *"serve_vllm_patched.py"* ]]; then
    kill -TERM -- "-$pid" 2>/dev/null || true
    for _ in $(seq 1 15); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 2
    done
    kill -KILL -- "-$pid" 2>/dev/null || true
  fi
  rm -f "$f"
}

started=$(date +%s)
stop_service
python3 "$ROOT/watch_library_opens.py" --library "$LIB" \
  --out "$OUT/open_events.json" --seconds 420 > "$OUT/watcher.log" 2>&1 &
watcher=$!
sleep 1
NEWLIB="$LIB" bash scripts/start_candidate_service.sh 2>&1 | tee "$OUT/service_start.log"
sleep 2
stop_service
kill -TERM "$watcher" 2>/dev/null || true
wait "$watcher" 2>/dev/null || true
ended=$(date +%s)
printf '%s\t%s\t%s\t%s\n' "${VERIFY_OUT:-library_open_verification}" "$started" "$ended" "$((ended-started))" >> "$ROOT/timings.tsv"
cat "$OUT/open_events.json"
