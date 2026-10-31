#!/usr/bin/env bash
set -euo pipefail
WS=/root/private_data/workspace/robo
RUN_ID=${RUN_ID:?"set RUN_ID"}
NEWLIB="${NEWLIB:-$WS/tools/auto_select_tools/optimization_configs/new/config/library_gpu4}"
ROOT="$WS/outputs/prompt_retune/$RUN_ID"
OUT="$ROOT/hipprof_session"
mkdir -p "$OUT"
cd "$WS"
started=$(date +%s)
cleanup() {
  set +u
  source ./env.sh >/dev/null 2>&1
  set -u
  hipprof --session-client robo --stop >/dev/null 2>&1 || true
  hipprof --session-client robo --flush >/dev/null 2>&1 || true
  hipprof --session-client robo --exit >/dev/null 2>&1 || true
}
trap cleanup EXIT
HIPPROF_OUT="$OUT" NEWLIB="$NEWLIB" bash scripts/start_hipprof_service.sh
set +u
source ./env.sh >/dev/null 2>&1
set -u
hipprof --session-client robo --start
RUN_ID="$RUN_ID" REUSE_SERVICE=1 TIERS=4096:8 ROUNDS=1 bash scripts/run_prompt_retune.sh hipprof_session_c8
hipprof --session-client robo --stop
hipprof --session-client robo --flush
hipprof --session-client robo --exit
trap - EXIT
ended=$(date +%s)
printf 'hipprof_session\t%s\t%s\t%s\n' "$started" "$ended" "$((ended-started))" >> "$ROOT/timings.tsv"
find "$OUT" -maxdepth 3 -type f -printf '%p %s\n' | sort
