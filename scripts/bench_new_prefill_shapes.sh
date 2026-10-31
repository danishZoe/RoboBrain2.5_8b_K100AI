#!/usr/bin/env bash
set -euo pipefail
WS=/root/private_data/workspace/robo
RUN_ID=${RUN_ID:?"set RUN_ID"}
ROOT="$WS/outputs/prompt_retune/$RUN_ID"
OUT="$ROOT/${OUT_NAME:-prefill_microbench}"
LIB="${CANDIDATE_LIB:-$WS/tools/auto_select_tools/optimization_configs/new/config/library_gpu4}"
mkdir -p "$OUT"
cd "$WS"
python3 - "$ROOT/trace/new_shapes_only.log" "$OUT/task.txt" <<'PY'
import re, sys
rows = []
for line in open(sys.argv[1], encoding="utf-8"):
    m = int(re.search(r" -m (\d+)", line).group(1))
    n = int(re.search(r" -n (\d+)", line).group(1))
    k = int(re.search(r" -k (\d+)", line).group(1))
    rows.append((m, n, k, 1))
with open(sys.argv[2], "w", encoding="utf-8") as f:
    for row in sorted(set(rows)):
        f.write("%d %d %d %d\n" % row)
PY
set +u
source ./env.sh >/dev/null 2>&1
set -u
started=$(date +%s)
for mode in stock candidate; do
  for round in 1 2; do
    echo "[microbench] $mode round$round"
    if [ "$mode" = candidate ]; then
      ROCBLAS_TENSILE_LIBPATH="$LIB" tools/auto_select_tools/source-test/bin/bench-test \
        "$OUT/task.txt" 2 20 NN bf16 0 > "$OUT/${mode}_round${round}.log" 2>&1
    else
      env -u ROCBLAS_TENSILE_LIBPATH tools/auto_select_tools/source-test/bin/bench-test \
        "$OUT/task.txt" 2 20 NN bf16 0 > "$OUT/${mode}_round${round}.log" 2>&1
    fi
    grep 'N,N,' "$OUT/${mode}_round${round}.log"
  done
done
ended=$(date +%s)
printf 'prefill_microbench\t%s\t%s\t%s\n' "$started" "$ended" "$((ended-started))" >> "$ROOT/timings.tsv"
