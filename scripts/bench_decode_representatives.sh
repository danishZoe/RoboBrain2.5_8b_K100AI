#!/usr/bin/env bash
set -euo pipefail
WS=/root/private_data/workspace/robo
RUN_ID=${RUN_ID:?"set RUN_ID"}
NEWLIB=${NEWLIB:?"set NEWLIB"}
ROOT="$WS/outputs/prompt_retune/$RUN_ID"
OUT="$ROOT/decode_microbench_merged"
OLDLIB="$WS/tools/auto_select_tools/optimization_configs/new/config/library_gpu4"
mkdir -p "$OUT"
cd "$WS"
cat > "$OUT/nn_task.txt" <<'EOF'
4096 1 4096 1
6144 1 4096 1
4096 64 4096 1
6144 64 4096 1
EOF
cat > "$OUT/tn_task.txt" <<'EOF'
151936 1 4096 1
151936 8 4096 1
151936 64 4096 1
EOF
set +u
source ./env.sh >/dev/null 2>&1
set -u
started=$(date +%s)
for mode in old merged; do
  if [ "$mode" = old ]; then lib="$OLDLIB"; else lib="$NEWLIB"; fi
  for transpose in NN TN; do
    lower=$(printf '%s' "$transpose" | tr 'A-Z' 'a-z')
    for round in 1 2; do
      echo "[decode-microbench] $mode $transpose round$round"
      ROCBLAS_TENSILE_LIBPATH="$lib" tools/auto_select_tools/source-test/bin/bench-test \
        "$OUT/${lower}_task.txt" 5 50 "$transpose" bf16 0 \
        > "$OUT/${mode}_${lower}_round${round}.log" 2>&1
      grep -E 'N,N,|T,N,' "$OUT/${mode}_${lower}_round${round}.log"
    done
  done
done
ended=$(date +%s)
printf 'decode_microbench_merged\t%s\t%s\t%s\n' "$started" "$ended" "$((ended-started))" >> "$ROOT/timings.tsv"
