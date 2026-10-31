#!/usr/bin/env bash
set -euo pipefail
WS=/root/private_data/workspace/robo
RUN_ID=${RUN_ID:?"set RUN_ID"}
NEWLIB=${NEWLIB:?"set NEWLIB"}
ROOT="$WS/outputs/prompt_retune/$RUN_ID"
OUT="$ROOT/decode_clean_microbench"
OLDLIB="$WS/tools/auto_select_tools/optimization_configs/new/config/library_gpu4"
mkdir -p "$OUT"
cd "$WS"
python3 - "$ROOT/decode_clean_shapes.log" "$OUT" <<'PY'
import re, sys
files = {"N": open(f"{sys.argv[2]}/nn_task.txt", "w"), "T": open(f"{sys.argv[2]}/tn_task.txt", "w")}
for line in open(sys.argv[1], encoding="utf-8"):
    ta = re.search(r"--transposeA (\w)", line).group(1)
    vals = [int(re.search(rf" -{x} (\d+)", line).group(1)) for x in ("m", "n", "k")]
    files[ta].write("%d %d %d 1\n" % tuple(vals))
for f in files.values():
    f.close()
PY
set +u
source ./env.sh >/dev/null 2>&1
set -u
started=$(date +%s)
for mode in old merged; do
  if [ "$mode" = old ]; then lib="$OLDLIB"; else lib="$NEWLIB"; fi
  for transpose in NN TN; do
    lower=$(printf '%s' "$transpose" | tr 'A-Z' 'a-z')
    for round in 1 2; do
      echo "[decode-clean] $mode $transpose round$round"
      ROCBLAS_TENSILE_LIBPATH="$lib" tools/auto_select_tools/source-test/bin/bench-test \
        "$OUT/${lower}_task.txt" 5 50 "$transpose" bf16 0 \
        > "$OUT/${mode}_${lower}_round${round}.log" 2>&1
      grep -E 'N,N,|T,N,' "$OUT/${mode}_${lower}_round${round}.log"
    done
  done
done
ended=$(date +%s)
printf 'decode_clean_microbench\t%s\t%s\t%s\n' "$started" "$ended" "$((ended-started))" >> "$ROOT/timings.tsv"
