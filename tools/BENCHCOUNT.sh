#!/usr/bin/env bash
set -o pipefail
W=/root/private_data/workspace/robo
B="$W/submission/bench"

echo "===== round line counts ====="
for f in "$B"/round*.jsonl; do
  [ -e "$f" ] || continue
  printf '%-16s %4s lines\n' "$(basename "$f")" "$(wc -l < "$f")"
done

echo
echo "===== round2 last record (truncated) ====="
if [ -s "$B/round2.jsonl" ]; then
  tail -n 1 "$B/round2.jsonl" | cut -c1-600
fi

echo
echo "===== summary.md ====="
cat "$B/summary.md" 2>/dev/null | head -80

echo
echo "===== run_config.txt ====="
cat "$B/run_config.txt" 2>/dev/null

echo
echo "===== bench round logs (tails) ====="
for f in $(ls -1t "$B"/*.log 2>/dev/null | head -4); do
  echo "--- $(basename "$f")"
  tail -n 12 "$f"
done
