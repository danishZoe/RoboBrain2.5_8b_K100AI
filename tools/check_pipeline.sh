#!/usr/bin/env bash
set -uo pipefail
pkill -f 'watch_bench_done.sh' 2>/dev/null && echo "stopped old watcher" || echo "no old watcher"
cd /root/private_data/workspace/robo/submission
echo "===== bench progress ====="
for f in bench/round1.jsonl bench/round2.jsonl; do
  [ -e "$f" ] || continue
  printf '%s: %s entries\n' "$f" "$(wc -l < "$f")"
  python3 - "$f" <<'PY'
import json,sys
for line in open(sys.argv[1],encoding="utf-8"):
    line=line.strip()
    if not line: continue
    r=json.loads(line)
    print("   in=%-6s conc=%-3s done=%-4s failed=%-3s dur=%8.2fs out_tps=%7.2f ttft=%9.2f" % (
        r.get("random_input_len"), r.get("max_concurrency"), r.get("completed"),
        r.get("failed",0), r.get("duration",0), r.get("output_throughput",0), r.get("mean_ttft_ms",0)))
PY
done
echo
echo "===== pipeline log tail ====="
tail -20 /root/private_data/workspace/robo/logs/pipeline_after_bench.log 2>/dev/null || echo "(no pipeline output yet)"
