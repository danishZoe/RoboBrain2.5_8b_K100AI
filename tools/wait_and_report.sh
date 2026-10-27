#!/usr/bin/env bash
set -uo pipefail
sleep "${1:-600}"
cd /root/private_data/workspace/robo/submission
echo "===== running client ====="
ps -eo pid,etime,args | grep -E 'bench_serving' | grep -v grep \
  | sed -E 's/.*--random-input-len ([0-9]+).*--max-concurrency ([0-9]+).*--num-prompts ([0-9]+).*--output-file ([^ ]+).*/IN=\1 CONC=\2 N=\3 OUT=\4/' \
  | head -2
echo
for f in bench/round1.jsonl bench/round2.jsonl; do
  [ -e "$f" ] || continue
  echo "-- $f ($(wc -l < "$f") lines)"
  python3 - "$f" <<'PY'
import json,sys
for line in open(sys.argv[1],encoding="utf-8"):
    line=line.strip()
    if not line: continue
    r=json.loads(line)
    print("   in=%-6s conc=%-3s done=%-4s failed=%-3s dur=%8.2fs out_tps=%7.2f ttft=%9.2f tpot=%7.2f in_tok=%-7s in_text=%-7s in_vis=%-7s out_tok=%-7s retok=%-7s" % (
        r.get("random_input_len"), r.get("max_concurrency"), r.get("completed"),
        r.get("failed",0), r.get("duration",0), r.get("output_throughput",0),
        r.get("mean_ttft_ms",0), r.get("mean_tpot_ms",0),
        r.get("total_input_tokens"), r.get("total_input_text_tokens"),
        r.get("total_input_vision_tokens"), r.get("total_output_tokens"),
        r.get("total_output_tokens_retokenized")))
PY
done
