#!/usr/bin/env bash
set -o pipefail
U=/root/private_data/workspace/robo/third_party/upstream_eval

echo "############################################################"
echo "### RoboTracer/Evaluation/summarize_acc.py"
echo "############################################################"
cat "$U/RoboTracer/Evaluation/summarize_acc.py"

echo
echo "############################################################"
echo "### RoboTracer/Evaluation/outputs/ 参考结果"
echo "############################################################"
ls -l "$U/RoboTracer/Evaluation/outputs/" 2>/dev/null
for f in "$U/RoboTracer/Evaluation/outputs"/*.jsonl; do
  [ -e "$f" ] || continue
  echo "--- $(basename "$f")  行数=$(wc -l < "$f")"
  head -n 1 "$f" | cut -c1-400
  echo
done
