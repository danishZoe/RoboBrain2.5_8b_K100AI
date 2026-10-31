#!/usr/bin/env bash
# 等待官方六档压测全部完成（round2.jsonl 出现 6 条），或超时；然后打印流水线日志
set -uo pipefail
W=/root/private_data/workspace/robo
MAX_WAIT=${1:-7200}
cd "$W/submission" || exit 1
DEADLINE=$(( $(date +%s) + MAX_WAIT ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  n=$( [ -f bench/round2.jsonl ] && wc -l < bench/round2.jsonl || echo 0 )
  if [ "$n" -ge 6 ]; then
    echo "[wait] six tiers complete at $(date -Is) (round2 entries=$n)"
    break
  fi
  if ! pgrep -f 'bash bench.sh' >/dev/null 2>&1 && ! pgrep -f 'bench_serving' >/dev/null 2>&1; then
    echo "[wait] bench loop exited at $(date -Is) (round2 entries=$n)"
    break
  fi
  sleep 30
done
echo "===== remote time: $(date -Is) ====="
wc -l bench/round1.jsonl bench/round2.jsonl 2>/dev/null
echo
echo "===== pipeline log (last 60) ====="
tail -60 "$W/logs/pipeline_after_bench.log" 2>/dev/null
