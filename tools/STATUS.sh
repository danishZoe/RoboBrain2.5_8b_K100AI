#!/usr/bin/env bash
# 当前任务全局状态快照（只读）
set -o pipefail
W=/root/private_data/workspace/robo

echo "===== time ====="
date

echo
echo "===== key processes ====="
ps -eo pid,etime,stat,cmd 2>/dev/null \
  | grep -Ei 'bench_serving|pipeline_after_bench|dryrun|serve_vllm|vllm.entrypoints|waiter' \
  | grep -v grep || echo "(none)"

echo
echo "===== bench round files ====="
ls -l "$W"/logs/ 2>/dev/null | grep -Ei 'round|bench|dryrun|pipeline' || echo "(no bench/dryrun/pipeline log files)"

echo
echo "===== bench rounds line counts ====="
for f in $(ls -1 "$W"/logs/*round*.jsonl 2>/dev/null); do
  printf '%-70s %s lines\n' "$(basename "$f")" "$(wc -l < "$f")"
done

echo
echo "===== official bench result summary (if any) ====="
for f in $(ls -1 "$W"/logs/*bench*.json 2>/dev/null); do
  echo "--- $f"
  cat "$f"
done

echo
echo "===== pipeline log tail ====="
for f in $(ls -1t "$W"/logs/*pipeline*.log 2>/dev/null | head -3); do
  echo "--- $f"
  tail -n 30 "$f"
done

echo
echo "===== dryrun log tail ====="
for f in $(ls -1t "$W"/logs/*dryrun*.log 2>/dev/null | head -3); do
  echo "--- $f"
  tail -n 30 "$f"
done

echo
echo "===== submission tree ====="
find "$W"/submission -maxdepth 2 -type f 2>/dev/null | head -50

echo
echo "===== verify_result.json ====="
cat "$W"/submission/verify_result.json 2>/dev/null || echo "(missing)"

echo
echo "===== disk ====="
df -h "$W" 2>/dev/null | tail -2
