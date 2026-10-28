#!/usr/bin/env bash
set -uo pipefail
WS=/root/private_data/workspace/robo
cd "$WS"

echo "=== missing chunks per job ==="
for j in 0 1 2 3; do
  done_list=$(grep -o "job${j} chunk [0-9]* 完成" logs/ts_score_3d.log | awk '{print $4}' | sort -n)
  miss=""
  for i in $(seq 0 99); do
    if ! echo "$done_list" | grep -qx "$i"; then miss="$miss $i"; fi
  done
  echo "job${j} missing:${miss:- none}"
done

echo "=== all python procs (score + children) ==="
ps -eo pid,ppid,etime,rss,stat,args --sort=-rss 2>/dev/null | grep -E '[p]ython3|[s]core_3d' | head -40

echo "=== tmp chunk dirs with output ==="
for tag in $(ls -d tmp/ts3d/*/ 2>/dev/null); do
  echo "--- $tag"
  ls "$tag" | head -3
done

echo "=== recent chunk stdout/err tails for slow ones ==="
for d in tmp/ts3d/*/; do
  for c in "$d"c*/; do
    if [ ! -f "$c/done.json" ] && [ -d "$c" ]; then
      echo "PENDING: $c"
      ls -la "$c" | head -10
    fi
  done
done 2>/dev/null | head -60
