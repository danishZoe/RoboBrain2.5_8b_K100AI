#!/usr/bin/env bash
set -uo pipefail
WS=/root/private_data/workspace/robo
cd "$WS"

echo "=== date ==="
date

echo "=== chunk completion per job ==="
for j in 0 1 2 3; do
  n=$(grep -c "job${j} chunk .* 完成" logs/ts_score_3d.log 2>/dev/null || echo 0)
  echo "job${j}: ${n}/100"
done

echo "=== last lines ==="
tail -n 5 logs/ts_score_3d.log 2>/dev/null

echo "=== running score procs (pid etime rss) ==="
ps -eo pid,etime,rss,args --sort=-rss 2>/dev/null | grep '[s]core_3d_chunked' | awk '{print $1, $2, $3, $NF}' | head -30
echo "count: $(ps -eo args 2>/dev/null | grep -c '[s]core_3d_chunked')"

echo "=== output json ==="
ls -la outputs/bench_public/tracespatial/*/scores_3dchunk_*.json 2>/dev/null
ls -la outputs/bench_public_noneager/tracespatial/*/scores_3dchunk_*.json 2>/dev/null

echo "=== mem ==="
free -g | head -3
cat /sys/fs/cgroup/memory.current 2>/dev/null
