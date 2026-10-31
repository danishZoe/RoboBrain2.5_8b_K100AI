#!/usr/bin/env bash
set -uo pipefail
WS=/root/private_data/workspace/robo
cd "$WS"
echo "=== 正在运行的判分子进程及其工作目录（= 所在分片）==="
for p in $(pgrep -f 'summarize_acc.py' 2>/dev/null); do
  cwd=$(readlink -f /proc/$p/cwd 2>/dev/null)
  et=$(ps -o etime= -p $p 2>/dev/null | tr -d ' ')
  md=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | sed 's/.*--model_name \([^ ]*\).*/\1/')
  echo "pid=$p etime=$et model=$md cwd=$cwd"
done

echo
echo "=== tmp/ts3d 下各组合的分片目录（判断哪些分片还没产出）==="
for tag in tracespatial_official_3D tracespatial_upstream_3D; do
  d="$WS/tmp/ts3d/$tag"
  [ -d "$d" ] || continue
  n_out=$(ls "$d" 2>/dev/null | wc -l)
  echo "--- $tag (目录数 $n_out)"
done

echo
echo "=== 每个分片目录里 outputs/ 下的 jsonl 行数（= 该分片样本数）==="
for tag in tracespatial_official_3D tracespatial_upstream_3D; do
  d="$WS/tmp/ts3d/$tag"
  [ -d "$d" ] || continue
  echo "--- $tag"
  for c in "$d"/c*/; do
    f=$(ls "$c"outputs/*.jsonl 2>/dev/null | head -1)
    if [ -n "$f" ]; then
      printf "%s %s\n" "$(basename $c)" "$(wc -l < "$f")"
    fi
  done | head -5
done
