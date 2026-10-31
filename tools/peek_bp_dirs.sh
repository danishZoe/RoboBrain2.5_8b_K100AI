#!/usr/bin/env bash
WS=/root/private_data/workspace/robo
cd "$WS/outputs/bench_public" || exit 1
for d in cvbench_upscale cvbench_prompt cvbench_protocol; do
  echo "=== $d ==="
  ls -1 "$d" 2>/dev/null
  echo
done
echo "=== 顶层 ==="
ls -1 | head -40
