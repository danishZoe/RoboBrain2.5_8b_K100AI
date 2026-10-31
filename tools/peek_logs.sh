#!/usr/bin/env bash
# 只看日志头尾小样，避免大段 dump
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
for f in \
  logs/public_bench_baseline-eager/cvbench.log \
  logs/public_bench_baseline-eager/refspatial.log \
  logs/public_bench_baseline-eager/tracespatial.log \
  logs/public_bench_noneager-candidate/cvbench.log \
  logs/public_bench_baseline-eager/cvbench_protocol.log \
  logs/cvbench_prompt.log
do
  echo "=== $f ==="
  head -3 "$f"
  echo "  ...(tail)..."
  tail -4 "$f"
  echo
done
