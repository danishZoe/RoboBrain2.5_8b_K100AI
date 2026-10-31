#!/usr/bin/env bash
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
echo "=== build_summary_table.py 的扫描范围 ==="
grep -nE 'glob|iterdir|walk|listdir|cvbench|refspatial|tracespatial' submission/bench_public/build_summary_table.py | head -25
echo
echo "=== 它读哪些子目录 ==="
grep -n -B2 -A8 'def main' submission/bench_public/build_summary_table.py | head -40
