#!/usr/bin/env bash
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
echo "=== 找 upscale / prompt 对比文件 ==="
find outputs -maxdepth 3 -name '*upscale*' -o -maxdepth 3 -name 'upscale_compare.json' 2>/dev/null | head -20
echo
find outputs -maxdepth 3 -name '*compare*' 2>/dev/null | head -20
echo
echo "=== cvbench_upscale 各子目录内容 ==="
for d in outputs/bench_public/cvbench_upscale/*/; do
  echo "--- $d"
  ls -1 "$d" | head -6
done
