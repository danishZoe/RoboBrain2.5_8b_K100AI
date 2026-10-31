#!/bin/bash
W=/root/private_data/workspace/robo
cd "$W" || exit 1
echo "=== eval_timing.md ==="
cat outputs/bench_public/eval_timing.md 2>/dev/null
echo
echo "=== eval_timing.json (head) ==="
head -c 2500 outputs/bench_public/eval_timing.json 2>/dev/null
echo
echo
echo "=== DCU 容器/卡 是否独占 ==="
hy-smi 2>/dev/null | head -20
echo
echo "=== DCU 端 CV-Bench 运行的进程与时间 ==="
ls -la --time-style=full-iso outputs/bench_public/cvbench/predictions/ 2>/dev/null
