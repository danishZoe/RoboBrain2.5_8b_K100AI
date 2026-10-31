#!/bin/bash
W=/root/private_data/workspace/robo
cd "$W" || exit 1
echo "=== submission/bench_public ==="
ls -la submission/bench_public 2>/dev/null
echo "=== du ==="
du -sh submission/bench_public third_party/benchmarks third_party/upstream_eval 2>/dev/null
echo
echo "=== third_party/benchmarks tree (depth 3) ==="
find third_party/benchmarks -maxdepth 3 2>/dev/null | head -60
echo
echo "=== du of each ==="
du -sh third_party/benchmarks/* 2>/dev/null
echo
echo "=== upstream_eval tree (depth 2) ==="
find third_party/upstream_eval -maxdepth 2 2>/dev/null | head -40
echo
echo "=== where is akaily/cv-bench ? ==="
find / -maxdepth 8 -type d -name "akaily*" 2>/dev/null | head -10
find / -maxdepth 9 -type d -path "*cv*bench*" 2>/dev/null | grep -viE "proc|sys" | head -20
echo "=== modelscope cache dirs ==="
ls -la /root/.cache/modelscope 2>/dev/null
ls -la /root/.cache/modelscope/hub 2>/dev/null
ls -la /root/.cache/modelscope/datasets 2>/dev/null
find /root/.cache/modelscope -maxdepth 3 -type d 2>/dev/null | head -20
