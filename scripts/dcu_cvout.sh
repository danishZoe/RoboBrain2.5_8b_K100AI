#!/bin/bash
W=/root/private_data/workspace/robo
cd "$W" || exit 1
echo "=== bench_public/cvbench dirs ==="
ls -la outputs/bench_public/ 2>/dev/null | head -30
echo
echo "=== cvbench_up2_full contents ==="
ls -la outputs/bench_public/cvbench_up2_full/ 2>/dev/null | head -20
du -sh outputs/bench_public/cvbench_up2_full 2>/dev/null
echo
echo "=== per_sample.json head ==="
head -c 1200 outputs/bench_public/cvbench_up2_full/per_sample.json 2>/dev/null
echo
echo "=== up2_full_result.json ==="
head -c 1500 outputs/bench_public/cvbench_up2_full_result.json 2>/dev/null
