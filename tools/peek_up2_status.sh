#!/usr/bin/env bash
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
echo "=== 全量 baseline cvbench.log 的评分段耗时线索 ==="
grep -nE '用时|评分|scor|汇总|accuracy_3d' logs/public_bench_baseline-eager/cvbench.log | head -20
echo
echo "=== GPU / 队列 ==="
cat logs/cvbench_up2_full.log | tail -4
echo
command -v rocm-smi >/dev/null 2>&1 && echo "rocm-smi 可用" || echo "无 rocm-smi"
grep -iE 'utilization|Util' logs/serve_graph_*.log 2>/dev/null | tail -3
echo
echo "=== 上一次 up2 Count（并发16）速率对照 ==="
grep -E '完成 788' logs/cvbench_upscale.log
