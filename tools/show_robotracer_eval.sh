#!/usr/bin/env bash
set -o pipefail
W=/root/private_data/workspace/robo
R="$W/third_party/upstream_eval/RoboTracer"

echo "===== 下载进度 ====="
du -sh "$W/third_party/benchmarks"/* 2>/dev/null
ls -l "$W/third_party/benchmarks/" 2>/dev/null | head

echo
echo "===== RoboTracer README 里的数据集与提示词说明 ====="
sed -n '1,120p' "$R/README.md" | grep -niE "TraceSpatial|RefSpatial|CV-Bench|prompt|benchmark|answer|evaluat" | head -30

echo
echo "===== Evaluation/test_benchmark.py（提示词与判分入口） ====="
sed -n '1,140p' "$R/Evaluation/test_benchmark.py"

echo
echo "===== API/Qwen3VL.py 的提示词构造 ====="
grep -nE "prompt|PROMPT|template|messages|question" "$R/API/Qwen3VL.py" | head -30

echo
echo "===== Evaluation/summarize_acc.py（准确率统计口径） ====="
sed -n '1,60p' "$R/Evaluation/summarize_acc.py"
