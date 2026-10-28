#!/usr/bin/env bash
set -uo pipefail
WS=/root/private_data/workspace/robo
RT="$WS/third_party/upstream_eval/RoboTracer/Evaluation/summarize_acc.py"

echo "########## calculate_metrics 全文（313-360）##########"
sed -n '305,365p' "$RT"

echo
echo "########## 3D 任务里 [Warn][2D] / Skip / 聚合（900-1060）##########"
sed -n '900,1060p' "$RT"

echo
echo "########## 所有 Skip / Warn 打印点 ##########"
grep -n "\[Skip\]\|\[Warn\]\|\[WARN\]" "$RT"
