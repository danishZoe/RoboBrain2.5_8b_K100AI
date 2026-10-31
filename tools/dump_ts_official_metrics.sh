#!/usr/bin/env bash
set -uo pipefail
WS=/root/private_data/workspace/robo
RT="$WS/third_party/upstream_eval/RoboTracer/Evaluation/summarize_acc.py"
echo "########## calculate_metrics / 2D 指标聚合 ##########"
grep -n "Warn\]\[2D\|calculate_metrics\|metrics_2d\|2D Path Metrics\|def compute_accuracy_2d\|dfd\|frechet" "$RT" | head -60
echo
echo "########## 2D 聚合函数全文 ##########"
awk '/def calculate_metrics/,/^def [a-z_]+\(/' "$RT" | head -80
echo
echo "########## compute_accuracy_2d 全文 ##########"
awk '/def compute_accuracy_2d/,/^def [a-z_]+\(/' "$RT" | head -120
