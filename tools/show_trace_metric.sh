#!/usr/bin/env bash
set -o pipefail
U=/root/private_data/workspace/robo/third_party/upstream_eval
F="$U/RoboTracer/Evaluation/summarize_acc.py"
echo "===== 行数 ====="; wc -l "$F"
echo
echo "===== 指标与主流程（从 interpolate_trajectory_by_distance 之后） ====="
sed -n '200,420p' "$F"
