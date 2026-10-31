#!/usr/bin/env bash
set -o pipefail
U=/root/private_data/workspace/robo/third_party/upstream_eval

echo "############################################################"
echo "### RoboRefer/Evaluation/test_benchmark.py"
echo "############################################################"
cat "$U/RoboRefer/Evaluation/test_benchmark.py"

echo
echo "############################################################"
echo "### FlagEvalMM tasks/robo_spatial_home/evaluate_norm1k.py （前 120 行）"
echo "############################################################"
sed -n '1,120p' "$U/FlagEvalMM/tasks/robo_spatial_home/evaluate_norm1k.py"

echo
echo "############################################################"
echo "### FlagEvalMM tasks/robo_spatial_home/process.py 与 robo_spatial_home_norm1k.py"
echo "############################################################"
cat "$U/FlagEvalMM/tasks/robo_spatial_home/process.py"
echo "---- runner ----"
cat "$U/FlagEvalMM/tasks/robo_spatial_home/robo_spatial_home_norm1k.py"
