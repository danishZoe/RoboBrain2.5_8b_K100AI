#!/usr/bin/env bash
set -o pipefail
U=/root/private_data/workspace/robo/third_party/upstream_eval

echo "############################################################"
echo "### RoboRefer/Evaluation/summarize_acc.py （RefSpatial 判分）"
echo "############################################################"
sed -n '1,200p' "$U/RoboRefer/Evaluation/summarize_acc.py" 2>/dev/null || ls -l "$U/RoboRefer/Evaluation/" 2>/dev/null

echo
echo "############################################################"
echo "### RoboRefer/Evaluation 目录"
echo "############################################################"
ls -l "$U/RoboRefer/Evaluation/" 2>/dev/null
