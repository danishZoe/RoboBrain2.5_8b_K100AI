#!/usr/bin/env bash
# 诊断：CV-Bench 官方 FlagEvalMM 输入链路（图像编码 / 缩放 / 提示词）
set -o pipefail
FE=/root/private_data/workspace/robo/third_party/upstream_eval/FlagEvalMM
cd "$FE" || exit 1

echo "########## 1. 全仓搜索 max_image_size / min_short_side / max_long_side ##########"
grep -rn 'max_image_size\|min_short_side\|max_long_side' --include='*.py' --include='*.yaml' --include='*.yml' --include='*.json' . | grep -v '\.pyc' | head -40

echo
echo "########## 2. tasks/cv_bench 目录树 ##########"
find tasks/cv_bench -type f | head -30

echo
echo "########## 3. cv_bench_test.py ##########"
sed -n '1,120p' tasks/cv_bench/cv_bench_test.py

echo
echo "########## 4. 配置目录（找 cv_bench 的 task config） ##########"
find . -maxdepth 3 -type d | grep -iv '\.git\|__pycache__\|site-packages' | head -40
