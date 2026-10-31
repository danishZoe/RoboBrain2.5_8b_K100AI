#!/usr/bin/env bash
set -uo pipefail
WS=/root/private_data/workspace/robo
FE="$WS/third_party/upstream_eval/FlagEvalMM"

echo "########## 1. vqa_base_dataset.py build_prompt ##########"
sed -n '1,400p' "$FE/flagevalmm/dataset/vqa_base_dataset.py" 2>/dev/null | grep -n "def build_prompt" -A 60

echo
echo "########## 2. prompt_template.py default_prompt_func ##########"
grep -n "def default_prompt_func" -A 60 "$FE/flagevalmm/prompt/prompt_template.py" 2>/dev/null

echo
echo "########## 3. tasks/cv_bench 目录 ##########"
ls -la "$FE/tasks/cv_bench/" 2>/dev/null

echo
echo "########## 4. tasks/cv_bench/process.py ##########"
sed -n '1,220p' "$FE/tasks/cv_bench/process.py" 2>/dev/null

echo
echo "########## 5. tasks/cv_bench/evaluate.py ##########"
sed -n '1,220p' "$FE/tasks/cv_bench/evaluate.py" 2>/dev/null

echo
echo "########## 6. common_types.py evaluate_multiple_choice ##########"
grep -n "def evaluate_multiple_choice" -A 40 "$FE/flagevalmm/evaluator/common_types.py" 2>/dev/null
grep -n "def process_multiple_choice" -A 60 "$FE/flagevalmm/evaluator/common_types.py" 2>/dev/null
grep -n "def maybe_clean_answer" -A 40 "$FE/flagevalmm/evaluator/common_types.py" 2>/dev/null

echo
echo "########## 7. robobrain model_adapter 关键片段 ##########"
grep -n "build_prompt\|post_prompt\|pre_prompt\|question\|prompt" "$FE/model_zoo/vlm/robobrain/model_adapter.py" 2>/dev/null | head -60
