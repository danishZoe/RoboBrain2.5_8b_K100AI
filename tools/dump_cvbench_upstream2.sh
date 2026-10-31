#!/usr/bin/env bash
set -uo pipefail
WS=/root/private_data/workspace/robo
FE="$WS/third_party/upstream_eval/FlagEvalMM"

echo "########## A. get_default_post_prompt ##########"
grep -n "def get_default_post_prompt" -A 40 "$FE/flagevalmm/prompt/prompt_template.py" 2>/dev/null

echo
echo "########## B. robobrain model_adapter 全文（前 140 行） ##########"
sed -n '1,140p' "$FE/model_zoo/vlm/robobrain/model_adapter.py" 2>/dev/null

echo
echo "########## C. robobrain model_adapter 140-280 ##########"
sed -n '140,280p' "$FE/model_zoo/vlm/robobrain/model_adapter.py" 2>/dev/null

echo
echo "########## D. 模型 preprocessor_config ##########"
cat /public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV/preprocessor_config.json 2>/dev/null

echo
echo "########## E. 协议实验产物列表 ##########"
ls -la "$WS/outputs/bench_public/cvbench_protocol/" 2>/dev/null
find "$WS/outputs/bench_public/cvbench_protocol" -name "summary.json" 2>/dev/null | head -20
