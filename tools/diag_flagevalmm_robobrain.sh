#!/usr/bin/env bash
# 诊断：FlagEvalMM 的 RoboBrain 适配器 / 模型配置 / 图像加载 / 提示词
set -o pipefail
FE=/root/private_data/workspace/robo/third_party/upstream_eval/FlagEvalMM
cd "$FE" || exit 1

echo "########## 1. model_configs 里有没有 RoboBrain ##########"
ls model_configs/open/ | grep -i 'robo\|brain' || echo "(open/ 下没有)"
ls model_configs/ || true
find model_configs -iname '*robo*' -o -iname '*brain*' | head

echo
echo "########## 2. model_zoo/vlm/robobrain/ 内容 ##########"
find model_zoo/vlm/robobrain -type f | head -20
echo "--- model_adapter.py ---"
sed -n '1,200p' model_zoo/vlm/robobrain/model_adapter.py

echo
echo "########## 3. VqaBaseDataset 的图像加载 ##########"
find . -name 'vqa_base_dataset.py' | head
sed -n '1,140p' flagevalmm/datasets/vqa_base_dataset.py 2>/dev/null || true

echo
echo "########## 4. 默认提示词模板 ##########"
find . -name 'prompt_template.py' | head
sed -n '1,120p' flagevalmm/prompt/prompt_template.py 2>/dev/null || true
