#!/usr/bin/env bash
# 取上游权威来源：HF 模型卡的评测表 + RoboTracer/RefSpatial 评测脚本与提示词
set -o pipefail
W=/root/private_data/workspace/robo
# shellcheck disable=SC1091
source "$W/tools/env_net.sh" >/dev/null 2>&1
mkdir -p "$W/third_party/upstream_eval" "$W/logs"

echo "===== 1) HF 模型卡：BAAI/RoboBrain2.5-8B-NV ====="
MC="$W/third_party/upstream_eval/model_card_RoboBrain2.5-8B-NV.md"
code=$(curl -sSL -m 60 -o "$MC" -w '%{http_code}' \
  "https://huggingface.co/BAAI/RoboBrain2.5-8B-NV/raw/main/README.md" 2>&1)
echo "http=$code  bytes=$(wc -c < "$MC" 2>/dev/null)"
echo "--- 含三个数据集名的段落 ---"
grep -niE "CV-?Bench|RefSpatial|TraceSpatial|Robo-?Dopamine" "$MC" 2>/dev/null | head -20
echo "--- 评测表格附近 ---"
grep -niE -A 18 "benchmark|evaluation" "$MC" 2>/dev/null | grep -iE "\||bench|spatial|trace|dopamine|score|%" | head -40

echo
echo "===== 2) RoboTracer 仓库（TraceSpatial / RefSpatial 的评测代码来源） ====="
if [ ! -d "$W/third_party/upstream_eval/RoboTracer/.git" ]; then
  git clone --depth 1 https://github.com/zhoues/RoboTracer.git \
      "$W/third_party/upstream_eval/RoboTracer" 2>&1 | tail -3
fi
echo "--- 目录结构（2 层） ---"
find "$W/third_party/upstream_eval/RoboTracer" -maxdepth 2 \
     -not -path '*/.git/*' -not -name '.git' | head -40

echo
echo "===== 3) 仓库内评测脚本 / 提示词 / 判分 ====="
grep -rniE "TraceSpatial|RefSpatial|CV-Bench|prompt|judge|accuracy|IoU" \
     "$W/third_party/upstream_eval/RoboTracer" \
     --include='*.py' --include='*.md' --include='*.sh' --include='*.json' \
     -l 2>/dev/null | head -25
