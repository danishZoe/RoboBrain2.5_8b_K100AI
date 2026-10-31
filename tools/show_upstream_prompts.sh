#!/usr/bin/env bash
set -o pipefail
W=/root/private_data/workspace/robo
F="$W/third_party/RoboBrain2.5/inference.py"

echo "===== RoboBrain2.5 上游 inference.py 的提示词模板原文 ====="
sed -n '1,140p' "$F"

echo
echo "===== 官方 CLI / 入口（__main__ 部分） ====="
sed -n "$(grep -n '__main__' "$F" | head -1 | cut -d: -f1),\$p" "$F" | head -80
