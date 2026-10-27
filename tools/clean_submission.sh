#!/usr/bin/env bash
# 清理提交包里的编译缓存与临时垃圾，并列出最终清单
set -uo pipefail
cd /root/private_data/workspace/robo/submission
echo "===== remove __pycache__ ====="
find . -name '__pycache__' -type d -prune -print -exec rm -rf {} + 2>/dev/null | sed 's/^/  removed /'
find . -name '*.pyc' -delete 2>/dev/null
echo
echo "===== submission tree (depth 2) ====="
find . -maxdepth 2 \( -path ./bench -prune \) -o -print 2>/dev/null | grep -v '^\.$' | sort | head -80
echo
echo "===== bench/ ====="
ls -la bench/ | sed 's/^/  /'
echo
echo "===== sizes ====="
du -sh . 2>/dev/null | sed 's/^/  total /'
