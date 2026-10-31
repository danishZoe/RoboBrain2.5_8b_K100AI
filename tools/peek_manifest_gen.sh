#!/usr/bin/env bash
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
echo "=== 生成 MANIFEST 的脚本？ ==="
ls -1 tools | grep -iE 'manifest|checksum|sha' | head
echo
echo "=== 提交包自带的清单脚本 ==="
ls -1 submission | grep -iE 'manifest'
echo
echo "=== 远端 submission/README.md 的章节（判断是否旧版） ==="
grep -nE '^#{2,4} ' submission/README.md | tail -12
