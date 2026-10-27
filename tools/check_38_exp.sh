#!/usr/bin/env bash
set -o pipefail
W=/root/private_data/workspace/robo

echo "===== §3.8 报告原文 ====="
sed -n '380,470p' "$W/reports/10_phase5_bench.md"

echo
echo "===== phase5_token_accounting/result.json ====="
cat "$W/outputs/phase5_token_accounting/result.json" 2>/dev/null | head -c 4000
echo
echo "===== 同目录其它文件 ====="
ls -l "$W/outputs/phase5_token_accounting/" 2>/dev/null

echo
echo "===== 产出该实验的脚本 ====="
grep -rln "token_accounting" "$W/tools" "$W/scripts" "$W/submission/scripts" 2>/dev/null | head
