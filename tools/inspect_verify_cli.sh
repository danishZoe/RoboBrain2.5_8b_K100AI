#!/usr/bin/env bash
W=/root/private_data/workspace/robo
cd "$W/submission" || exit 1
echo "=== verify.py CLI ==="
grep -n 'add_argument' verify.py | head -30
echo
echo "=== baseline 路径解析 ==="
grep -n 'baselines' verify.py | head -20
echo
echo "=== 输出文件路径 ==="
grep -n 'verify_result\|verify_evidence\|out_file\|OUTPUT' verify.py | head -20
