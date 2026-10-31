#!/usr/bin/env bash
set -o pipefail
W=/root/private_data/workspace/robo
echo "===== verify.py 中 base_dir / HERE / baseline_path 的定义与用法 ====="
grep -n "base_dir\|^HERE\|HERE =\|baseline_path\|def main" "$W/submission/verify.py" | head -40
echo
echo "----- main() 前 60 行 -----"
sed -n "$(grep -n 'def main' "$W/submission/verify.py" | cut -d: -f1),+62p" "$W/submission/verify.py"
