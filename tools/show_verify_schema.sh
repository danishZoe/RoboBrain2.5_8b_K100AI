#!/usr/bin/env bash
set -o pipefail
W=/root/private_data/workspace/robo
echo "===== verify.py 390-470（指标解析 / 参考输出 / 判分器） ====="
sed -n '390,470p' "$W/submission/verify.py"
echo
echo "===== verify.py 470-532（逐样本评测 / 写证据） ====="
sed -n '470,532p' "$W/submission/verify.py"
echo
echo "===== 演练脚本如何造假基线 ====="
grep -nE "metrics|subset|ref|reference|judge|baselines|BLOCKED" "$W/tools/verify_path_dryrun.sh" | head -40
