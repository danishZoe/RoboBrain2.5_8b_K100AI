#!/usr/bin/env bash
set -o pipefail
U=/root/private_data/workspace/robo/third_party/upstream_eval/FlagEvalMM

for t in robo_spatial_home ocrbench mmmu cv_bench; do
  echo "############################################################"
  echo "### tasks/$t"
  echo "############################################################"
  ls -l "$U/tasks/$t" 2>/dev/null | head -20
  for f in "$U/tasks/$t"/*.md; do
    [ -e "$f" ] || continue
    echo "--- $(basename "$f")"
    sed -n '1,70p' "$f"
  done
done

echo
echo "===== 全仓库里含 'RoboBrain' 的文件 ====="
grep -rli "robobrain" "$U" 2>/dev/null | sed "s#$U#  #" | head -30
