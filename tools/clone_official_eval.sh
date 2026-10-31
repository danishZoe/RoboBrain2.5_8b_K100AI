#!/usr/bin/env bash
set -o pipefail
W=/root/private_data/workspace/robo
# shellcheck disable=SC1091
source "$W/tools/env_net.sh" >/dev/null 2>&1
U="$W/third_party/upstream_eval"
mkdir -p "$U"

clone() {
  local url="$1" dir="$2"
  if [ -d "$U/$dir/.git" ]; then
    echo "[skip] $dir 已存在"
  else
    echo "[clone] $url -> $dir"
    git clone --depth 1 "$url" "$U/$dir" 2>&1 | tail -3
  fi
}

clone https://github.com/Zhoues/RoboRefer.git        RoboRefer
clone https://github.com/flageval-baai/FlagEvalMM.git FlagEvalMM
clone https://github.com/FlagOpen/Robo-Dopamine.git  Robo-Dopamine

echo
echo "===== 顶层结构 ====="
for d in RoboRefer FlagEvalMM RoboTracer Robo-Dopamine; do
  [ -d "$U/$d" ] || continue
  echo "--- $d ($(du -sh "$U/$d" | cut -f1))"
  find "$U/$d" -maxdepth 2 -not -path '*/.git/*' -not -name '.git' -type d | sed "s#$U/$d#  .#" | head -25
done

echo
echo "===== FlagEvalMM 里与 RoboBrain / 三个数据集相关的文件 ====="
grep -rliE "robobrain|CV-?Bench|RefSpatial|TraceSpatial" "$U/FlagEvalMM" \
  --include='*.py' --include='*.yaml' --include='*.yml' --include='*.json' --include='*.md' --include='*.sh' \
  2>/dev/null | sed "s#$U/FlagEvalMM#  #" | head -40

echo
echo "===== RoboRefer 里与评测相关的文件 ====="
grep -rliE "test_benchmark|summarize_acc|mask|success" "$U/RoboRefer" \
  --include='*.py' --include='*.md' 2>/dev/null | sed "s#$U/RoboRefer#  #" | head -20
