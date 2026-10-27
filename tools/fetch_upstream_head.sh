#!/usr/bin/env bash
# 拉取官方 upstream HEAD 快照，与工作目录内的本地快照做差异对照（只读比对）
source /root/private_data/workspace/robo/tools/env_net.sh

W=/root/private_data/workspace/robo
UP="$W/third_party/RoboBrain2.5-upstream-HEAD"
mkdir -p "$W/logs" "$W/tmp"

echo "=== 1. clone upstream HEAD (depth 1) ==="
if [ -d "$UP/.git" ]; then
  echo "already cloned; fetching"
  git -C "$UP" fetch --depth 1 origin 2>&1 | tail -2
else
  rm -rf "$UP"
  git clone --depth 1 https://github.com/FlagOpen/RoboBrain2.5.git "$UP" 2>&1 | tail -3
fi

if [ -d "$UP/.git" ]; then
  echo
  echo "=== 2. upstream HEAD commit ==="
  git -C "$UP" log -1 --format='commit=%H%ndate=%cI%nsubject=%s'
  echo
  echo "=== 3. upstream assets/demo 资产清单 ==="
  ls -la "$UP/assets/demo/" 2>/dev/null
  echo
  echo "=== 4. 本地快照 vs upstream HEAD: inference.py 差异 ==="
  if diff -q "$W/third_party/RoboBrain2.5/inference.py" "$UP/inference.py" >/dev/null 2>&1; then
    echo "IDENTICAL  (提示词模板 / 解析正则 / 示例逐字一致)"
  else
    diff -u "$W/third_party/RoboBrain2.5/inference.py" "$UP/inference.py" | head -80
  fi
  echo
  echo "=== 5. upstream HEAD 里的任务提示词模板（逐字） ==="
  grep -n -A2 -E 'task == "pointing"|task == "trajectory"|task == "grounding"' "$UP/inference.py" | head -30
  echo
  echo "=== 6. upstream HEAD assets/demo 的 SHA256 ==="
  sha256sum "$UP"/assets/demo/* 2>/dev/null
  echo
  echo "=== 7. upstream HEAD 的示例 Case（含 navigation2） ==="
  grep -n -E 'prompt = |image = |task=' "$UP/inference.py" | sed -n '1,40p'
fi

echo
echo "=== 8. 压测状态 ==="
date +%H:%M:%S
wc -l "$W"/submission/bench/round1.jsonl "$W"/submission/bench/round2.jsonl 2>/dev/null
