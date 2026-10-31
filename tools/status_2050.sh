#!/usr/bin/env bash
set -o pipefail
W=/root/private_data/workspace/robo

echo "===== 时间 ====="
date

echo
echo "===== 关键进程 ====="
ps -eo pid,etime,stat,cmd 2>/dev/null | grep -Ei 'python|bash tools|pipeline|dryrun|bench' | grep -v -E 'grep|check_|show_|probe_|status' | cut -c1-160 | head -25

echo
echo "===== 近 3 小时内新增/修改的文件（排除大目录） ====="
find "$W" -type f -newermt '-180 minutes' \
  -not -path '*/tmp/pypi_probe/*' -not -path '*/cache/*' -not -path '*/envs/pylibs/*' \
  -not -name '*.pyc' 2>/dev/null | head -60

echo
echo "===== 压测轮次 ====="
for f in "$W"/submission/bench/round*.jsonl; do [ -e "$f" ] && printf '%-16s %4s lines\n' "$(basename "$f")" "$(wc -l < "$f")"; done
echo "--- summary.md ---"
cat "$W/submission/bench/summary.md" 2>/dev/null | head -20

echo
echo "===== 报告里哪些文件含 '3.8' 小节 ====="
grep -rln '^#\+.*3\.8' "$W/reports" 2>/dev/null | head
for f in $(grep -rln '^#\+.*3\.8' "$W/reports" 2>/dev/null | head -3); do
  echo "--- $f"
  grep -n '^#\+.*3\.8' "$f" | head
done

echo
echo "===== outputs/ 下最近的实验目录 ====="
ls -lt "$W/outputs" 2>/dev/null | head -20
