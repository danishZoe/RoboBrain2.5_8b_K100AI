#!/usr/bin/env bash
# 检查：收尾流水线是否会整体重写 00_RUNBOOK.md；对比本地/远端 RUNBOOK
set -o pipefail
W=/root/private_data/workspace/robo

echo "===== 流水线里对 00_RUNBOOK 的处理 ====="
grep -rn "00_RUNBOOK" "$W/tools" "$W/submission" 2>/dev/null | head -20 || echo "(无引用)"

echo
echo "===== inject_*.py 是否整文件覆盖 ====="
for f in "$W"/tools/inject_*.py; do
  [ -e "$f" ] || continue
  printf '%-46s ' "$(basename "$f")"
  if grep -qE "write_text|open\(.*['\"]w['\"]" "$f"; then
    echo "存在写文件动作"
  else
    echo "只读/追加"
  fi
done

echo
echo "===== 远端 RUNBOOK 指纹 ====="
ls -l "$W/reports/00_RUNBOOK.md"
md5sum "$W/reports/00_RUNBOOK.md"
grep -c "" "$W/reports/00_RUNBOOK.md" | sed 's/^/行数: /'
grep -n "^## 5.5\|^## 6\. \|^## 7\. " "$W/reports/00_RUNBOOK.md" | head

echo
echo "===== 压测/流水线状态 ====="
for f in "$W"/submission/bench/round*.jsonl; do [ -e "$f" ] && printf '%-16s %4s lines\n' "$(basename "$f")" "$(wc -l < "$f")"; done
ps -eo pid,etime,cmd | grep -E 'bench_serving|pipeline_after_bench|dryrun' | grep -v grep | cut -c1-110 || true
