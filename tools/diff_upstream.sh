#!/usr/bin/env bash
W=/root/private_data/workspace/robo
UP="$W/third_party/RoboBrain2.5-upstream-HEAD"

echo "=== 1. 忽略行尾 CR 后的真实差异 ==="
if diff --strip-trailing-cr -q "$W/third_party/RoboBrain2.5/inference.py" "$UP/inference.py" >/dev/null 2>&1; then
  echo "IDENTICAL —— 去掉行尾 CR 后完全一致（此前 diff 显示全文变化只是 CRLF/LF 差异）"
else
  echo "REAL DIFF:"
  diff --strip-trailing-cr -u "$W/third_party/RoboBrain2.5/inference.py" "$UP/inference.py" | head -60
fi
echo
echo "行尾符：本地快照 CRLF 行数 / upstream LF 行数"
printf 'local CRLF=%s  upstream CRLF=%s\n' \
  "$(grep -c $'\r' "$W/third_party/RoboBrain2.5/inference.py")" \
  "$(grep -c $'\r' "$UP/inference.py")"

echo
echo "=== 2. 我们额外加的格式桥（answer_adapter.py 35-60 行） ==="
sed -n '30,60p' "$W/scripts/answer_adapter.py"

echo
echo "=== 3. upstream demo 脚本引用但仓库缺失的资产 ==="
grep -oE 'navigation[0-9]*\.jpg' "$UP/inference.py" | sort -u | while read -r f; do
  if [ -f "$UP/assets/demo/$f" ]; then echo "  $f  EXISTS"; else echo "  $f  MISSING (upstream 自身 demo 脚本引用了不存在的文件)"; fi
done

echo
echo "=== 4. 压测状态 ==="
date +%H:%M:%S
wc -l "$W"/submission/bench/round1.jsonl "$W"/submission/bench/round2.jsonl 2>/dev/null
tr '\r' '\n' < "$W"/submission/bench/round1_16384_1024_c64.log | grep -E '^[0-9]+%' | tail -1
