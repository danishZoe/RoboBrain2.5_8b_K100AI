#!/usr/bin/env bash
set -uo pipefail
WS=/root/private_data/workspace/robo
cd "$WS"
echo "=== bash -n 语法检查 ==="
for f in tools/verify_3d_merge.sh tools/run_ts_score_3d_96.sh tools/chain_cvbench.sh \
         tools/exp_cvbench_upscale.sh tools/exp_cvbench_prompt_style.sh \
         tools/integrate_ts_results.sh; do
  if bash -n "$f" 2>/tmp/err_$$; then
    echo "  OK   $f"
  else
    echo "  FAIL $f"
    cat /tmp/err_$$
  fi
done
rm -f /tmp/err_$$
echo
echo "=== verify_3d_merge.sh 的行尾/引号统计 ==="
python3 - <<'PY'
p = "/root/private_data/workspace/robo/tools/verify_3d_merge.sh"
raw = open(p, "rb").read()
print("bytes:", len(raw))
print("有 CRLF:", b"\r\n" in raw)
try:
    txt = raw.decode("utf-8")
except UnicodeDecodeError as e:
    print("非 UTF-8:", e)
    txt = raw.decode("utf-8", "replace")
lines = txt.splitlines()
print("行数:", len(lines))
for i, ln in enumerate(lines, 1):
    if ln.count("'") % 2 == 1:
        print("  单引号数为奇数 行%d: %r" % (i, ln[:120]))
PY
