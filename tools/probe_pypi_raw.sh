#!/usr/bin/env bash
source /root/private_data/workspace/robo/tools/env_net.sh
W=/root/private_data/workspace/robo
TMP="$W/tmp/pypi_probe"
mkdir -p "$TMP"

echo "=== 1. devpi simple 页面原始内容探测 ==="
for u in "https://pypi.sourcefind.cn/release/dtk2604/aiter/" \
         "https://pypi.sourcefind.cn/release/dtk2604/+simple/aiter/" \
         "https://pypi.sourcefind.cn/release/dtk2604/aiter" \
         "https://pypi.sourcefind.cn/release/dtk2604/+simple/" ; do
  echo "--- $u"
  code=$(curl -sS -m 40 -L -o "$TMP/x.html" -w '%{http_code}' "$u" 2>/dev/null)
  echo "    code=$code  size=$(wc -c < "$TMP/x.html" 2>/dev/null)"
  if [ "$code" = "200" ]; then head -c 300 "$TMP/x.html" | tr -d '\n'; echo; fi
done

echo
echo "=== 2. pip download 详细输出（aiter torch290 das.opt1） ==="
cd "$TMP" || exit 1
pip download --no-deps -d "$TMP" \
  "aiter==0.1.3+das.opt1.dtk2604.torch290.2605211823.gccde69" \
  -i "https://pypi.sourcefind.cn/release/dtk2604/" -v 2>&1 | tail -30

echo
echo "=== 3. pip download 详细输出（aiter das185 torch2100） ==="
pip download --no-deps -d "$TMP" \
  "aiter==0.1.5+das185.dtk2604.torch2100.2608131714.gb0979d" \
  -i "https://pypi.sourcefind.cn/release/dtk2604/" -v 2>&1 | tail -20
