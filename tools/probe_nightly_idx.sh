#!/usr/bin/env bash
# 1) jsyadmin sglang-0.5.12 镜像 history[53] 全文（找 sglang 安装行 + pip 索引）
# 2) 新公共索引 https://pypi.sourcefind.cn/nightly/dtk/ 的包清单
set -o pipefail
W=/root/private_data/workspace/robo
# shellcheck disable=SC1091
source "$W/tools/env_net.sh" >/dev/null 2>&1
H="appstore.scnet.cn:5000"
REPO="aihub/dcu/jsyadmin/sglang-0.5.12-ubuntu22.04-dtk26.04-py3.10"

echo "===== 1) history 全文中的 sglang / 索引 / torch 行 ====="
TOK=$(curl -sS -m 25 "https://$H/service/token?service=harbor-registry&scope=repository:$REPO:pull" 2>/dev/null \
      | python3 -c 'import sys,json;print(json.load(sys.stdin).get("token",""))' 2>/dev/null)
curl -sS -m 30 -H "Authorization: Bearer $TOK" \
  -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
  "https://$H/v2/$REPO/manifests/v1.0" -o "$W/tmp/m.json" 2>/dev/null
CFG=$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(d["config"]["digest"])' "$W/tmp/m.json" 2>/dev/null)
curl -sS -m 30 -H "Authorization: Bearer $TOK" "https://$H/v2/$REPO/blobs/$CFG" -o "$W/tmp/cfg.json" 2>/dev/null
sed -i -E 's#(https?://)[^:/@]+:[^@/]+@#\1[REDACTED]@#g' "$W/tmp/cfg.json"
python3 - "$W/tmp/cfg.json" <<'PY'
import json,sys,re
d=json.load(open(sys.argv[1]))
print('created =', d.get('created'))
for h in d.get('history',[]):
    cb=h.get('created_by') or ''
    if 'torchversion=2.10.0' in cb:
        print('=== FULL PIP COMMAND (len=%d) ==='%len(cb))
        for part in cb.split(';'):
            p=part.strip()
            if p: print('  ', p[:260])
        print()
    if re.search(r'pip\.conf|index-url|PYPI_URL', cb):
        print('--- pip index 配置段:')
        for part in cb.split(';'):
            p=part.strip()
            if re.search(r'index-url|simple|nightly', p): print('  ', p[:260])
PY

echo
echo "===== 2) 新公共索引 nightly/dtk 包清单 ====="
for pkg in sglang aiter vllm vllm-hcu sglang-kernel sglang-router torch; do
  url="https://pypi.sourcefind.cn/nightly/dtk/+simple/$pkg/"
  code=$(curl -sS -m 30 -o "$W/tmp/idx2.html" -w '%{http_code}' "$url" 2>/dev/null)
  echo "--- $pkg  http=$code"
  if [ "$code" = "200" ]; then
    grep -o '>[^<]*\.whl<' "$W/tmp/idx2.html" 2>/dev/null | sed -e 's/^>//' -e 's/<$//' | sort -u | tail -12
  fi
done
