#!/usr/bin/env bash
# 核验若干候选镜像的构建栈：created 时间 + torchversion + sglang/vllm/aiter 安装行
set -o pipefail
W=/root/private_data/workspace/robo
# shellcheck disable=SC1091
source "$W/tools/env_net.sh" >/dev/null 2>&1
H="appstore.scnet.cn:5000"

REPOS=(
  "aihub/dcu/jsyadmin/sglang0.5.18-ubuntu22.04-dtk2604-py3.10-sf_b075-minimax-h3"
  "aihub/dcu/jsyadmin/vllm_0.21_sf_b075_d2"
  "aihub/dcu/jsyadmin/tmk_0.21.0-ubuntu22.04-dtk2604-py3.10-sf_b075"
  "aihub/dcu/accoaeakan4/vllm_0.21_sf_b075_d2"
  "aihub/dcu/accoaeakan4/tmk_0.21.0-ubuntu22.04-dtk2604-py3.10-sf_b075"
)

for REPO in "${REPOS[@]}"; do
  echo
  echo "############ $REPO"
  TOK=$(curl -sS -m 25 "https://$H/service/token?service=harbor-registry&scope=repository:$REPO:pull" 2>/dev/null \
        | python3 -c 'import sys,json;print(json.load(sys.stdin).get("token",""))' 2>/dev/null)
  if [ -z "$TOK" ]; then echo "  (no token / repo 不存在)"; continue; fi
  TAGS=$(curl -sS -m 25 -H "Authorization: Bearer $TOK" "https://$H/v2/$REPO/tags/list" 2>/dev/null \
         | python3 -c 'import sys,json
try: print(",".join(json.load(sys.stdin).get("tags") or []))
except Exception: print("")' 2>/dev/null)
  echo "  tags: $TAGS"
  [ -z "$TAGS" ] && continue
  TAG=$(echo "$TAGS" | cut -d, -f1)
  curl -sS -m 30 -H "Authorization: Bearer $TOK" \
    -H 'Accept: application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json' \
    "https://$H/v2/$REPO/manifests/$TAG" -o "$W/tmp/mm.json" 2>/dev/null
  CFG=$(python3 -c 'import json,sys
try:
    d=json.load(open(sys.argv[1])); print(d.get("config",{}).get("digest",""))
except Exception: print("")' "$W/tmp/mm.json" 2>/dev/null)
  [ -z "$CFG" ] && { echo "  (no manifest)"; continue; }
  curl -sS -m 30 -H "Authorization: Bearer $TOK" "https://$H/v2/$REPO/blobs/$CFG" -o "$W/tmp/cc.json" 2>/dev/null
  sed -i -E 's#(https?://)[^:/@]+:[^@/]+@#\1[REDACTED]@#g' "$W/tmp/cc.json"
  python3 - "$W/tmp/cc.json" <<'PY'
import json,sys,re
try: d=json.load(open(sys.argv[1]))
except Exception as e:
    print('  (config parse fail)',e); raise SystemExit
print('  created =', d.get('created'))
c=d.get('config',{})
env=[e for e in c.get('Env',[]) if re.search(r'TORCH|AITER|SGLANG|VLLM|DTK_VERSION|PYTHON',e,re.I)]
for e in env: print('  ENV',e[:200])
for h in d.get('history',[]):
    cb=h.get('created_by') or ''
    if re.search(r'torchversion|pip install.*(sglang|aiter|vllm|torch)', cb, re.I):
        for part in cb.split(';'):
            p=part.strip()
            if re.search(r'torchversion|pip install.*(sglang|aiter|vllm|torch)', p, re.I):
                print('   >', p[:220])
PY
done
