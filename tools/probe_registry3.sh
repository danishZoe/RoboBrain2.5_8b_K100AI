#!/usr/bin/env bash
# 1) Harbor 匿名取 token 尝试（读镜像 config labels）
# 2) 压测进度
set -o pipefail
W=/root/private_data/workspace/robo
# shellcheck disable=SC1091
source "$W/tools/env_net.sh" >/dev/null 2>&1

H="appstore.scnet.cn:5000"
REPO="aihub/dcu/jsyadmin/sglang0.5.18-ubuntu22.04-dtk2604-py3.10-sf_b075-minimax-h3"
TAG="sglang0.5.18-ubuntu22.04-dtk2604-py3.10-sf_b075-minimax-h3"

echo "===== 1) /v2/ 响应头 ====="
curl -sSI -m 25 "https://$H/v2/" | sed -n '1,20p'

echo
echo "===== 2) token 端点尝试 ====="
TOK=""
for svc in harbor-registry harbor registry docker-registry; do
  for ep in "/service/token?service=$svc&scope=repository:$REPO:pull" \
            "/service/token?service=$svc&scope=repository:$REPO:pull&account=$svc"; do
    out=$(curl -sS -m 25 "https://$H$ep" 2>&1 | head -c 400)
    echo "--- $svc$ep"
    echo "$out" | head -c 300
    echo
    if echo "$out" | grep -q '"token"'; then
      TOK=$(echo "$out" | sed -nE 's/.*"token":"([^"]+)".*/\1/p')
      echo ">>> got token (len=${#TOK})"
      break 2
    fi
  done
done

if [ -n "$TOK" ]; then
  echo
  echo "===== 3) manifest ====="
  curl -sS -m 30 -H "Authorization: Bearer $TOK" \
    -H 'Accept: application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.list.v2+json' \
    "https://$H/v2/$REPO/manifests/$TAG" -o "$W/tmp/manifest.json" -w 'http=%{http_code}\n'
  head -c 1200 "$W/tmp/manifest.json"
  echo
  CFG=$(sed -nE 's/.*"config"[^}]*"digest":"([^"]+)".*/\1/p' "$W/tmp/manifest.json" | head -1)
  if [ -n "$CFG" ]; then
    echo
    echo "===== 4) config labels (config digest=$CFG) ====="
    curl -sS -m 30 -H "Authorization: Bearer $TOK" \
      "https://$H/v2/$REPO/blobs/$CFG" -o "$W/tmp/imgcfg.json" -w 'http=%{http_code}\n'
    python3 - <<'PY' 2>/dev/null || head -c 2000 "$W/tmp/imgcfg.json"
import json
d=json.load(open('/root/private_data/workspace/robo/tmp/imgcfg.json'))
cfg=d.get('config',{})
print('ENV:')
for e in cfg.get('Env',[]):
    print('  ',e[:300])
print('LABELS:')
for k,v in (cfg.get('Labels') or {}).items():
    print('  ',k,'=',str(v)[:400])
print('DIFF_IDS:', d.get('rootfs',{}).get('diff_ids',[])[:2], '...')
PY
  fi
fi

echo
echo "===== 5) 压测进度 ====="
B="$W/submission/bench"
for f in "$B"/round*.jsonl; do [ -e "$f" ] && printf '%-16s %4s lines\n' "$(basename "$f")" "$(wc -l < "$f")"; done
ps -eo pid,etime,cmd | grep -E 'bench_serving' | grep -v grep | cut -c1-140 || true
