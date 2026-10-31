#!/usr/bin/env bash
# 1) config.created + history[53] 全文
# 2) 逐 layer 流式扫描 dist-info，读取镜像内实际安装的 aiter/sglang/torch/vllm 版本（不启动镜像）
set -o pipefail
W=/root/private_data/workspace/robo
# shellcheck disable=SC1091
source "$W/tools/env_net.sh" >/dev/null 2>&1
H="appstore.scnet.cn:5000"
REPO="aihub/dcu/jsyadmin/sglang-0.5.12-ubuntu22.04-dtk26.04-py3.10"
TAG=v1.0

TOK=$(curl -sS -m 25 "https://$H/service/token?service=harbor-registry&scope=repository:$REPO:pull" 2>/dev/null \
      | python3 -c 'import sys,json;print(json.load(sys.stdin).get("token",""))' 2>/dev/null)
echo "token_len=${#TOK}"

echo
echo "===== 1) config.created + 关键 history 全文 ====="
curl -sS -m 30 -H "Authorization: Bearer $TOK" \
  -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
  "https://$H/v2/$REPO/manifests/$TAG" -o "$W/tmp/m.json" -w 'manifest_http=%{http_code}\n'
CFG=$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(d["config"]["digest"])' "$W/tmp/m.json")
python3 - "$W/tmp/m.json" <<'PY' > "$W/tmp/layers.txt"
import json,sys
d=json.load(open(sys.argv[1]))
for i,l in enumerate(d.get('layers',[])):
    print(i,l['size'],l['digest'])
PY
curl -sS -m 30 -H "Authorization: Bearer $TOK" "https://$H/v2/$REPO/blobs/$CFG" -o "$W/tmp/cfg.json" -w 'config_http=%{http_code}\n'
sed -i -E 's#(https?://)[^:/@]+:[^@/]+@#\1[REDACTED]@#g' "$W/tmp/cfg.json"
python3 - "$W/tmp/cfg.json" <<'PY'
import json,sys,re
d=json.load(open(sys.argv[1]))
print('created =', d.get('created'))
print('architecture =', d.get('architecture'), ' os =', d.get('os'))
print()
print('--- 含 sglang/aiter 的完整 history 段落 ---')
for h in d.get('history',[]):
    cb=(h.get('created_by') or '')
    if re.search(r'sglang|aiter', cb, re.I):
        # 把长命令按 ';' 拆开逐条打印
        print('### HISTORY ENTRY (len=%d)'%len(cb))
        for part in cb.split(';'):
            if part.strip():
                print('   ', part.strip()[:300])
        print()
PY

echo
echo "===== 2) layer 流式扫描（找 dist-info） ====="
PAT='(aiter|sglang|torch|vllm|flashinfer|sgl_kernel)[^/]*/METADATA'
# 先小后大
ORDER=$(sort -k2 -n "$W/tmp/layers.txt" | awk '{print $1" "$3}')
found=0
while read -r idx dg; do
  [ -z "$dg" ] && continue
  sz=$(awk -v i="$idx" '$1==i{print $2}' "$W/tmp/layers.txt")
  echo "--- layer $idx size=$sz"
  out=$(timeout 300 bash -c "curl -sS -m 280 -H 'Authorization: Bearer $TOK' 'https://$H/v2/$REPO/blobs/$dg' | tar -tz 2>/dev/null | grep -E '$PAT' | head -12" 2>/dev/null)
  if [ -n "$out" ]; then
    echo "$out" | sed 's/^/    /'
    found=$((found+1))
  else
    echo "    (no match)"
  fi
  [ "$found" -ge 4 ] && { echo "(stopping after 4 layers with matches)"; break; }
done <<< "$ORDER"

echo
echo "===== 3) 压测进度 ====="
B="$W/submission/bench"
for f in "$B"/round*.jsonl; do [ -e "$f" ] && printf '%-16s %4s lines\n' "$(basename "$f")" "$(wc -l < "$f")"; done
tail -c 300 "$B/round2_16384_1024_c64.log" 2>/dev/null | tr '\r' '\n' | tail -2
