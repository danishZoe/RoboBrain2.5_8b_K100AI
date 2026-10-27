#!/usr/bin/env bash
# 1) 清理带明文凭据的临时文件
# 2) jsyadmin 官方 sglang 镜像：tag 清单 + 完整 layer 列表 + 完整 build history
set -o pipefail
W=/root/private_data/workspace/robo
# shellcheck disable=SC1091
source "$W/tools/env_net.sh" >/dev/null 2>&1
H="appstore.scnet.cn:5000"

echo "===== 1) 清理含凭据的临时文件 ====="
for f in "$W/tmp/cfg.json" "$W/tmp/imgcfg.json" "$W/tmp/manifest.json" "$W/tmp/m.json" "$W/tmp/m2.json"; do
  [ -e "$f" ] || continue
  sed -i -E 's#(https?://)[^:/@]+:[^@/]+@#\1[REDACTED]@#g' "$f" && echo "redacted: $f"
done

REPO="aihub/dcu/jsyadmin/sglang-0.5.12-ubuntu22.04-dtk26.04-py3.10"

echo
echo "===== 2) token + tags ====="
TOK=$(curl -sS -m 25 "https://$H/service/token?service=harbor-registry&scope=repository:$REPO:pull" 2>/dev/null \
      | python3 -c 'import sys,json;print(json.load(sys.stdin).get("token",""))' 2>/dev/null)
echo "token_len=${#TOK}"
curl -sS -m 25 -H "Authorization: Bearer $TOK" "https://$H/v2/$REPO/tags/list" -o "$W/tmp/tags.json" -w 'tags_http=%{http_code}\n'
python3 - "$W/tmp/tags.json" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    print('tags:', d.get('tags'))
except Exception as e:
    print('parse fail', e, open(sys.argv[1]).read()[:300])
PY

echo
echo "===== 3) 完整 layer 列表 (tag=v1.0) ====="
curl -sS -m 30 -H "Authorization: Bearer $TOK" \
  -H 'Accept: application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json' \
  "https://$H/v2/$REPO/manifests/v1.0" -o "$W/tmp/m.json" -w 'http=%{http_code}\n'
python3 - "$W/tmp/m.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
ls=d.get('layers',[])
tot=sum(l.get('size',0) for l in ls)
print('layer_count=',len(ls),' total_bytes=',tot, ' (%.2f GB)'%(tot/1024**3))
for i,l in enumerate(ls):
    print('  %2d  %12d  %s'%(i,l.get('size',0),l.get('digest')))
cfg=d.get('config',{}).get('digest','')
open('/tmp/cfgdigest.txt','w').write(cfg)
print('config_digest=',cfg)
PY

CFG=$(cat /tmp/cfgdigest.txt 2>/dev/null)
echo
echo "===== 4) 完整 build history ====="
curl -sS -m 30 -H "Authorization: Bearer $TOK" "https://$H/v2/$REPO/blobs/$CFG" -o "$W/tmp/cfg.json" -w 'config_http=%{http_code}\n'
sed -i -E 's#(https?://)[^:/@]+:[^@/]+@#\1[REDACTED]@#g' "$W/tmp/cfg.json"
python3 - "$W/tmp/cfg.json" <<'PY'
import json,sys,re
d=json.load(open(sys.argv[1]))
c=d.get('config',{})
print('--- 关键 ENV')
for e in c.get('Env',[]):
    if re.search(r'TORCH|AITER|SGLANG|VLLM|DTK_VERSION|PYTHON_VER|CONDA|PIP_INDEX', e, re.I):
        print('  ',e[:300])
print('--- 全部 history（截断到 1500 字符/条）')
for i,h in enumerate(d.get('history',[])):
    cb=(h.get('created_by') or '')
    if len(cb)>1500: cb=cb[:1500]+' ...[TRUNC]'
    print('[%02d] %s'%(i,cb))
    print()
PY

echo
echo "===== 5) 压测进度 ====="
B="$W/submission/bench"
for f in "$B"/round*.jsonl; do [ -e "$f" ] && printf '%-16s %4s lines\n' "$(basename "$f")" "$(wc -l < "$f")"; done
tail -c 400 "$B/round2_16384_1024_c64.log" 2>/dev/null | tr '\r' '\n' | tail -3
