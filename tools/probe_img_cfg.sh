#!/usr/bin/env bash
# A) docker 可用性核查
# B) Harbor 匿名 token -> manifest -> config(Env/Labels/history)，对比三个镜像仓库
# C) 压测进度
set -o pipefail
W=/root/private_data/workspace/robo
# shellcheck disable=SC1091
source "$W/tools/env_net.sh" >/dev/null 2>&1
H="appstore.scnet.cn:5000"

echo "===== A) docker / 容器运行时可用性 ====="
for c in docker podman nerdctl ctr crictl skopeo crane; do
  p=$(command -v "$c" 2>/dev/null) && echo "FOUND $c -> $p" || echo "missing $c"
done
ls -la /var/run/docker.sock 2>/dev/null || echo "no /var/run/docker.sock"
ls -la /run/containerd/containerd.sock 2>/dev/null || echo "no containerd.sock"
echo "--- cgroup (容器内?)"
head -3 /proc/1/cgroup
echo "--- 设备节点"
ls /dev | grep -iE 'kfd|dri|hsa|render' || echo "no dcu device nodes in /dev"

echo
echo "===== B) Harbor token ====="
TOK=""
REPOS=(
  "aihub/dcu/accoaeakan4/sglang-0.5.12-ubuntu22.04-dtk26.04-py3.10"
  "aihub/dcu/jsyadmin/sglang-0.5.12-ubuntu22.04-dtk26.04-py3.10"
  "aihub/dcu/zhr5535/jupyter-sglang5.12-ubuntu22.04-dtk26.04-py3.10-qwen3.6"
)
for svc in harbor-registry; do
  for R in "${REPOS[@]}"; do
    body=$(curl -sS -m 25 "https://$H/service/token?service=$svc&scope=repository:$R:pull" 2>/dev/null)
    t=$(printf '%s' "$body" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("token",""))' 2>/dev/null)
    echo "repo=$R  token_len=${#t}"
    if [ -n "$t" ]; then TOK="$t"; REPO="$R"; break 2; fi
  done
done
if [ -z "$TOK" ]; then echo "!! no token"; else echo "token ok (len=${#TOK})"; fi

fetch_cfg() {
  local repo="$1" tag="$2"
  echo
  echo "=========== $repo : $tag ==========="
  curl -sS -m 30 -H "Authorization: Bearer $TOK" \
    -H 'Accept: application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.oci.image.index.v1+json' \
    "https://$H/v2/$repo/manifests/$tag" -o "$W/tmp/m.json" -w 'manifest_http=%{http_code}\n'
  head -c 300 "$W/tmp/m.json"; echo
  python3 - "$W/tmp/m.json" <<'PY' 2>/dev/null
import json,sys
d=json.load(open(sys.argv[1]))
print('mediaType=',d.get('mediaType'))
if 'manifests' in d:
    for m in d['manifests'][:6]:
        p=m.get('platform',{})
        print('  sub',m.get('digest'),p.get('os'),p.get('architecture'),m.get('mediaType'))
if 'layers' in d:
    for l in d['layers'][:12]:
        print('  layer',l.get('size'),l.get('digest'))
PY
  CFG=$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(d.get("config",{}).get("digest",""))' "$W/tmp/m.json" 2>/dev/null)
  if [ -z "$CFG" ] && grep -q '"manifests"' "$W/tmp/m.json"; then
    SUB=$(python3 -c '
import json,sys
d=json.load(open(sys.argv[1]))
for m in d["manifests"]:
    p=m.get("platform",{})
    if p.get("architecture") in ("amd64","x86_64"): print(m["digest"]);break
' "$W/tmp/m.json" 2>/dev/null)
    [ -n "$SUB" ] && {
      curl -sS -m 30 -H "Authorization: Bearer $TOK" \
        -H 'Accept: application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json' \
        "https://$H/v2/$repo/manifests/$SUB" -o "$W/tmp/m2.json" -w 'sub_manifest_http=%{http_code}\n'
      CFG=$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(d.get("config",{}).get("digest",""))' "$W/tmp/m2.json" 2>/dev/null)
      python3 -c '
import json,sys
d=json.load(open(sys.argv[1]))
for l in d.get("layers",[])[:12]: print("  layer",l.get("size"),l.get("digest"))
' "$W/tmp/m2.json" 2>/dev/null
    }
  fi
  [ -z "$CFG" ] && { echo "no config digest"; return; }
  curl -sS -m 30 -H "Authorization: Bearer $TOK" "https://$H/v2/$repo/blobs/$CFG" -o "$W/tmp/cfg.json" -w 'config_http=%{http_code}\n'
  python3 - "$W/tmp/cfg.json" <<'PY' 2>/dev/null
import json,sys
d=json.load(open(sys.argv[1]))
c=d.get('config',{})
print('--- ENV')
for e in c.get('Env',[]):
    print('  ',e[:400])
print('--- LABELS')
for k,v in (c.get('Labels') or {}).items():
    print('  ',k,'=',str(v)[:400])
print('--- HISTORY (pip/aiter/sglang/torch 相关)')
for h in d.get('history',[]):
    cb=(h.get('created_by') or '')
    if any(k in cb.lower() for k in ('pip','aiter','sglang','torch','dtk','wheel','install')):
        print('  *',cb[:500])
PY
}

if [ -n "$TOK" ]; then
  fetch_cfg "aihub/dcu/accoaeakan4/sglang-0.5.12-ubuntu22.04-dtk26.04-py3.10" "v1.0"
  fetch_cfg "aihub/dcu/jsyadmin/sglang-0.5.12-ubuntu22.04-dtk26.04-py3.10" "v1.0"
  fetch_cfg "aihub/dcu/zhr5535/jupyter-sglang5.12-ubuntu22.04-dtk26.04-py3.10-qwen3.6" "v1.0"
fi

echo
echo "===== C) 压测进度 ====="
B="$W/submission/bench"
for f in "$B"/round*.jsonl; do [ -e "$f" ] && printf '%-16s %4s lines\n' "$(basename "$f")" "$(wc -l < "$f")"; done
ps -eo pid,etime,cmd | grep -E 'bench_serving|pipeline_after_bench|dryrun' | grep -v grep | cut -c1-120 || true
