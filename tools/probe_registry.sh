#!/usr/bin/env bash
source /root/private_data/workspace/robo/tools/env_net.sh
TMP=/root/private_data/workspace/robo/tmp/scnet_image_probe
mkdir -p "$TMP"; cd "$TMP" || exit 1

echo "=== 1. 试探 SCNet 容器 registry 端点 ==="
for host in image.scnet.cn registry.scnet.cn docker.scnet.cn harbor.scnet.cn scnet.cn; do
  code=$(curl -sS -m 20 -o /dev/null -w '%{http_code}' "https://$host/v2/" 2>/dev/null)
  echo "  https://$host/v2/ -> ${code:-FAIL}"
done

echo
echo "=== 2. 若 registry 可达，列举 sglang 镜像 tag ==="
for host in image.scnet.cn registry.scnet.cn; do
  for repo in "jupyter-sglang5.12-ubuntu22.04-dtk26.04-py3.10-qwen3.6" "sglang" "library/sglang"; do
    u="https://$host/v2/$repo/tags/list"
    code=$(curl -sS -m 25 -o out.json -w '%{http_code}' "$u" 2>/dev/null)
    if [ "$code" = "200" ]; then
      echo "  ✓ $u"
      head -c 1200 out.json; echo
    else
      echo "  $u -> ${code:-FAIL}"
    fi
  done
done

echo
echo "=== 3. 公开的 aihub 镜像搜索页 HTML（SPA 预渲染数据？） ==="
for u in "https://www.scnet.cn/ui/aihub/image" "https://www.scnet.cn/ui/aihub/images"; do
  curl -sS -m 40 -L "$u" 2>/dev/null > p.html
  echo "  $u  size=$(wc -c < p.html)  含 sglang=$(grep -ioc sglang p.html)"
  grep -oiE '[a-z0-9.-]*sglang[a-z0-9.:_-]{0,60}' p.html 2>/dev/null | sort -u | head -12
done

echo
echo "=== 4. 压测状态 ==="
date +%H:%M:%S
wc -l /root/private_data/workspace/robo/submission/bench/round*.jsonl 2>/dev/null
