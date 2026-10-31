#!/usr/bin/env bash
# 探测 SCNet 镜像库搜索 API（只读），尝试不同请求体形状
set -o pipefail
W=/root/private_data/workspace/robo
# shellcheck disable=SC1091
source "$W/tools/env_net.sh" >/dev/null 2>&1

API="https://www.scnet.cn/acx/search/dolphin/image/full-text"
echo "proxy: ${https_proxy:-${HTTPS_PROXY:-none}}"

bodies=(
  '{"keyword":"sglang","page":1,"pageSize":50}'
  '{"keyword":"sglang","currentPage":1,"pageSize":50}'
  '{"keyword":"sglang","pageNum":1,"pageSize":50}'
  '{"keyword":"sglang","page":1,"size":50}'
  '{"keyword":"sglang"}'
)

for b in "${bodies[@]}"; do
  echo
  echo "=== body: $b"
  code=$(curl -sS -m 40 -o "$W/tmp/scnet_img_api.json" -w '%{http_code}' \
         -X POST "$API" \
         -H 'Content-Type: application/json' \
         -H 'Accept: application/json' \
         -H 'Origin: https://www.scnet.cn' \
         -H 'Referer: https://www.scnet.cn/ui/aihub/image' \
         --data-raw "$b" 2>&1)
  echo "http_code=$code"
  head -c 1200 "$W/tmp/scnet_img_api.json" 2>/dev/null
  echo
done
