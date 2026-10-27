#!/usr/bin/env bash
# 第二轮探测：补齐 query / sort 字段
set -o pipefail
W=/root/private_data/workspace/robo
# shellcheck disable=SC1091
source "$W/tools/env_net.sh" >/dev/null 2>&1

API="https://www.scnet.cn/acx/search/dolphin/image/full-text"

bodies=(
  '{"query":"sglang","sort":"updateTime","page":1,"pageSize":50}'
  '{"query":"sglang","sort":{"updateTime":"desc"},"page":1,"pageSize":50}'
  '{"query":"sglang","sort":[{"updateTime":"desc"}],"page":1,"pageSize":50}'
  '{"query":"sglang","sort":"update_time","page":1,"pageSize":50}'
  '{"query":"sglang","sort":"updateTime desc","page":1,"pageSize":50}'
  '{"query":"sglang","sort":"name","page":1,"pageSize":50}'
)

for b in "${bodies[@]}"; do
  echo
  echo "=== body: $b"
  code=$(curl -sS -m 40 -o "$W/tmp/scnet_img_api.json" -w '%{http_code}' \
         -X POST "$API" -H 'Content-Type: application/json' \
         -H 'Accept: application/json' \
         -H 'Origin: https://www.scnet.cn' \
         -H 'Referer: https://www.scnet.cn/ui/aihub/image' \
         --data-raw "$b" 2>&1)
  echo "http_code=$code"
  head -c 1500 "$W/tmp/scnet_img_api.json" 2>/dev/null
  echo
done
