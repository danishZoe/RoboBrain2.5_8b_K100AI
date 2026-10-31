#!/usr/bin/env bash
# 第四轮：确定分页字段与可用 sort 字段
set -o pipefail
W=/root/private_data/workspace/robo
# shellcheck disable=SC1091
source "$W/tools/env_net.sh" >/dev/null 2>&1

API="https://www.scnet.cn/acx/search/dolphin/image/full-text"

bodies=(
  '{"query":{"keyword":"sglang"},"sort":{"updateTime":"desc"},"page":1,"pageSize":50}'
  '{"query":{"keyword":"sglang"},"sort":{"createTime":"desc"},"page":1,"pageSize":50}'
  '{"query":{"keyword":"sglang"},"sort":{"name":"asc"},"page":1,"pageSize":50}'
  '{"query":{"keyword":"sglang"},"sort":{"downloadCount":"desc"},"page":1,"pageSize":50}'
  '{"query":{"keyword":"sglang"},"sort":{"updateTime":"desc"},"pageNum":1,"pageSize":50}'
  '{"query":{"keyword":"sglang"},"sort":{"updateTime":"desc"},"current":1,"size":50}'
  '{"query":{"keyword":"sglang"},"sort":{"updateTime":"desc"},"pageNo":1,"pageSize":50}'
  '{"query":{"keyword":"jupyter"},"sort":{"name":"asc"},"page":1,"pageSize":50}'
)

for b in "${bodies[@]}"; do
  code=$(curl -sS -m 40 -o "$W/tmp/scnet_img_api.json" -w '%{http_code}' \
         -X POST "$API" -H 'Content-Type: application/json' \
         -H 'Accept: application/json' \
         -H 'Origin: https://www.scnet.cn' \
         -H 'Referer: https://www.scnet.cn/ui/aihub/image' \
         --data-raw "$b" 2>&1)
  short=$(head -c 300 "$W/tmp/scnet_img_api.json" | tr -d '\n')
  printf '%s\n  -> %s | %s\n\n' "$b" "$code" "$short"
done
