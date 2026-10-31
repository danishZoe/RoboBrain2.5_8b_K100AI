#!/usr/bin/env bash
# 第三轮探测：确定 query(BaseSearchQuery) / sort 的具体结构
set -o pipefail
W=/root/private_data/workspace/robo
# shellcheck disable=SC1091
source "$W/tools/env_net.sh" >/dev/null 2>&1

API="https://www.scnet.cn/acx/search/dolphin/image/full-text"

bodies=(
  '{"query":{"keyword":"sglang"},"sort":["updateTime"]}'
  '{"query":{"keyword":"sglang"},"sort":{"updateTime":"desc"}}'
  '{"query":{"text":"sglang"},"sort":["updateTime"]}'
  '{"query":{"query":"sglang"},"sort":["updateTime"]}'
  '{"query":{"searchKey":"sglang"},"sort":["updateTime"]}'
  '{"query":{"words":"sglang"},"sort":["updateTime"]}'
  '{"query":{"key":"sglang"},"sort":["updateTime"]}'
  '{"query":{"value":"sglang"},"sort":["updateTime"]}'
)

for b in "${bodies[@]}"; do
  code=$(curl -sS -m 40 -o "$W/tmp/scnet_img_api.json" -w '%{http_code}' \
         -X POST "$API" -H 'Content-Type: application/json' \
         -H 'Accept: application/json' \
         -H 'Origin: https://www.scnet.cn' \
         -H 'Referer: https://www.scnet.cn/ui/aihub/image' \
         --data-raw "$b" 2>&1)
  short=$(head -c 400 "$W/tmp/scnet_img_api.json" | tr -d '\n')
  printf '%s\n  -> %s | %s\n' "$b" "$code" "$short"
done
