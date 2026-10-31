#!/usr/bin/env bash
# SCNet 镜像库：找 SGLang 镜像 tag（含 aiter 完整性的 das185/torch2100 栈）
source /root/private_data/workspace/robo/tools/env_net.sh
W=/root/private_data/workspace/robo
TMP="$W/tmp/scnet_image_probe"
mkdir -p "$TMP"; cd "$TMP" || exit 1

echo "=== 1. 从 aihub 的 JS bundle 里找镜像列表 API 路径 ==="
curl -sS -m 60 -L "https://www.scnet.cn/ui/aihub/" 2>/dev/null > home.html
for js in $(grep -oE '/ui/aihub/_nuxt/[A-Za-z0-9_.-]+\.js' home.html | sort -u | head -40); do
  curl -sS -m 40 -L "https://www.scnet.cn$js" 2>/dev/null >> all.js
done
echo "  抓取 JS 总量: $(wc -c < all.js) bytes"
echo "  --- 含 image 的 API 路径候选 ---"
grep -oE '"/[a-zA-Z0-9_/.-]*(image|Image)[a-zA-Z0-9_/.-]*"' all.js 2>/dev/null | sort -u | head -25
echo "  --- 含 /acx/ 的路径候选 ---"
grep -oE '"/acx/[a-zA-Z0-9_/.-]{3,80}"' all.js 2>/dev/null | sort -u | head -25

echo
echo "=== 2. 试探常见镜像列表接口 ==="
for u in "https://www.scnet.cn/acx/resource/image/list" \
         "https://www.scnet.cn/acx/resource/images" \
         "https://www.scnet.cn/acx/resource/aihub/image/list" \
         "https://www.scnet.cn/acx/resource/imagehub/list" \
         "https://www.scnet.cn/acx/resource/sites/imagelist" ; do
  code=$(curl -sS -m 25 -o out.json -w '%{http_code}' "$u" 2>/dev/null)
  sz=$(wc -c < out.json 2>/dev/null)
  echo "  $u -> $code (${sz}B)"
  if [ "$code" = "200" ] && [ "${sz:-0}" -gt 40 ]; then head -c 300 out.json; echo; fi
done
