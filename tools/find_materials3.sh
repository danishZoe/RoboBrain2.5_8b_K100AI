#!/usr/bin/env bash
# 深挖官方材料：gitlab sugon_scnet 组 + 光合社区接口
source /root/private_data/workspace/robo/tools/env_net.sh

mkdir -p /root/private_data/workspace/robo/tmp

echo "=== A. group 624 (sugon_scnet) 原始响应 ==="
curl -sS -m 30 "https://gitlab.scnet.cn:9002/api/v4/groups/sugon_scnet/projects?per_page=100" 2>/dev/null | head -c 600
echo

echo
echo "=== B. group 624 的 subgroups ==="
curl -sS -m 30 "https://gitlab.scnet.cn:9002/api/v4/groups/624/subgroups?per_page=100" 2>/dev/null \
 | python3 -c "import sys,json;d=json.load(sys.stdin);[print(g['id'],g['full_path']) for g in d]" 2>&1 | head -20

echo
echo "=== C. 搜脚手架类仓库 ==="
for kw in inference-cookbook cookbook-das vllm-plugin-das sglang-das submission scaffold template taskpool; do
  printf '%-24s ' "$kw"
  curl -sS -m 25 "https://gitlab.scnet.cn:9002/api/v4/projects?search=$kw&per_page=8" 2>/dev/null \
   | python3 -c "import sys,json;d=json.load(sys.stdin);print(', '.join(p['path_with_namespace'] for p in d) or '-')" 2>&1 | head -3
done

echo
echo "=== D. 光合首页里找国芯国模入口 ==="
curl -sS -m 30 "https://developer.sourcefind.cn/" 2>/dev/null > /root/private_data/workspace/robo/tmp/gh.html
wc -c /root/private_data/workspace/robo/tmp/gh.html
grep -oE '国芯国模|任务池|黑客松' /root/private_data/workspace/robo/tmp/gh.html | sort | uniq -c
grep -oE 'href="[^"]{1,120}"' /root/private_data/workspace/robo/tmp/gh.html | sort -u | head -30

echo
echo "=== E. 光合公开接口探测 ==="
for u in "https://developer.sourcefind.cn/api/article/list" \
         "https://developer.sourcefind.cn/api/activity/list" \
         "https://developer.sourcefind.cn/api/v1/articles" \
         "https://developer.sourcefind.cn/api/news/list" \
         "https://developer.sourcefind.cn/api/forum/topic/list" \
         "https://developer.sourcefind.cn/gateway/article/list" ; do
  code=$(curl -sS -m 20 -o /root/private_data/workspace/robo/tmp/apiout -w '%{http_code}' "$u" 2>/dev/null)
  echo "$u -> ${code:-FAIL}"
  [ "$code" = "200" ] && head -c 300 /root/private_data/workspace/robo/tmp/apiout && echo
done
