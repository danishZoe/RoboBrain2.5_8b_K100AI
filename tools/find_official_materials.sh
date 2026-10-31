#!/usr/bin/env bash
# 定位官方材料：gitlab 组/项目清单 + 光合社区入口
source /root/private_data/workspace/robo/tools/env_net.sh

echo "=== A. gitlab 全部组 ==="
curl -sS -m 30 "https://gitlab.scnet.cn:9002/api/v4/groups?per_page=100" 2>/dev/null \
 | python3 -c "import sys,json;d=json.load(sys.stdin);[print(g['id'], g['path'], '|', g.get('full_path')) for g in d]" 2>&1 | head -40

echo
echo "=== B. 全部可见项目（按模型名/基线/评测关键词过滤） ==="
curl -sS -m 30 "https://gitlab.scnet.cn:9002/api/v4/projects?per_page=100&simple=true&order_by=last_activity_at" 2>/dev/null \
 | python3 -c "import sys,json;d=json.load(sys.stdin);print('total',len(d));[print(p['id'], p['path_with_namespace']) for p in d]" 2>&1 | head -60

echo
echo "=== C. 搜索关键词 ==="
for kw in baseline judge RoboBrain bench eval material T02; do
  echo "--- $kw ---"
  curl -sS -m 25 "https://gitlab.scnet.cn:9002/api/v4/projects?search=$kw&per_page=20" 2>/dev/null \
   | python3 -c "import sys,json;d=json.load(sys.stdin);[print('   ',p['id'],p['path_with_namespace']) for p in d]" 2>&1 | head -12
done

echo
echo "=== D. 光合开发者社区：任务池/赛题页探测 ==="
for u in https://developer.sourcefind.cn/codes \
         https://developer.sourcefind.cn/competition \
         https://developer.sourcefind.cn/api/v4/projects ; do
  code=$(curl -sS -m 25 -o /dev/null -w '%{http_code}' "$u" 2>/dev/null)
  echo "$u -> ${code:-FAIL}"
done
