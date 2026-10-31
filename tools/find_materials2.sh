#!/usr/bin/env bash
# 在 gitlab 的 dataset / space 命名空间里找赛事官方材料
source /root/private_data/workspace/robo/tools/env_net.sh

echo "=== A. 所有含 sugon / 官方味道的组 ==="
curl -sS -m 30 "https://gitlab.scnet.cn:9002/api/v4/groups?per_page=100&search=sugon" 2>/dev/null \
 | python3 -c "import sys,json;d=json.load(sys.stdin);[print(g['id'],g['full_path']) for g in d]" 2>&1 | head -20

echo
echo "=== B. sugon_scnet 组下的所有项目（含 dataset/ space/） ==="
for ns in model dataset space; do
  echo "--- $ns/sugon_scnet ---"
  curl -sS -m 30 "https://gitlab.scnet.cn:9002/api/v4/groups/sugon_scnet/projects?per_page=100&include_subgroups=true" 2>/dev/null \
   | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    for p in d:
        if p['path_with_namespace'].startswith('$ns/'):
            print('   ', p['id'], p['path_with_namespace'])
except Exception as e:
    print('   ERR', e)
" 2>&1 | head -40
done

echo
echo "=== C. 项目名模糊搜：robobrain / embodied / vla / 国芯 ==="
for kw in robobrain embodied vla dcu-eval gxgm 2026bench; do
  echo "--- $kw ---"
  curl -sS -m 25 "https://gitlab.scnet.cn:9002/api/v4/projects?search=$kw&per_page=15" 2>/dev/null \
   | python3 -c "import sys,json;d=json.load(sys.stdin);[print('   ',p['id'],p['path_with_namespace']) for p in d]" 2>&1 | head -8
done

echo
echo "=== D. SCNet 平台：找任务池 / 赛事 API ==="
for u in https://www.scnet.cn/api/aihub/task/pool \
         https://www.scnet.cn/api/task/pool \
         https://www.scnet.cn/api/competition/list \
         https://www.scnet.cn/ui/aihub/image ; do
  code=$(curl -sS -m 20 -o /dev/null -w '%{http_code}' "$u" 2>/dev/null)
  echo "$u -> ${code:-FAIL}"
done
