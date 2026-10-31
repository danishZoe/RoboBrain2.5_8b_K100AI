#!/usr/bin/env bash
source /root/private_data/workspace/robo/tools/env_net.sh

mkdir -p /root/private_data/workspace/robo/tmp

echo "=== A. 首页 HTML 里 国芯国模 的上下文 ==="
python3 - <<'PY'
import re
h = open('/root/private_data/workspace/robo/tmp/gh.html', encoding='utf-8', errors='replace').read()
for m in re.finditer('国芯国模', h):
    s = max(0, m.start()-700); e = min(len(h), m.end()+700)
    print('----- context -----')
    print(h[s:e].replace('\\u002F', '/'))
PY

echo
echo "=== B. 正确 group 路径 model/sugon_scnet 下的项目 ==="
for enc in "model%2Fsugon_scnet" "sugon_scnet"; do
  echo "--- $enc ---"
  curl -sS -m 30 "https://gitlab.scnet.cn:9002/api/v4/groups/$enc/projects?per_page=100&include_subgroups=true" 2>/dev/null \
   | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    if isinstance(d,dict):
        print('   ', str(d)[:160])
    else:
        print('   count',len(d))
        for p in d: print('   ',p['id'],p['path_with_namespace'])
except Exception as e:
    print('   ERR', e)
" 2>&1 | head -40
done

echo
echo "=== C. dataset/ 命名空间里有没有官方评测集 ==="
for enc in "dataset%2Fsugon_scnet" "dataset%2Fsugon" ; do
  echo "--- $enc ---"
  curl -sS -m 30 "https://gitlab.scnet.cn:9002/api/v4/groups/$enc/projects?per_page=100&include_subgroups=true" 2>/dev/null \
   | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    if isinstance(d,dict): print('   ',str(d)[:160])
    else:
        for p in d: print('   ',p['id'],p['path_with_namespace'])
except Exception as e: print('   ERR',e)
" 2>&1 | head -30
done
