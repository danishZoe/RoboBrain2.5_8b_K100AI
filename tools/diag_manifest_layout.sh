#!/usr/bin/env bash
set -uo pipefail
WS=/root/private_data/workspace/robo
echo "=== manifest 目录 ==="
ls -la "$WS/outputs/bench_public/manifest/" 2>/dev/null | head -20
echo
echo "=== cvbench_images 目录前 5 个 ==="
ls "$WS/outputs/bench_public/manifest/cvbench_images/" 2>/dev/null | head -5
echo
echo "=== map 文件结构 ==="
python3 -c "
import json
p='$WS/outputs/bench_public/manifest/cvbench_image_map.json'
import os
print('exists', os.path.exists(p))
d=json.load(open(p,encoding='utf-8'))
print('type', type(d), 'len', len(d))
print('first:', json.dumps(d[0] if isinstance(d,list) else list(d.items())[0], ensure_ascii=False)[:400])
"
echo
echo "=== annotations.json 第一条 ==="
python3 -c "
import json
d=json.load(open('$WS/outputs/bench_public/cvbench/annotations.json',encoding='utf-8'))
k=list(d)[0]
print(k, json.dumps(d[k], ensure_ascii=False)[:600])
"
