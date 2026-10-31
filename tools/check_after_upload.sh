#!/usr/bin/env bash
set -uo pipefail
WS=/root/private_data/workspace/robo
cd "$WS"
export PYTHONPATH="$WS/envs/pylibs_core:$WS/scripts:${PYTHONPATH:-}"
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

echo "=== py_compile ==="
python3 -m py_compile submission/bench_public/common.py && echo "COMPILE_OK"

echo "=== 协议列表 ==="
python3 - <<'PY'
import sys
sys.path.insert(0, "/root/private_data/workspace/robo/submission/bench_public")
import common as C
print(C.IMAGE_PROTOCOLS)
PY

echo "=== 与基线 common.py 的差异（应只有新增分支） ==="
diff -u submission/bench_public/common.py.bak_20261028 submission/bench_public/common.py | head -80

echo "=== 3D 判分进度 ==="
echo "剩余 summarize_acc 进程: $(ps -eo args | grep -c '[s]ummarize_acc.py')"
ls -la outputs/bench_public/tracespatial/*/scores_3dchunk_3D.json 2>/dev/null
ls -la outputs/bench_public_noneager/tracespatial/*/scores_3dchunk_3D.json 2>/dev/null

echo "=== 服务健康 ==="
curl -s -o /dev/null -w "health=%{http_code}\n" http://127.0.0.1:30000/health
