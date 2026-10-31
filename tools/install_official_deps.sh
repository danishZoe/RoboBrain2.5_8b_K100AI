#!/usr/bin/env bash
# 安装：1) torch_fidelity（FlagEvalMM 判分链路）2) open3d（TraceSpatial 3D 判分硬依赖）
# 全部装进工作目录内隔离前缀 envs/pylibs，不污染公共 site-packages
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
mkdir -p envs/pylibs cache/pip tmp
export PIP_CACHE_DIR="$WS/cache/pip"
export TMPDIR="$WS/tmp"

source tools/env_net.sh 2>/dev/null || true

echo "##### 1) torch_fidelity (no-deps) #####"
timeout 600 pip install --no-cache-dir --no-deps --target "$WS/envs/pylibs" \
  --index-url https://pypi.sourcefind.cn/release/dtk2604/ \
  --trusted-host pypi.sourcefind.cn torch_fidelity 2>&1 | tail -n 6

echo
echo "##### 2) open3d: 先查各源可用版本 #####"
for IDX in "https://pypi.sourcefind.cn/release/dtk2604/" \
           "https://pypi.sourcefind.cn/nightly/dtk/" \
           "https://pypi.tuna.tsinghua.edu.cn/simple"; do
  HOST=$(echo "$IDX" | sed -E 's#https?://([^/]+)/?.*#\1#')
  echo "--- $IDX ---"
  timeout 120 pip index versions open3d --index-url "$IDX" --trusted-host "$HOST" 2>&1 | head -n 4
done

echo
echo "##### 3) 尝试安装 open3d（内部源优先，失败则清华源）#####"
ok=0
for IDX in "https://pypi.sourcefind.cn/release/dtk2604/" \
           "https://pypi.tuna.tsinghua.edu.cn/simple"; do
  HOST=$(echo "$IDX" | sed -E 's#https?://([^/]+)/?.*#\1#')
  echo "--- install from $IDX ---"
  if timeout 1800 pip install --no-cache-dir --target "$WS/envs/pylibs" \
      --index-url "$IDX" --trusted-host "$HOST" open3d 2>&1 | tail -n 15; then
    if [ -d "$WS/envs/pylibs/open3d" ]; then ok=1; break; fi
  fi
done
echo "open3d_installed=$ok"

echo
echo "##### 4) 验证 #####"
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u
PYTHONPATH="$WS/envs/pylibs:${PYTHONPATH:-}" python3 - <<'PY'
import importlib.util as u
for m in ("open3d", "torch_fidelity"):
    try:
        print(f"  {m}: {u.find_spec(m) is not None}")
    except Exception as e:
        print(f"  {m}: ERR {e}")
try:
    import open3d as o3d
    print("  open3d version:", o3d.__version__)
except Exception as e:
    print("  open3d import FAIL:", type(e).__name__, str(e)[:200])
PY
