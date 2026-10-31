#!/usr/bin/env bash
# 1) 检查内部源是否有 open3d；2) 若有，装进工作目录内隔离前缀 envs/pylibs；3) 列出官方评测脚本
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
mkdir -p envs/pylibs

source tools/env_net.sh 2>/dev/null || true

echo "===== A) pip index 是否能查到 open3d ====="
for IDX in "https://pypi.sourcefind.cn/release/dtk2604/" "https://pypi.sourcefind.cn/nightly/dtk/" "https://pypi.tuna.tsinghua.edu.cn/simple"; do
  echo "--- index: $IDX ---"
  timeout 60 pip index versions open3d --index-url "$IDX" --trusted-host "$(echo "$IDX" | sed -E 's#https?://([^/]+)/?.*#\1#')" 2>&1 | head -n 6
done

echo
echo "===== B) 尝试安装到隔离前缀（不污染公共 site-packages） ====="
timeout 900 pip install --no-cache-dir --target "$WS/envs/pylibs" \
  --index-url https://pypi.sourcefind.cn/release/dtk2604/ \
  --trusted-host pypi.sourcefind.cn \
  "open3d" 2>&1 | tail -n 25

echo
echo "===== C) 导入验证 ====="
PYTHONPATH="$WS/envs/pylibs:${PYTHONPATH:-}" python3 - <<'PY'
try:
    import open3d as o3d
    print("open3d OK", getattr(o3d, "__version__", "?"))
except Exception as e:
    print("open3d FAIL:", type(e).__name__, e)
PY

echo
echo "===== D) 官方评测脚本清单 ====="
for d in third_party/upstream_eval/RoboTracer/Evaluation third_party/upstream_eval/RoboRefer/Evaluation; do
  echo "--- $d ---"
  ls -la "$d" 2>/dev/null | head -n 25
done
