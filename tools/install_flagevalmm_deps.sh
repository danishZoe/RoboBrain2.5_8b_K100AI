#!/usr/bin/env bash
# 安装 FlagEvalMM 官方判分链路所需依赖到工作目录内隔离前缀 envs/pylibs
# 不污染公共 site-packages；不改系统库
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
mkdir -p envs/pylibs cache/pip tmp
export PIP_CACHE_DIR="$WS/cache/pip"
export TMPDIR="$WS/tmp"

source tools/env_net.sh 2>/dev/null || true
IDX="https://pypi.sourcefind.cn/release/dtk2604/"
HOST="pypi.sourcefind.cn"

echo "===== 安装 FlagEvalMM 判分链路依赖（隔离前缀 envs/pylibs）====="
timeout 900 pip install --no-cache-dir --target "$WS/envs/pylibs" \
  --index-url "$IDX" --trusted-host "$HOST" \
  tenacity word2number easydict flask mmengine 2>&1 | tail -n 12

echo
echo "===== 验证（source DTK env 后）====="
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u
PYTHONPATH="$WS/third_party/upstream_eval/FlagEvalMM:$WS/envs/pylibs:${PYTHONPATH:-}" python3 - <<'PY'
import importlib, sys
mods = ["flagevalmm.evaluator.pre_process",
        "flagevalmm.evaluator.common_types",
        "flagevalmm.evaluator.point_utils"]
for m in mods:
    try:
        importlib.import_module(m)
        print("  OK  ", m)
    except Exception as e:
        print("  FAIL", m, type(e).__name__, str(e)[:160])
PY

echo
echo "===== open3d 状态 ====="
ls -d "$WS/envs/pylibs/open3d" 2>/dev/null || echo "  未安装"
