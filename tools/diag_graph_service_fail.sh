#!/usr/bin/env bash
# 诊断：graph 服务启动时的 accelerate/numpy 冲突
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1

echo "===== 1. 前缀里有没有 numpy / accelerate ====="
for P in envs/pylibs_core envs/pylibs envs/pylibs_o3d; do
  echo "--- $P ---"
  ls "$P" 2>/dev/null | head -30
  ls -d "$P"/numpy "$P"/numpy-* "$P"/accelerate* 2>/dev/null
done

echo
echo "===== 2. 系统 numpy 版本与 accelerate 的判定 ====="
python3 - <<'PY'
import numpy, sys
print("  sys.path[:6] =", sys.path[:6])
print("  numpy.__version__ =", numpy.__version__)
print("  numpy.__file__ =", numpy.__file__)
try:
    from importlib.metadata import version
    print("  dist numpy version =", version("numpy"))
except Exception as e:
    print("  dist numpy 查询失败:", e)
try:
    print("  has _core =", hasattr(numpy, "_core"))
except Exception as e:
    print("  _core 检查异常:", e)
PY

echo
echo "===== 3. 带 pipeline 的 PYTHONPATH 时 ====="
PYTHONPATH="$WS/envs/pylibs_core:$WS/third_party/upstream_eval/FlagEvalMM:$WS/scripts" python3 - <<'PY'
import numpy, sys
print("  numpy.__version__ =", numpy.__version__, numpy.__file__)
try:
    from importlib.metadata import version
    print("  dist numpy version =", version("numpy"))
except Exception as e:
    print("  dist numpy 查询失败:", e)
PY

echo
echo "===== 4. accelerate 触发点复现 ====="
PYTHONPATH="$WS/envs/pylibs_core:$WS/third_party/upstream_eval/FlagEvalMM:$WS/scripts" python3 -c "
import accelerate.utils.other as o
print('accelerate import OK')
" 2>&1 | tail -6

echo
echo "===== 5. graph 服务日志开头（看是不是同一个错）====="
head -40 logs/serve_graph_*.log 2>/dev/null | tail -30
