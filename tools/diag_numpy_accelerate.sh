#!/usr/bin/env bash
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONPATH="$WS/envs/pylibs_core:$WS/third_party/upstream_eval/FlagEvalMM:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

echo "===== numpy / accelerate 实际来源 ====="
python3 - <<'PY'
import numpy, sys
print("  numpy version:", numpy.__version__)
print("  numpy file   :", numpy.__file__)
print("  has np._core :", hasattr(numpy, "_core"))
print("  has np.core  :", hasattr(numpy, "core"))
try:
    import accelerate
    print("  accelerate   :", accelerate.__version__, accelerate.__file__)
except Exception as e:
    print("  accelerate import FAIL:", type(e).__name__, str(e)[:120])
PY

echo
echo "===== 系统 site-packages 的 numpy ====="
ls -d /usr/local/lib/python3.10/dist-packages/numpy* 2>/dev/null
cat /usr/local/lib/python3.10/dist-packages/numpy/version.py 2>/dev/null | head -n 5

echo
echo "===== accelerate 的那行源码 ====="
grep -n "np_core" /usr/local/lib/python3.10/dist-packages/accelerate/utils/other.py 2>/dev/null | head -n 5

echo
echo "===== pylibs_core 里有没有 numpy ====="
ls -d "$WS/envs/pylibs_core"/numpy* 2>/dev/null || echo "  无（正确）"

echo
echo "===== 逐个前缀验证 ====="
for P in "$WS/envs/pylibs_core" "$WS/envs/pylibs" "$WS/envs/pylibs_o3d"; do
  echo "--- PYTHONPATH=$P ---"
  PYTHONPATH="$P" python3 -c "import numpy; print('   numpy', numpy.__version__, numpy.__file__)" 2>&1 | tail -n 2
done
echo "--- 无前缀（纯系统）---"
python3 -c "import numpy; print('   numpy', numpy.__version__, numpy.__file__)" 2>&1 | tail -n 2
