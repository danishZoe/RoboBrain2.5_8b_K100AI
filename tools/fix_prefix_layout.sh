#!/usr/bin/env bash
# 拆分隔离前缀：open3d 及其 numpy 2.x 依赖单独放 envs/pylibs_o3d
# 原因：open3d 的依赖链会装 numpy 2.x，若放在 envs/pylibs 会遮蔽系统 numpy 1.25，
#       导致 torch（按 numpy 1.x 编译）导入失败。
# 用途：envs/pylibs_o3d 只给 RoboTracer 判分子进程使用；其余进程用系统 numpy 1.25。
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
SRC="$WS/envs/pylibs"
DST="$WS/envs/pylibs_o3d"
mkdir -p "$DST"

echo "===== 1) 迁移 numpy2 / open3d 到 envs/pylibs_o3d ====="
for pat in 'numpy' 'numpy-*.dist-info' 'numpy.libs' 'open3d' 'open3d-*.dist-info'; do
  for f in "$SRC"/$pat; do
    [ -e "$f" ] || continue
    mv "$f" "$DST"/ && echo "  moved: $(basename "$f")"
  done
done

echo
echo "===== 2) 验证 A: 主进程前缀（envs/pylibs）+ 系统 numpy ====="
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u
PYTHONPATH="$SRC:${PYTHONPATH:-}" python3 - <<'PY'
import numpy, sys
print("  numpy:", numpy.__version__, numpy.__file__)
try:
    import torch
    print("  torch:", torch.__version__, "-> import OK")
except Exception as e:
    print("  torch import FAIL:", type(e).__name__, str(e)[:160])
try:
    from qwen_vl_utils.vision_process import fetch_image
    print("  qwen_vl_utils fetch_image: OK（官方预处理可用）")
except Exception as e:
    print("  qwen_vl_utils FAIL:", type(e).__name__, str(e)[:160])
PY

echo
echo "===== 3) 验证 B: 判分子进程前缀（envs/pylibs_o3d + envs/pylibs）====="
PYTHONPATH="$DST:$SRC:${PYTHONPATH:-}" python3 - <<'PY'
import numpy, sys
print("  numpy:", numpy.__version__)
try:
    import open3d as o3d
    print("  open3d:", o3d.__version__, "-> import OK")
except Exception as e:
    print("  open3d FAIL:", type(e).__name__, str(e)[:200])
try:
    import scipy, scipy.spatial.distance
    print("  scipy:", scipy.__version__, "-> OK")
except Exception as e:
    print("  scipy FAIL:", type(e).__name__, str(e)[:160])
try:
    from PIL import Image
    print("  PIL: OK")
except Exception as e:
    print("  PIL FAIL:", type(e).__name__, str(e)[:160])
PY

echo
echo "===== 4) 记录版本 ====="
PYTHONPATH="$DST:$SRC:${PYTHONPATH:-}" python3 -c "
import importlib.metadata as m
for p in ('open3d','numpy','scipy','pillow','torch-fidelity','tenacity'):
    try:
        print('  %-16s %s' % (p, m.version(p)))
    except Exception as e:
        print('  %-16s N/A' % p)
" 2>/dev/null

echo
echo "===== 5) 前缀体积 ====="
du -sh "$SRC" "$DST"
