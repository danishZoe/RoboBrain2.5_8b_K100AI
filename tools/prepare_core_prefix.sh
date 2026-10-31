#!/usr/bin/env bash
# 建立「通用依赖前缀」 envs/pylibs_core：= envs/pylibs 去掉 numpy* 与 open3d*
# 目的：open3d 的依赖链带 numpy 2.x，会遮蔽系统 numpy 1.25（torch 的编译基线），
#       导致 torch 导入失败。open3d + numpy2 单独放 envs/pylibs_o3d，
#       只给 RoboTracer 判分子进程使用。
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
SRC="$WS/envs/pylibs"
CORE="$WS/envs/pylibs_core"
O3D="$WS/envs/pylibs_o3d"
mkdir -p "$CORE" "$O3D"

echo "===== 1) 复制通用依赖到 envs/pylibs_core（排除 numpy* / open3d*）====="
n=0
for f in "$SRC"/*; do
  b=$(basename "$f")
  case "$b" in
    numpy|numpy.*|numpy.libs|open3d|open3d.*) continue ;;
  esac
  if [ -e "$CORE/$b" ]; then continue; fi
  cp -a "$f" "$CORE/" && n=$((n+1))
done
echo "  复制 $n 项"

echo
echo "===== 2) 把 numpy2 / open3d 从 envs/pylibs 挪到 envs/pylibs_o3d ====="
for pat in 'numpy' 'numpy-*.dist-info' 'numpy.libs' 'open3d' 'open3d-*.dist-info'; do
  for f in "$SRC"/$pat; do
    [ -e "$f" ] || continue
    rm -rf "$O3D/$(basename "$f")"
    mv "$f" "$O3D"/ && echo "  moved: $(basename "$f")"
  done
done

echo
echo "===== 3) 验证 A：通用前缀（系统 numpy 1.25 + torch + 官方预处理）====="
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u
PYTHONPATH="$CORE:${PYTHONPATH:-}" python3 - <<'PY'
import numpy
print("  numpy:", numpy.__version__, "(应为 1.25.0 = 系统版)")
try:
    import torch
    print("  torch:", torch.__version__, "-> OK")
except Exception as e:
    print("  torch FAIL:", type(e).__name__, str(e)[:140])
try:
    from qwen_vl_utils.vision_process import fetch_image
    print("  qwen_vl_utils fetch_image: OK（官方预处理可用）")
except Exception as e:
    print("  qwen_vl_utils FAIL:", type(e).__name__, str(e)[:140])
for m in ("PIL", "pyarrow", "requests", "tqdm", "scipy", "torch_fidelity"):
    try:
        __import__(m)
        print("  %-16s OK" % m)
    except Exception as e:
        print("  %-16s FAIL %s" % (m, str(e)[:90]))
PY

echo
echo "===== 4) 验证 B：TraceSpatial 判分前缀（open3d + numpy2）====="
PYTHONPATH="$O3D:$CORE:${PYTHONPATH:-}" python3 - <<'PY'
try:
    import numpy
    print("  numpy:", numpy.__version__)
    import open3d as o3d
    print("  open3d:", o3d.__version__, "-> OK")
    import scipy.spatial.distance
    print("  scipy.spatial -> OK")
except Exception as e:
    print("  FAIL:", type(e).__name__, str(e)[:200])
PY

echo
echo "===== 5) 体积 ====="
du -sh "$SRC" "$CORE" "$O3D" 2>/dev/null
