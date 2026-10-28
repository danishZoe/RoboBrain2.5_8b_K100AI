#!/usr/bin/env bash
# open3d 就位：把 open3d + numpy2 集中到 envs/pylibs_o3d，envs/pylibs 去掉 numpy
# 判分子进程路径顺序：pylibs_o3d(numpy2+open3d) -> pylibs(open3d 的其它依赖) -> pylibs_core(通用)
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
SRC="$WS/envs/pylibs"
O3D="$WS/envs/pylibs_o3d"
CORE="$WS/envs/pylibs_core"
mkdir -p "$O3D"

echo "===== 1) open3d 归位到 envs/pylibs_o3d ====="
for pat in 'open3d' 'open3d-*.dist-info'; do
  for f in "$SRC"/$pat; do
    [ -e "$f" ] || continue
    rm -rf "$O3D/$(basename "$f")"
    mv "$f" "$O3D"/ && echo "  moved: $(basename "$f")"
  done
done

echo
echo "===== 2) envs/pylibs 里去掉 numpy（避免遮蔽）====="
for pat in 'numpy' 'numpy-*.dist-info' 'numpy.libs'; do
  for f in "$SRC"/$pat; do
    [ -e "$f" ] || continue
    rm -rf "$O3D/$(basename "$f")" 2>/dev/null || true
    mv "$f" "$O3D"/ 2>/dev/null && echo "  moved to o3d: $(basename "$f")" \
      || { rm -rf "$f" && echo "  removed from pylibs: $(basename "$f")"; }
  done
done

echo
echo "===== 3) 验证判分子进程前缀 ====="
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u
PYTHONPATH="$O3D:$SRC:$CORE:${PYTHONPATH:-}" python3 - <<'PY'
import numpy
print("  numpy:", numpy.__version__, numpy.__file__)
try:
    import open3d as o3d
    print("  open3d:", o3d.__version__, "-> OK")
except Exception as e:
    print("  open3d FAIL:", type(e).__name__, str(e)[:300])
for m in ("scipy", "scipy.spatial.distance", "PIL", "cv2"):
    try:
        __import__(m)
        print("  %-22s OK" % m)
    except Exception as e:
        print("  %-22s FAIL %s" % (m, str(e)[:110]))
PY

echo
echo "===== 4) 验证通用前缀未被污染 ====="
PYTHONPATH="$CORE:${PYTHONPATH:-}" python3 -c "
import numpy, torch
print('  numpy', numpy.__version__, '| torch', torch.__version__)
"

echo
echo "===== 5) 体积 ====="
du -sh "$SRC" "$CORE" "$O3D"
