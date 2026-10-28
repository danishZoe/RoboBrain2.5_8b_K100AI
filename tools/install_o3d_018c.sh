#!/usr/bin/env bash
# 给 open3d 0.18 隔离前缀补 sklearn 桩件（只为让 open3d.ml 在 import 期可用），再测 TSDF
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONWARNINGS=ignore
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u
V2=$WS/envs/pylibs_o3d_v2

mkdir -p "$V2/sklearn"
cat > "$V2/sklearn/__init__.py" <<'PY'
"""最小 sklearn 桩件。

open3d 0.18 的 ``open3d/__init__.py`` 会无条件 ``import open3d.ml``，
而 ``open3d.ml.datasets`` 在 import 期需要 ``sklearn.neighbors.KDTree`` 这个**符号**。
官方 3D 判分只用 ``open3d.geometry`` 与 ``open3d.pipelines``，完全不使用 ML 模块，
因此这里只放一个"import 期存在、调用即报错"的占位实现，避免为评测引入
几百 MB 的 scikit-learn 依赖。
"""
from . import neighbors  # noqa: F401
PY
cat > "$V2/sklearn/neighbors.py" <<'PY'
class KDTree:
    def __init__(self, *a, **k):
        raise RuntimeError("sklearn 桩件：open3d.ml 的 KDTree 不可用（评测不使用 ML 模块）")
PY

echo "===== TSDF 测试（官方脚本那组参数 voxel=0.02 sdf_trunc=0.08）====="
PYTHONPATH="$V2:$WS/envs/pylibs:$WS/envs/pylibs_core" python3 - <<'PY' 2>&1 | tail -30
import numpy as np, traceback
try:
    import open3d as o3d
    print("open3d", o3d.__version__, "| numpy", np.__version__)
    print("路径:", open3d.__file__)
except Exception:
    traceback.print_exc(); raise SystemExit(1)
try:
    print(o3d.pipelines.integration.ScalableTSDFVolume.__init__.__doc__[:400])
except Exception:
    pass
H, W = 384, 512
intr = o3d.camera.PinholeCameraIntrinsic(W, H, 421.66, 421.66, 253.93, 192.12)
try:
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.zeros((H, W, 3), np.uint8)),
        o3d.geometry.Image(np.full((H, W), 1000.0, np.float32)),
        depth_scale=1000.0, depth_trunc=5.0, convert_rgb_to_intensity=False)
    v = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.02, sdf_trunc=0.08,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
    v.integrate(rgbd, intr, np.linalg.inv(np.eye(4)))
    n = len(v.extract_point_cloud().points)
    print("  官方参数 TSDF points =", n, "[OK]" if n else "[EMPTY]")
except Exception:
    traceback.print_exc()
PY
