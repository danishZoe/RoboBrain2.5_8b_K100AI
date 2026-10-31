#!/usr/bin/env bash
# 用直接下载 wheel 的方式装 open3d 0.18.0，并顺带用 sklearn 桩件测 0.19.0
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
source "$WS/tools/env_net.sh"
export PYTHONWARNINGS=ignore
export PIP_CACHE_DIR="$WS/cache/pip"
export TMPDIR="$WS/tmp"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.sourcefind.cn/release/dtk2604/}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.sourcefind.cn}"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$WS/tmp/wheels"

echo "===== 0. 先测已装好的 open3d 0.19.0（用 sklearn 桩件绕过 open3d.ml）====="
V2=$WS/envs/pylibs_o3d_v2
mkdir -p "$V2/sklearn/neighbors"
cat > "$V2/sklearn/__init__.py" <<'PY'
"""最小 sklearn 桩件：open3d.ml 只在 import 期需要 KDTree 符号存在。"""
from . import neighbors  # noqa: F401
PY
cat > "$V2/sklearn/neighbors.py" <<'PY'
class KDTree:
    def __init__(self, *a, **k):
        raise RuntimeError("sklearn 桩件：open3d.ml 的 KDTree 不可用（评测不使用）")
PY
PYTHONPATH="$V2:$WS/envs/pylibs_o3d:$WS/envs/pylibs:$WS/envs/pylibs_core" python3 - <<'PY' 2>&1 | tail -25
import numpy as np, traceback
try:
    import open3d as o3d
    print("open3d", o3d.__version__, "| numpy", np.__version__)
except Exception:
    traceback.print_exc(); raise SystemExit(1)
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
    print("  open3d 0.19 TSDF 合成平面 points =", n, "[OK]" if n else "[EMPTY]")
except Exception:
    traceback.print_exc()
PY

echo
echo "===== 下载 open3d 0.18.0 / numpy 1.26.4 wheel（--no-deps）====="
pip download --no-deps -d "$WS/tmp/wheels" "open3d==0.18.0" "numpy==1.26.4" 2>&1 | tail -5
ls -la "$WS/tmp/wheels" | head

echo
echo "===== 安装到 $V2（--no-deps）====="
pip install --no-deps --target "$V2" --upgrade \
  "$WS/tmp/wheels"/open3d-0.18.0-*.whl "$WS/tmp/wheels"/numpy-1.26.4-*.whl 2>&1 | tail -4
ls "$V2" | head -20

echo
echo "===== 测试 open3d 0.18.0 的 TSDF ====="
PYTHONPATH="$V2:$WS/envs/pylibs:$WS/envs/pylibs_core" python3 - <<'PY' 2>&1 | tail -30
import numpy as np, traceback
try:
    import open3d as o3d
    print("open3d", o3d.__version__, "| numpy", np.__version__)
except Exception:
    traceback.print_exc(); raise SystemExit(1)
print(open3d.pipelines.integration.ScalableTSDFVolume.__init__.__doc__[:600])
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
    print("  open3d 0.18 TSDF 合成平面 points =", n, "[OK]" if n else "[EMPTY]")
except Exception:
    traceback.print_exc()
PY
