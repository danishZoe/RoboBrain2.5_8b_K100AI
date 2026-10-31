#!/usr/bin/env bash
# 补齐 open3d 0.18 的 open3d.ml 依赖（只补 import 期需要的纯 Python 包），再测 TSDF
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
source "$WS/tools/env_net.sh"
export PYTHONWARNINGS=ignore
export PIP_CACHE_DIR="$WS/cache/pip"
export TMPDIR="$WS/tmp"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.sourcefind.cn/release/dtk2604/}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.sourcefind.cn}"
V2=$WS/envs/pylibs_o3d_v2

echo "===== 先看 envs/pylibs 里已有哪些 ====="
for m in joblib pandas pytz tzdata matplotlib sklearn scipy; do
  if [ -d "$WS/envs/pylibs/$m" ]; then echo "  $m: envs/pylibs 有"; else echo "  $m: 缺"; fi
done

echo
echo "===== 装缺失的（--no-deps，进 $V2）====="
timeout 600 pip install --no-deps --target "$V2" joblib pandas pytz tzdata 2>&1 | tail -4

echo
echo "===== TSDF 测试 ====="
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
