#!/usr/bin/env bash
# 安装 open3d 0.18.0 到隔离前缀并测试 TSDF（不经过任何公共 site-packages）
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

echo "===== 1. pip install --target open3d==0.18.0（--no-deps）====="
rm -rf "$V2" 2>/dev/null || true
mkdir -p "$V2"
timeout 900 pip install --no-deps --target "$V2" "open3d==0.18.0" 2>&1 | tail -5
echo "--- $V2 内容 ---"
ls "$V2" | head -12
echo "--- $V2/open3d 内容 ---"
ls "$V2/open3d" 2>/dev/null | head -12
[ -f "$V2/open3d/__init__.py" ] && echo "__init__.py 存在" || echo "__init__.py 缺失 !!"

echo
echo "===== 2. numpy 1.26.4（open3d 0.18 按 numpy1 ABI 编译）====="
timeout 900 pip install --no-deps --target "$V2" "numpy==1.26.4" 2>&1 | tail -3
ls "$V2" | grep -i numpy

echo
echo "===== 3. TSDF 测试（PYTHONPATH 不含 envs/pylibs_o3d，避免 0.20 抢优先级）====="
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
    print("  TSDF 合成平面 points =", n, "[OK]" if n else "[EMPTY]")
except Exception:
    traceback.print_exc()
PY
