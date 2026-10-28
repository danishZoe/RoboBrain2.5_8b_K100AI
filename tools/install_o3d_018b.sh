#!/usr/bin/env bash
# 直接把 wheel 当 zip 解到隔离前缀（绕开 pip 的文件名校验），并测 TSDF
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONWARNINGS=ignore
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u
V2=$WS/envs/pylibs_o3d_v2
W=$WS/tmp/wheels

echo "===== 解包 wheel 到 $V2 ====="
rm -rf "$V2" 2>/dev/null || true
mkdir -p "$V2"
python3 - <<'PY'
import glob, os, zipfile
V2 = "/root/private_data/workspace/robo/envs/pylibs_o3d_v2"
for w in sorted(glob.glob("/root/private_data/workspace/robo/tmp/wheels/*.whl")):
    with zipfile.ZipFile(w) as z:
        z.extractall(V2)
    print("  解包", os.path.basename(w))
PY
ls "$V2" | head -15
[ -f "$V2/open3d/__init__.py" ] && echo "open3d 包体已就位" || echo "open3d 包体缺失"

echo
echo "===== TSDF 测试（官方脚本那组参数 voxel=0.02 sdf_trunc=0.08）====="
PYTHONPATH="$V2:$WS/envs/pylibs:$WS/envs/pylibs_core" python3 - <<'PY' 2>&1 | tail -25
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
