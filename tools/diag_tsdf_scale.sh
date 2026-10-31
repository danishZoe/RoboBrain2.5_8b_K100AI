#!/usr/bin/env bash
# 验证假设：open3d 0.19+ 的 ScalableTSDFVolume 默认 scale=1.0 导致 SDF 带落不进体素
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONPATH="$WS/envs/pylibs_o3d:$WS/envs/pylibs:$WS/envs/pylibs_core"
export PYTHONWARNINGS=ignore
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

python3 - <<'PY'
import inspect
import numpy as np
import open3d as o3d
print("open3d", o3d.__version__)
print("\n=== ScalableTSDFVolume 构造签名 ===")
print(o3d.pipelines.integration.ScalableTSDFVolume.__init__.__doc__)
print("=== UniformTSDFVolume 构造签名 ===")
print(o3d.pipelines.integration.UniformTSDFVolume.__init__.__doc__)

H, W = 384, 512
intr = o3d.camera.PinholeCameraIntrinsic(W, H, 421.66, 421.66, 253.93, 192.12)
d_mm = np.full((H, W), 1000.0, np.float32)   # 1 m 平面
col = np.zeros((H, W, 3), np.uint8)

def test(tag, **kw):
    try:
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(col), o3d.geometry.Image(d_mm),
            depth_scale=1000.0, depth_trunc=5.0, convert_rgb_to_intensity=False)
        v = o3d.pipelines.integration.ScalableTSDFVolume(
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor, **kw)
        v.integrate(rgbd, intr, np.linalg.inv(np.eye(4)))
        n = len(v.extract_point_cloud().points)
        print("  %-58s -> points=%d %s" % (tag, n, "[OK]" if n else "[EMPTY]"))
        return n
    except Exception as e:
        print("  %-58s EXC %s: %s" % (tag, type(e).__name__, str(e)[:150]))
        return -1

print("\n=== scale 参数扫描（voxel_length=0.02, sdf_trunc=0.08）===")
test("scale 默认(不传)", voxel_length=0.02, sdf_trunc=0.08)
test("scale=1.0", voxel_length=0.02, sdf_trunc=0.08, scale=1.0)
test("scale=0.02  (= voxel_length)", voxel_length=0.02, sdf_trunc=0.08, scale=0.02)
test("scale=0.01", voxel_length=0.02, sdf_trunc=0.08, scale=0.01)
test("scale=0.05", voxel_length=0.02, sdf_trunc=0.08, scale=0.05)

print("\n=== 对照：uniform 网格（legacy UniformTSDFVolume）===")
try:
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(col), o3d.geometry.Image(d_mm),
        depth_scale=1000.0, depth_trunc=5.0, convert_rgb_to_intensity=False)
    v = o3d.pipelines.integration.UniformTSDFVolume(
        0.02, 100000, 0.08, o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
    v.integrate(rgbd, intr, np.linalg.inv(np.eye(4)))
    n = len(v.extract_point_cloud().points)
    print("  UniformTSDFVolume points =", n, "[OK]" if n else "[EMPTY]")
except Exception as e:
    print("  EXC", type(e).__name__, str(e)[:200])
PY
