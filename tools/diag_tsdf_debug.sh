#!/usr/bin/env bash
# 实验：开 debug 日志定位 open3d TSDF 失效点；并尝试 UniformTSDFVolume / 米制深度等变体
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONPATH="$WS/envs/pylibs_o3d:$WS/envs/pylibs:$WS/envs/pylibs_core"
export PYTHONWARNINGS=ignore
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

python3 - <<'PY'
import numpy as np, traceback
import open3d as o3d

H, W = 384, 512
fx = fy = 421.66
cx, cy = 253.93, 192.12
intr = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

def build_rgbd(depth_m=1.0, scale=1000.0, trunc=5.0):
    d = np.full((H, W), depth_m*scale, dtype=np.float32)
    dummy = o3d.geometry.Image(np.zeros((H, W, 3), dtype=np.uint8))
    o3d_depth = o3d.geometry.Image(d)
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        dummy, o3d_depth, depth_scale=scale, depth_trunc=trunc,
        convert_rgb_to_intensity=False)

print("########## 变体 1：深度已是米制，depth_scale=1 ##########")
try:
    rgbd = build_rgbd(1.0, scale=1.0, trunc=5.0)
    v = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.02, sdf_trunc=0.08,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
    v.integrate(rgbd, intr, np.linalg.inv(np.eye(4)))
    print("  points:", len(v.extract_point_cloud().points))
except Exception:
    traceback.print_exc()

print("\n########## 变体 2：UniformTSDFVolume（legacy 均匀网格）##########")
try:
    rgbd = build_rgbd(1.0)
    v = o3d.pipelines.integration.UniformTSDFVolume(
        voxel_length=0.02, sdf_trunc=0.08,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
    v.integrate(rgbd, intr, np.linalg.inv(np.eye(4)))
    print("  points:", len(v.extract_point_cloud().points))
except Exception:
    traceback.print_exc()

print("\n########## 变体 3：open3d debug 日志（看 integrate 内部）##########")
try:
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    rgbd = build_rgbd(1.0)
    v = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.02, sdf_trunc=0.08,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
    v.integrate(rgbd, intr, np.linalg.inv(np.eye(4)))
    pc = v.extract_point_cloud()
    print("  points:", len(pc.points))
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
except Exception:
    traceback.print_exc()

print("\n########## 变体 4：非全零彩色图 + convert_rgb_to_intensity=True ##########")
try:
    d = np.full((H, W), 1000.0, dtype=np.float32)
    color = np.zeros((H, W, 3), dtype=np.uint8); color[:, :, 0] = 128
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color), o3d.geometry.Image(d),
        depth_scale=1000.0, depth_trunc=5.0, convert_rgb_to_intensity=True)
    v = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.02, sdf_trunc=0.08,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    v.integrate(rgbd, intr, np.linalg.inv(np.eye(4)))
    print("  points:", len(v.extract_point_cloud().points))
    print("  mesh verts:", len(v.extract_triangle_mesh().vertices))
except Exception:
    traceback.print_exc()

print("\n########## 变体 5：检查 numpy 2 与 open3d 的数值互操作 ##########")
try:
    a = np.zeros((4,4), dtype=np.float32); a[0,0]=1
    print("  np.matmul:", (a @ a)[0,0])
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.zeros((5,3), dtype=np.float64))
    print("  Vector3dVector ok:", len(p.points))
    p2 = o3d.geometry.PointCloud()
    p2.points = o3d.utility.Vector3dVector(np.zeros((5,3), dtype=np.float32))
    print("  Vector3dVector float32 ok:", len(p2.points))
except Exception:
    traceback.print_exc()
PY
