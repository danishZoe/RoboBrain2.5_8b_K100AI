#!/usr/bin/env bash
# 实验：open3d TSDF 在本机是否根本不可用（合成平面对照）
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
print("open3d", o3d.__version__, "| numpy", np.__version__)

H, W = 384, 512
fx = fy = 421.66
cx, cy = 253.93, 192.12

def synth_plane(depth_m=1.0, dtype=np.float32):
    """合成：正前方 1m 处的平面（毫米单位）"""
    d = np.full((H, W), depth_m * 1000.0, dtype=dtype)
    return d

def synth_random(seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.5, 3.0, size=(H, W)) * 1000.0

def run(tag, depth, voxel=0.02, sdf_mult=4, trunc=5.0, scale=1000.0, legacy=False):
    print("\n=== %s (voxel=%.3f sdf_trunc=%.3f) ===" % (tag, voxel, sdf_mult*voxel))
    try:
        d = depth
        dummy = o3d.geometry.Image(np.zeros((H, W, 3), dtype=np.uint8))
        o3d_depth = o3d.geometry.Image(np.ascontiguousarray(d))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            dummy, o3d_depth, depth_scale=scale, depth_trunc=trunc,
            convert_rgb_to_intensity=False)
        intr = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
        if legacy:
            vol = o3d.pipelines.integration.TSDFVolume(
                voxel_length=voxel, sdf_trunc=sdf_mult*voxel,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
        else:
            vol = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=voxel, sdf_trunc=sdf_mult*voxel,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
        vol.integrate(rgbd, intr, np.linalg.inv(np.eye(4)))
        pcd = vol.extract_point_cloud()
        print("  Scalable/legacy extract_point_cloud: %d points" % len(pcd.points))
        if len(pcd.points) == 0:
            try:
                mesh = vol.extract_triangle_mesh()
                print("  extract_triangle_mesh: %d verts" % len(mesh.vertices))
            except Exception as e:
                print("  mesh failed:", e)
        return len(pcd.points)
    except Exception as e:
        print("  FAILED:", type(e).__name__, e)
        traceback.print_exc()
        return -1

print("\n########## 合成数据 ##########")
run("合成平面 1m float32", synth_plane(1.0, np.float32))
run("合成平面 1m float64", synth_plane(1.0, np.float64))
run("合成平面 1m uint16", synth_plane(1.0, np.uint16))
run("合成随机深度 float32", synth_random())
run("合成平面 legacy TSDFVolume", synth_plane(1.0, np.float32), legacy=True)
run("合成平面 voxel=0.05", synth_plane(1.0, np.float32), voxel=0.05)
run("合成平面 sdf_mult=8", synth_plane(1.0, np.float32), sdf_mult=8)
run("合成平面 trunc=100", synth_plane(1.0, np.float32), trunc=100.0)

print("\n########## open3d 能力探测 ##########")
for attr in ("ScalableTSDFVolume", "TSDFVolume", "ScalableTSDFVolumeColorType"):
    print(" ", attr, hasattr(o3d.pipelines.integration, attr))
print("  integration 成员:", [x for x in dir(o3d.pipelines.integration) if not x.startswith("_")][:30])
PY
