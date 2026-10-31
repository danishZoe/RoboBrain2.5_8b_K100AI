#!/usr/bin/env bash
# 实验：为什么官方 create_occupancy_grid_from_tsdf 抽不出点？
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONPATH="$WS/envs/pylibs_o3d:$WS/envs/pylibs:$WS/envs/pylibs_core"
export PYTHONWARNINGS=ignore
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

python3 - <<'PY'
import json, os, sys, traceback
import numpy as np
from PIL import Image
import open3d as o3d

print("open3d", o3d.__version__)

BENCH = "/root/private_data/workspace/robo/third_party/benchmarks/TraceSpatial-Bench"
OJ = "/root/private_data/workspace/robo/outputs/bench_public/tracespatial/official/outputs/RoboTracer_RoboBrain2.5-8B-NV_3D.jsonl"
ans = [json.loads(l) for l in open(OJ, encoding="utf-8") if l.strip()]
a = ans[0]
K = np.array(a["gt_depth_intrinsics"])
print("intrinsics shape", K.shape)
print(K)
gp = os.path.join(BENCH, "raw_data", a["gt_depth_path"])
mp = os.path.join(BENCH, "raw_data", a["mask_path"])
depth_u16 = np.array(Image.open(gp))
mask = np.array(Image.open(mp))
if mask.ndim == 3:
    mask = mask[:, :, 0]
print("depth", depth_u16.shape, depth_u16.dtype, "max", depth_u16.max(),
      "nonzero%", round((depth_u16 > 0).mean()*100, 2))
print("mask", mask.shape, "nonzero%", round((mask > 0).mean()*100, 2))

H, W = depth_u16.shape
fx, fy = K[0,0], K[1,1]
cx, cy = K[0,2], K[1,2]
print("fx,fy,cx,cy =", fx, fy, cx, cy, " image WxH =", W, H)
print("cx/W =", cx/W, " cy/H =", cy/H)

def try_variant(tag, depth_arr, scale, trunc, voxel=0.02, remove_mask=True):
    print("\n=== %s ===" % tag)
    try:
        d = np.copy(depth_arr)
        if remove_mask:
            d[mask == 255] = 0
        o3d_depth = o3d.geometry.Image(d)
        print("  Image dtype:", getattr(o3d_depth, "dtype", "?"))
        dummy = o3d.geometry.Image(np.zeros((H, W, 3), dtype=np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            dummy, o3d_depth, depth_scale=scale, depth_trunc=trunc,
            convert_rgb_to_intensity=False)
        # 看看 Open3D 内部把深度换算成了什么
        dd = np.asarray(rgbd.depth)
        print("  rgbd depth: dtype=%s min=%.4f max=%.4f nonzero%%=%.2f" % (
            dd.dtype, dd.min(), dd.max(), (dd > 0).mean()*100))
        intr = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
        vol = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel, sdf_trunc=4*voxel,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
        vol.integrate(rgbd, intr, np.linalg.inv(np.eye(4)))
        pcd = vol.extract_point_cloud()
        print("  extract_point_cloud: %s points" % len(pcd.points))
        if len(pcd.points) > 0:
            grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel)
            print("  VoxelGrid voxels:", len(grid.get_voxels()))
            return True
    except Exception as e:
        print("  FAILED:", type(e).__name__, e)
    return False

# A: 官方原样 —— float32 深度
try_variant("A 官方原样 float32 + 挖掉物体", depth_u16.astype(np.float32), 1000.0, 5.0)
# B: uint16 深度（不是 depth image）
try_variant("B uint16 + 挖掉物体", depth_u16, 1000.0, 5.0)
# C: float32，不挖物体
try_variant("C float32 不挖物体", depth_u16.astype(np.float32), 1000.0, 5.0)
# D: float32 + 不截断
try_variant("D float32 + 挖物体 + depth_trunc=100", depth_u16.astype(np.float32), 1000.0, 100.0)
PY
