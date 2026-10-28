#!/usr/bin/env bash
# 验证 open3d 0.18.0 在官方脚本那组参数下 TSDF 能否抽出点
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONWARNINGS=ignore
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u
V2=$WS/envs/pylibs_o3d_v2

PYTHONPATH="$V2:$WS/envs/pylibs:$WS/envs/pylibs_core" python3 - <<'PY' 2>&1 | tail -30
import numpy as np, traceback
import open3d as o3d
print("open3d", o3d.__version__, "| numpy", np.__version__)
print("路径:", o3d.__file__)
print("签名:", o3d.pipelines.integration.ScalableTSDFVolume.__init__.__doc__[:300])

H, W = 384, 512
intr = o3d.camera.PinholeCameraIntrinsic(W, H, 421.66, 421.66, 253.93, 192.12)


def t(tag, depth_arr, scale, trunc=5.0):
    try:
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.zeros((H, W, 3), np.uint8)),
            o3d.geometry.Image(depth_arr),
            depth_scale=scale, depth_trunc=trunc, convert_rgb_to_intensity=False)
        v = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.02, sdf_trunc=0.08,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
        v.integrate(rgbd, intr, np.linalg.inv(np.eye(4)))
        n = len(v.extract_point_cloud().points)
        print("  %-44s -> points=%d %s" % (tag, n, "[OK]" if n else "[EMPTY]"))
        return n
    except Exception as e:
        print("  %-44s EXC %s: %s" % (tag, type(e).__name__, str(e)[:140]))
        return -1


print("\n=== 官方脚本那组参数 ===")
t("合成平面 1m  float32(mm)  [官方构造]", np.full((H, W), 1000.0, np.float32), 1000.0)
t("合成平面 1m  uint16(mm)", np.full((H, W), 1000, np.uint16), 1000.0)
t("合成平面 1m  float32(m) scale=1", np.full((H, W), 1.0, np.float32), 1.0)

print("\n=== 真实 TraceSpatial 样本 0（官方构造路径）===")
import os
from PIL import Image
BENCH = "/root/private_data/workspace/robo/third_party/benchmarks/TraceSpatial-Bench"
d = np.array(Image.open(os.path.join(BENCH, "raw_data/0/gt_depth.png"))).astype(np.float32)
mk = np.array(Image.open(os.path.join(BENCH, "raw_data/0/mask.png")))
if mk.ndim == 3:
    mk = mk[:, :, 0]
d[mk == 255] = 0
try:
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.zeros((d.shape[0], d.shape[1], 3), np.uint8)),
        o3d.geometry.Image(d),
        depth_scale=1000.0, depth_trunc=5.0, convert_rgb_to_intensity=False)
    intr2 = o3d.camera.PinholeCameraIntrinsic(d.shape[1], d.shape[0],
                                               421.6617431640625, 421.6598815917969,
                                               253.9289093017578, 192.12013244628906)
    v = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.02, sdf_trunc=0.08,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
    v.integrate(rgbd, intr2, np.linalg.inv(np.eye(4)))
    pcd = v.extract_point_cloud()
    print("  真实样本 TSDF points =", len(pcd.points))
    if len(pcd.points):
        grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.02)
        print("  VoxelGrid voxels =", len(grid.get_voxels()))
except Exception:
    traceback.print_exc()
PY
