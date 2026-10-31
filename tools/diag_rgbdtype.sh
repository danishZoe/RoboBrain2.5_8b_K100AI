#!/usr/bin/env bash
# 实验：open3d 0.20 下 RGBDImage 构造矩阵 —— 哪种能被 ScalableTSDFVolume 接受？
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONPATH="$WS/envs/pylibs_o3d:$WS/envs/pylibs:$WS/envs/pylibs_core"
export PYTHONWARNINGS=ignore
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

python3 - <<'PY'
import os
import numpy as np
import open3d as o3d

H, W = 384, 512
fx = fy = 421.66
cx, cy = 253.93, 192.12
intr = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

u8_3 = np.zeros((H, W, 3), np.uint8)
u8_1 = np.zeros((H, W), np.uint8)
u16 = np.full((H, W), 1000, np.uint16)
f32 = np.full((H, W), 1000.0, np.float32)

def attempt(tag, color_arr, depth_arr, scale, trunc, convert_intensity=False,
            color_type=None, voxel=0.02):
    try:
        ci = o3d.geometry.Image(color_arr)
        di = o3d.geometry.Image(depth_arr)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            ci, di, depth_scale=scale, depth_trunc=trunc,
            convert_rgb_to_intensity=convert_intensity)
        dep = np.asarray(rgbd.depth)
        ct = color_type or o3d.pipelines.integration.TSDFVolumeColorType.NoColor
        v = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel, sdf_trunc=4*voxel, color_type=ct)
        v.integrate(rgbd, intr, np.linalg.inv(np.eye(4)))
        n = len(v.extract_point_cloud().points)
        print("  %-52s depth[dtype=%s max=%.3f] -> points=%d %s" % (
            tag, dep.dtype, float(dep.max()), n, "[OK]" if n > 0 else "[EMPTY]"))
        return n
    except Exception as e:
        print("  %-52s EXC %s: %s" % (tag, type(e).__name__, str(e)[:120]))
        return -1

print("=== 构造矩阵 ===")
attempt("A color=u8(3) depth=f32(mm)  scale=1000  [官方脚本]", u8_3, f32, 1000.0, 5.0)
attempt("B color=u8(3) depth=u16(mm)  scale=1000", u8_3, u16, 1000.0, 5.0)
attempt("C color=u8(1) depth=u16(mm)  scale=1000", u8_1, u16, 1000.0, 5.0)
attempt("D color=u8(3) depth=f32(m)   scale=1", u8_3, f32.astype(np.float32)/1000.0, 1.0, 5.0)
attempt("E color=u8(3) depth=u16(mm)  +intensity RGB8", u8_3, u16, 1000.0, 5.0, True,
        o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
attempt("F color=u8(3) depth=u16(mm)  +intensity NoColor", u8_3, u16, 1000.0, 5.0, True,
        o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
attempt("G color=u8(3) depth=f32(mm)  scale=1000 voxel=0.05", u8_3, f32, 1000.0, 5.0, voxel=0.05)

print("\n=== 用 o3d.io.read_image 读真实 PNG（上游 Open3D 官方示例做法）===")
BENCH = "/root/private_data/workspace/robo/third_party/benchmarks/TraceSpatial-Bench"
d = o3d.io.read_image(os.path.join(BENCH, "raw_data/0/gt_depth.png"))
c = o3d.io.read_image(os.path.join(BENCH, "raw_data/0/image.png"))
rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    c, d, depth_scale=1000.0, depth_trunc=5.0, convert_rgb_to_intensity=False)
dep = np.asarray(rgbd.depth)
print("  rgbd.depth dtype=%s max=%.3f nonzero%%=%.2f" % (
    dep.dtype, float(dep.max()), (dep > 0).mean()*100))
v = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.02, sdf_trunc=0.08,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
v.integrate(rgbd, intr, np.linalg.inv(np.eye(4)))
print("  >>> read_image 路径 points:", len(v.extract_point_cloud().points))

print("\n=== 真实数据 + float32(官方脚本构造) 对照 ===")
from PIL import Image
du = np.array(Image.open(os.path.join(BENCH, "raw_data/0/gt_depth.png")))
mk = np.array(Image.open(os.path.join(BENCH, "raw_data/0/mask.png")))
if mk.ndim == 3:
    mk = mk[:, :, 0]
do = du.astype(np.float32).copy()
do[mk == 255] = 0
ci = o3d.geometry.Image(np.zeros((H, W, 3), np.uint8))
di = o3d.geometry.Image(do)
rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    ci, di, depth_scale=1000.0, depth_trunc=5.0, convert_rgb_to_intensity=False)
dep = np.asarray(rgbd.depth)
print("  rgbd.depth dtype=%s max=%.3f nonzero%%=%.2f" % (
    dep.dtype, float(dep.max()), (dep > 0).mean()*100))
v = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.02, sdf_trunc=0.08,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
v.integrate(rgbd, intr, np.linalg.inv(np.eye(4)))
print("  >>> 官方脚本构造 points:", len(v.extract_point_cloud().points))
PY
