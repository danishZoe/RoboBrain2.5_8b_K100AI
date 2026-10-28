#!/usr/bin/env bash
# 扫描 open3d 0.20 ScalableTSDFVolume 的参数空间，找出"能抽出点"的组合
# 目的：确认 0.19+ 的体素语义（volume_unit_resolution / depth_sampling_stride），
#       以及"官方脚本那组参数"在 0.19/0.20 下为何失效。
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONPATH="$WS/envs/pylibs_o3d:$WS/envs/pylibs:$WS/envs/pylibs_core"
export PYTHONWARNINGS=ignore
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

python3 - <<'PY'
import numpy as np, itertools, traceback
import open3d as o3d
print("open3d", o3d.__version__)

H, W = 384, 512
intr = o3d.camera.PinholeCameraIntrinsic(W, H, 421.66, 421.66, 253.93, 192.12)
d_mm = np.full((H, W), 1000.0, np.float32)
col = np.zeros((H, W, 3), np.uint8)

def run(voxel_length, sdf_trunc, vur, dss):
    try:
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(col), o3d.geometry.Image(d_mm),
            depth_scale=1000.0, depth_trunc=5.0, convert_rgb_to_intensity=False)
        kw = dict(voxel_length=voxel_length, sdf_trunc=sdf_trunc,
                  color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
        if vur is not None:
            kw["volume_unit_resolution"] = vur
        if dss is not None:
            kw["depth_sampling_stride"] = dss
        v = o3d.pipelines.integration.ScalableTSDFVolume(**kw)
        v.integrate(rgbd, intr, np.linalg.inv(np.eye(4)))
        n = len(v.extract_point_cloud().points)
        return n
    except Exception as e:
        return "EXC:%s" % str(e)[:60]

print("\n=== 官方脚本那组参数 ===")
print("  voxel=0.02 sdf=0.08 (vur/dss 默认) ->", run(0.02, 0.08, None, None))

print("\n=== volume_unit_resolution 扫描（voxel=0.02 sdf=0.08 dss=默认）===")
for vur in (1, 2, 4, 8, 16, 32, 64):
    print("  vur=%-3d -> %s" % (vur, run(0.02, 0.08, vur, None)))

print("\n=== depth_sampling_stride 扫描（voxel=0.02 sdf=0.08 vur=默认）===")
for dss in (1, 2, 4, 8, 16):
    print("  dss=%-3d -> %s" % (dss, run(0.02, 0.08, None, dss)))

print("\n=== sdf_trunc 扫描（voxel=0.02 vur=16 dss=4）===")
for sdf in (0.005, 0.02, 0.08, 0.32, 1.28):
    print("  sdf=%-7.3f -> %s" % (sdf, run(0.02, sdf, 16, 4)))

print("\n=== voxel_length 扫描（sdf=4*voxel vur=16 dss=4）===")
for vox in (0.00125, 0.005, 0.02, 0.32):
    print("  voxel=%-8.5f -> %s" % (vox, run(vox, 4 * vox, 16, 4)))

print("\n=== 全默认扫（官方参数 + vur=1 dss=1）===")
print("  vur=1 dss=1 ->", run(0.02, 0.08, 1, 1))
PY
