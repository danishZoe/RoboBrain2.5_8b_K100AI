#!/usr/bin/env bash
# 修复 open3d 环境缺陷：找出一个 ScalableTSDFVolume 能正常工作的版本，装到隔离前缀
#
# 背景：envs/pylibs_o3d 里的 open3d 0.20.0，其 ScalableTSDFVolume::Integrate
# 在本环境**抽不出任何点**（合成平面、上游 read_image 路径均为 0 点），
# 导致官方 summarize_acc.py 的 3D 碰撞检查必然抛 ValueError，
# 碰撞标志恒为 False —— No-Collision 恒 1.0、Overall Success 虚高。
#
# 本脚本只往工作目录内的隔离前缀装包，不动公共 site-packages。
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
source "$WS/tools/env_net.sh"
export PYTHONWARNINGS=ignore
export PIP_CACHE_DIR="$WS/cache/pip"
export TMPDIR="$WS/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

V2=$WS/envs/pylibs_o3d_v2
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.sourcefind.cn/release/dtk2604/}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.sourcefind.cn}"

echo "===== 可用 open3d 版本 ====="
pip index versions open3d 2>&1 | head -5 || pip install "open3d==" 2>&1 | grep -o 'from versions:.*' | head -2

echo
echo "===== 装 open3d==0.19.0 到 $V2（--no-deps，不动 numpy）====="
rm -rf "$V2" 2>/dev/null || true
mkdir -p "$V2"
pip install --target "$V2" --no-deps "open3d==0.19.0" 2>&1 | tail -6
ls "$V2" | head -20

echo
echo "===== 测试：open3d 0.19.0 的 ScalableTSDFVolume ====="
PYTHONPATH="$V2:$WS/envs/pylibs_o3d:$WS/envs/pylibs:$WS/envs/pylibs_core" python3 - <<'PY'
import numpy as np, traceback
import open3d as o3d
print("open3d", o3d.__version__, "| numpy", np.__version__)
H, W = 384, 512
intr = o3d.camera.PinholeCameraIntrinsic(W, H, 421.66, 421.66, 253.93, 192.12)
def test(tag, depth_arr, scale, color_arr=None, color_type=None):
    try:
        if color_arr is None:
            color_arr = np.zeros((H, W, 3), np.uint8)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_arr), o3d.geometry.Image(depth_arr),
            depth_scale=scale, depth_trunc=5.0, convert_rgb_to_intensity=False)
        ct = color_type or o3d.pipelines.integration.TSDFVolumeColorType.NoColor
        v = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.02, sdf_trunc=0.08, color_type=ct)
        v.integrate(rgbd, intr, np.linalg.inv(np.eye(4)))
        n = len(v.extract_point_cloud().points)
        print("  %-46s -> points=%d %s" % (tag, n, "[OK]" if n > 0 else "[EMPTY]"))
        return n
    except Exception as e:
        print("  %-46s EXC %s: %s" % (tag, type(e).__name__, str(e)[:160]))
        return -1

test("合成平面 1m  float32(mm) [官方脚本构造]",
     np.full((H, W), 1000.0, np.float32), 1000.0)
test("合成平面 1m  uint16(mm)", np.full((H, W), 1000, np.uint16), 1000.0)
PY

echo
echo "===== 若上面全 EMPTY，再试 0.18.0 + numpy1.26 ====="
if ! PYTHONPATH="$V2:$WS/envs/pylibs_o3d:$WS/envs/pylibs:$WS/envs/pylibs_core" python3 - <<'PY' 2>/dev/null | grep -q 'OK'
import numpy as np, open3d as o3d
H, W = 384, 512
intr = o3d.camera.PinholeCameraIntrinsic(W, H, 421.66, 421.66, 253.93, 192.12)
rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.geometry.Image(np.zeros((H, W, 3), np.uint8)),
    o3d.geometry.Image(np.full((H, W), 1000.0, np.float32)),
    depth_scale=1000.0, depth_trunc=5.0, convert_rgb_to_intensity=False)
v = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.02, sdf_trunc=0.08,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
v.integrate(rgbd, intr, np.linalg.inv(np.eye(4)))
print("OK" if len(v.extract_point_cloud().points) > 0 else "EMPTY")
PY
then
  echo "  0.19.0 不可用 -> 试 0.18.0（自带 numpy 1.x）"
  rm -rf "$V2" 2>/dev/null || true
  mkdir -p "$V2"
  pip install --target "$V2" "open3d==0.18.0" 2>&1 | tail -8
  PYTHONPATH="$V2:$WS/envs/pylibs:$WS/envs/pylibs_core" python3 - <<'PY'
import numpy as np, traceback
import open3d as o3d
print("open3d", o3d.__version__, "| numpy", np.__version__)
H, W = 384, 512
intr = o3d.camera.PinholeCameraIntrinsic(W, H, 421.66, 421.66, 253.93, 192.12)
rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.geometry.Image(np.zeros((H, W, 3), np.uint8)),
    o3d.geometry.Image(np.full((H, W), 1000.0, np.float32)),
    depth_scale=1000.0, depth_trunc=5.0, convert_rgb_to_intensity=False)
v = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.02, sdf_trunc=0.08,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
v.integrate(rgbd, intr, np.linalg.inv(np.eye(4)))
n = len(v.extract_point_cloud().points)
print("  合成平面 points =", n, "[OK]" if n > 0 else "[EMPTY]")
PY
fi
