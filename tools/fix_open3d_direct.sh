#!/usr/bin/env bash
# 1) 检查 envs/pylibs_o3d_v2 的实际布局；2) 直连镜像下载 open3d 0.18.0 + numpy 1.26.4；3) 测试 TSDF
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
source "$WS/tools/env_net.sh"
export PYTHONWARNINGS=ignore
export PIP_CACHE_DIR="$WS/cache/pip"
export TMPDIR="$WS/tmp"
V2=$WS/envs/pylibs_o3d_v2
mkdir -p "$WS/tmp/wheels"

echo "===== 1. $V2 布局 ====="
ls -la "$V2" | head -12
echo "--- $V2/open3d 前 10 项 ---"
ls -la "$V2/open3d" 2>/dev/null | head -10
echo "--- __init__.py 是否存在 ---"
[ -f "$V2/open3d/__init__.py" ] && echo "有" || echo "无（说明 pip 只装了 dist-info/元数据，包体缺失）"

echo
echo "===== 2. 直连镜像索引，取 open3d 0.18.0 / numpy 1.26.4 的 wheel URL ====="
for PKG in open3d numpy; do
  echo "--- $PKG ---"
  curl -s --max-time 60 "https://pypi.sourcefind.cn/release/dtk2604/${PKG}/" \
    | grep -oE 'href="[^"]+\.whl[^"]*"' | sed 's/href="//; s/"$//' | head -20
done

echo
echo "===== 3. 下载并安装 ====="
O3D_URL=$(curl -s --max-time 60 "https://pypi.sourcefind.cn/release/dtk2604/open3d/" \
  | grep -oE 'href="[^"]*open3d-0\.18\.0-cp310-cp310-manylinux[^"]*x86_64\.whl[^"]*"' \
  | sed 's/href="//; s/"$//' | head -1)
NP_URL=$(curl -s --max-time 60 "https://pypi.sourcefind.cn/release/dtk2604/numpy/" \
  | grep -oE 'href="[^"]*numpy-1\.26\.4-cp310-cp310-manylinux[^"]*x86_64\.whl[^"]*"' \
  | sed 's/href="//; s/"$//' | head -1)
echo "O3D_URL=$O3D_URL"
echo "NP_URL=$NP_URL"

if [ -n "$O3D_URL" ]; then
  case "$O3D_URL" in http*) FULL="$O3D_URL" ;; *) FULL="https://pypi.sourcefind.cn/release/dtk2604/open3d/${O3D_URL}" ;; esac
  echo "下载 $FULL"
  curl -sL --max-time 900 -o "$WS/tmp/wheels/open3d-0.18.0.whl" "$FULL"
  ls -la "$WS/tmp/wheels/"
fi
if [ -n "$NP_URL" ]; then
  case "$NP_URL" in http*) FULL="$NP_URL" ;; *) FULL="https://pypi.sourcefind.cn/release/dtk2604/numpy/${NP_URL}" ;; esac
  echo "下载 $FULL"
  curl -sL --max-time 900 -o "$WS/tmp/wheels/numpy-1.26.4.whl" "$FULL"
  ls -la "$WS/tmp/wheels/"
fi

echo
echo "===== 4. 安装到 $V2（--no-deps）====="
rm -rf "$V2/open3d" "$V2/open3d-0.19.0.dist-info" 2>/dev/null || true
pip install --no-deps --target "$V2" --upgrade "$WS/tmp/wheels"/*.whl 2>&1 | tail -4
ls -la "$V2/open3d/__init__.py" 2>/dev/null && echo "包体已就位" || echo "包体缺失"

echo
echo "===== 5. 测试 open3d 0.18.0 的 TSDF ====="
PYTHONPATH="$V2:$WS/envs/pylibs:$WS/envs/pylibs_core" python3 - <<'PY' 2>&1 | tail -30
import numpy as np, traceback
try:
    import open3d as o3d
    print("open3d", o3d.__version__, "| numpy", np.__version__, "|", open3d.__file__)
except Exception:
    traceback.print_exc(); raise SystemExit(1)
try:
    print(o3d.pipelines.integration.ScalableTSDFVolume.__init__.__doc__[:500])
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
