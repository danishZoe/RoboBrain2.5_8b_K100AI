#!/usr/bin/env bash
# A) 从缓存里找已下载的 open3d wheel；B) 直接用 PyPI JSON API 取 0.18.0 的 wheel URL 并下载
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
source "$WS/tools/env_net.sh"
export PYTHONWARNINGS=ignore
export TMPDIR="$WS/tmp"
V2=$WS/envs/pylibs_o3d_v2
mkdir -p "$WS/tmp/wheels"

echo "===== A. pip 缓存里的 open3d wheel ====="
python3 - <<'PY'
import os, zipfile
CACHE = "/root/private_data/workspace/robo/cache/pip"
hits = []
for root, _, files in os.walk(CACHE):
    for f in files:
        if not f.endswith(".body"):
            continue
        p = os.path.join(root, f)
        try:
            sz = os.path.getsize(p)
        except OSError:
            continue
        if sz < 50_000_000:
            continue
        try:
            with zipfile.ZipFile(p) as z:
                names = z.namelist()
                di = [n for n in names if n.endswith(".dist-info/METADATA")]
                if any("open3d" in n for n in names[:50]):
                    hits.append((p, sz, di[:1]))
        except Exception:
            pass
for p, sz, di in hits:
    print("  %s  %.0fMB  %s" % (p, sz/2**20, di))
if not hits:
    print("  (缓存里没找到 open3d wheel)")
PY

echo
echo "===== B. PyPI JSON API ====="
for SPEC in "open3d 0.18.0" "numpy 1.26.4"; do
  set -- $SPEC
  PKG=$1; VER=$2
  echo "--- $PKG $VER ---"
  URL=$(curl -s --max-time 60 "https://pypi.org/pypi/${PKG}/${VER}/json" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
except Exception as e:
    print(''); raise SystemExit
for f in d.get('urls',[]):
    fn=f.get('filename','')
    if 'cp310-cp310-manylinux' in fn and 'x86_64' in fn and fn.endswith('.whl'):
        print(f.get('url','')); break
")
  echo "  URL=$URL"
  if [ -n "$URL" ]; then
    curl -sL --max-time 1200 -o "$WS/tmp/wheels/${PKG}-${VER}-cp310.whl" "$URL"
    ls -la "$WS/tmp/wheels/${PKG}-${VER}-cp310.whl" 2>/dev/null
  fi
done

echo
echo "===== C. 装进 $V2 并测 TSDF ====="
ls -la "$WS/tmp/wheels/"
rm -rf "$V2" 2>/dev/null || true
mkdir -p "$V2"
if ls "$WS/tmp/wheels"/*.whl >/dev/null 2>&1; then
  pip install --no-deps --target "$V2" "$WS/tmp/wheels"/*.whl 2>&1 | tail -4
fi
ls "$V2" | head -15
[ -f "$V2/open3d/__init__.py" ] && echo "open3d 包体已就位" || echo "open3d 包体缺失"

echo
echo "===== D. TSDF 测试（官方脚本那组参数）====="
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
