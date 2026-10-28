#!/usr/bin/env bash
# 定点探测：open3d 修复后，官方 3D 判分单样本的真实成本（耗时 + 峰值 RSS）
#
# 用法: probe_3d_fixed.sh <o3d_prefix> [offset] [limit] [concurrency]
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONPATH="$WS/envs/pylibs_core:$WS/third_party/upstream_eval/FlagEvalMM:$WS/scripts:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore
export TMPDIR="$WS/tmp"

O3D="${1:-$WS/envs/pylibs_o3d_v2}"
OFFSET="${2:-50}"
LIMIT="${3:-5}"
CONC="${4:-5}"
export ROBO_O3D_PREFIX="$O3D"

set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

echo "===== open3d 前缀自检: $O3D ====="
PYTHONPATH="$O3D:$WS/envs/pylibs:$WS/envs/pylibs_core" python3 - <<'PY'
import numpy as np, traceback
import open3d as o3d
print("open3d", o3d.__version__, "| numpy", np.__version__, "|", open3d.__file__)
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
    print("  TSDF 合成平面 points =", n, "[OK]" if n else "[EMPTY —— 碰撞检查仍会失效]")
except Exception:
    traceback.print_exc()
PY

echo
echo "===== 3D 判分定点探测（offset=$OFFSET limit=$LIMIT 并发=$CONC）====="
python3 "$WS/submission/bench_public/score_3d_chunked.py" \
  --job "$WS/outputs/bench_public/tracespatial|official|3D" \
  --offset "$OFFSET" --limit "$LIMIT" \
  --chunk-size 1 --concurrency "$CONC" --mem-budget-gb 60 \
  --out "$WS/tmp/probe_3d_fixed.json" \
  2>&1 | grep -E 'chunk|任务|metrics|start_|end_|no_collision|overall|audit'

echo
echo "===== 逐样本耗时 / 峰值 RSS ====="
python3 - <<'PY'
import json
p = "/root/private_data/workspace/robo/tmp/probe_3d_fixed.json"
d = json.load(open(p, encoding="utf-8"))
cs = sorted(d["per_chunk"], key=lambda c: -c["wall_s"])
for c in cs:
    print("  样本#%-4d  %8.1fs  峰值RSS=%7.0fMB  skip=%d warn3d=%d warn3d_start=%d warn3d_end=%d" % (
        c["chunk_index"], c["wall_s"], c["peak_rss_mb"], c["counts"]["skip"],
        c["counts"]["warn3d_collision"], c["counts"]["warn3d_start"],
        c["counts"]["warn3d_end"]))
print("  合计墙钟 %.1fs，最大峰值 RSS %.0fMB" % (
    d.get("wall_s_total", 0), d.get("peak_rss_mb_max", 0)))
PY
