#!/usr/bin/env bash
# 诊断：官方 3D 判分里 open3d 相关环节为何全部失败
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONWARNINGS=ignore
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

O3D=$WS/envs/pylibs_o3d
BASE=$WS/envs/pylibs
CORE=$WS/envs/pylibs_core

echo "===== 1. 三个前缀里各有什么 ====="
for p in "$O3D" "$BASE" "$CORE"; do
  echo "--- $p ---"
  ls "$p" 2>/dev/null | head -30
done

echo
echo "===== 2. 只加 pylibs_o3d 时 open3d 能否 import ====="
PYTHONPATH="$O3D:$BASE:$CORE" python3 - <<'PY' 2>&1 | tail -40
import traceback
try:
    import numpy
    print("numpy", numpy.__version__, numpy.__file__)
except Exception:
    traceback.print_exc()
try:
    import open3d
    print("open3d OK", open3d.__version__, open3d.__file__)
except Exception:
    print("open3d IMPORT FAILED:")
    traceback.print_exc()
PY

echo
echo "===== 3. 评分用 PYTHONPATH（o3d -> pylibs -> core）实际解析 ====="
PYTHONPATH="$O3D:$BASE:$CORE" python3 - <<'PY' 2>&1 | tail -40
import importlib.util as u, sys, traceback
for name in ("numpy", "open3d", "cv2", "scipy", "PIL", "tqdm"):
    spec = u.find_spec(name)
    print(name, "->", (spec.origin if spec else "NOT FOUND"))
PY

echo
echo "===== 4. 直接复现官方 create_occupancy_grid_from_tsdf 的一步 ====="
PYTHONPATH="$O3D:$BASE:$CORE" python3 - <<'PY' 2>&1 | tail -60
import json, os, sys, traceback
import numpy as np
sys.path.insert(0, "/root/private_data/workspace/robo/third_party/upstream_eval/RoboTracer/Evaluation")
try:
    import summarize_acc as SA
    print("summarize_acc imported OK")
except Exception:
    traceback.print_exc(); raise SystemExit(1)

oj = "/root/private_data/workspace/robo/outputs/bench_public/tracespatial/official/outputs/RoboTracer_RoboBrain2.5-8B-NV_3D.jsonl"
ans = json.loads(open(oj, encoding="utf-8").readline())
bench = "/root/private_data/workspace/robo/third_party/benchmarks/TraceSpatial-Bench"
gt_depth = os.path.join(bench, "raw_data", ans["gt_depth_path"])
mask_path = os.path.join(bench, "raw_data", ans["mask_path"])
intr = np.array(ans["gt_depth_intrinsics"])
print("depth:", gt_depth, os.path.exists(gt_depth))
print("mask :", mask_path, os.path.exists(mask_path))

from PIL import Image
depth = np.array(Image.open(gt_depth)).astype(np.float32)
mask = np.array(Image.open(mask_path)).astype(np.uint8)
if mask.ndim == 3:
    mask = mask[:, :, 0]
print("depth shape", depth.shape, "range", float(depth.min()), float(depth.max()))
print("mask shape", mask.shape, "unique", np.unique(mask)[:5])

print("\n--- create_object_pcd_from_mask ---")
try:
    pts = SA.create_object_pcd_from_mask(mask_path, gt_depth, ans["gt_depth_intrinsics"])
    print("obj points:", None if pts is None else len(pts))
except Exception:
    traceback.print_exc()

print("\n--- create_occupancy_grid_from_tsdf ---")
try:
    g = SA.create_occupancy_grid_from_tsdf(depth, mask, ans["gt_depth_intrinsics"])
    print("occupied voxels:", len(g[0]))
except Exception:
    traceback.print_exc()
PY

echo
echo "===== 5. smoke3d 的 stderr ====="
python3 - <<'PY'
import json, os
p = "/root/private_data/workspace/robo/tmp/smoke3d.json"
if os.path.exists(p):
    d = json.load(open(p, encoding="utf-8"))
    for c in d.get("per_chunk", [])[:1]:
        print(c.get("stderr_tail"))
else:
    print("(no smoke3d.json)")
PY
