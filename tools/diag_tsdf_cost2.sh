#!/usr/bin/env bash
set -uo pipefail
WS=/root/private_data/workspace/robo
RT="$WS/third_party/upstream_eval/RoboTracer/Evaluation/summarize_acc.py"

echo "=== pcd_denoise_dbscan ==="
grep -n "def pcd_denoise_dbscan" -A 30 "$RT"
echo
echo "=== create_object_pcd_from_mask ==="
grep -n "def create_object_pcd_from_mask" -A 40 "$RT"
echo
echo "=== calculate_trajectory_collisions ==="
grep -n "def calculate_trajectory_collisions" -A 45 "$RT"
echo
echo "=== 病态样本 vs 正常样本：目标掩码面积 / 深度有效像素 ==="
cd "$WS"
python3 - <<'PY'
import json, os
import numpy as np
from PIL import Image
WS = "/root/private_data/workspace/robo"
p = os.path.join(WS, "outputs/bench_public/tracespatial/official/outputs",
                 "RoboTracer_RoboBrain2.5-8B-NV_3D.jsonl")
lines = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
bench = os.path.join(WS, "third_party/benchmarks/TraceSpatial-Bench/raw_data")
for i in (56, 78, 82, 98, 0, 1, 50):
    a = lines[i]
    mk = np.array(Image.open(os.path.join(bench, a["mask_path"])))
    dp = np.array(Image.open(os.path.join(bench, a["gt_depth_path"])))
    if mk.ndim == 3:
        mk = mk[:, :, 0]
    obj_px = int((mk == 255).sum())
    valid = int(((dp > 0) & (dp < 15000)).sum())
    dmax = int(dp.max())
    # 粗估 TSDF 体素数：视场宽度 x 深度范围
    print("  样本 %-3d mask=%-12s 目标像素=%-8d 有效深度像素=%-8d 深度max=%-6d 掩码占比=%.3f" %
          (i, str(mk.shape), obj_px, valid, dmax, obj_px / max(mk.size, 1)))
PY
