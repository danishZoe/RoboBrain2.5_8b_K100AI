#!/usr/bin/env bash
set -uo pipefail
WS=/root/private_data/workspace/robo
cd "$WS"
export PYTHONPATH="$WS/envs/pylibs_core:$WS/third_party/upstream_eval/FlagEvalMM:$WS/scripts:${PYTHONPATH:-}"
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

echo "=== py_compile ==="
python3 -m py_compile submission/bench_public/score_3d_chunked.py && echo COMPILE_OK

echo
echo "=== 官方 TSDF / 占用栅格的成本来源 ==="
RT="$WS/third_party/upstream_eval/RoboTracer/Evaluation/summarize_acc.py"
grep -n "def create_occupancy_grid_from_tsdf" -A 45 "$RT" | head -60
echo
echo "--- voxel_length / sdf_trunc / depth_scale 等常量 ---"
grep -n "voxel_length\|sdf_trunc\|depth_scale\|collision_ratio_thresh\|start_thresh_m\|end_thresh_m" "$RT" | head -20

echo
echo "=== 4 个病态样本的深度图尺寸（估计 TSDF 体素数）==="
python3 - <<'PY'
import json, os
WS = "/root/private_data/workspace/robo"
p = os.path.join(WS, "outputs/bench_public/tracespatial/official/outputs",
                 "RoboTracer_RoboBrain2.5-8B-NV_3D.jsonl")
if not os.path.exists(p):
    print("缺", p); raise SystemExit
lines = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
print("总样本:", len(lines))
try:
    from PIL import Image
    import numpy as np
    bench = os.path.join(WS, "third_party/benchmarks/TraceSpatial-Bench/raw_data")
    for i in (56, 78, 82, 98, 0, 1):
        if i >= len(lines):
            continue
        a = lines[i]
        dp = os.path.join(bench, a["gt_depth_path"])
        try:
            im = Image.open(dp)
            arr = np.array(im)
            nz = int((arr > 0).sum())
            print("  样本 %-3d depth=%-12s shape=%-14s 非零=%-9d max=%s" %
                  (i, os.path.basename(dp), str(arr.shape), nz, arr.max()))
        except Exception as e:
            print("  样本 %-3d ERR %r  path=%s" % (i, e, dp))
except Exception as e:
    print("ERR", e)
PY
