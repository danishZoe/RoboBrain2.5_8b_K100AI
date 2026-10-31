#!/usr/bin/env bash
# 诊断：TraceSpatial 的深度图布局 —— gt_depth 是否只覆盖目标物体？
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONPATH="$WS/envs/pylibs_o3d:$WS/envs/pylibs:$WS/envs/pylibs_core"
export PYTHONWARNINGS=ignore
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

python3 - <<'PY'
import json, os, glob
import numpy as np
from PIL import Image

BENCH = "/root/private_data/workspace/robo/third_party/benchmarks/TraceSpatial-Bench"
OJ = "/root/private_data/workspace/robo/outputs/bench_public/tracespatial/official/outputs/RoboTracer_RoboBrain2.5-8B-NV_3D.jsonl"
ans = [json.loads(l) for l in open(OJ, encoding="utf-8") if l.strip()]
print("official jsonl 样本数:", len(ans))
print("字段:", sorted(ans[0].keys()))
print()

print("%-6s %-26s %-22s %10s %10s %10s %10s" % (
    "idx", "gt_depth_path", "depth_path?", "gt_nz%", "mask_nz%", "ovlap%", "other_nz%"))
for i, a in enumerate(ans[:8]):
    gp = os.path.join(BENCH, "raw_data", a["gt_depth_path"])
    gt = np.array(Image.open(gp))
    gt_nz = (gt > 0).mean() * 100
    mp = os.path.join(BENCH, "raw_data", a["mask_path"])
    mk = np.array(Image.open(mp))
    if mk.ndim == 3:
        mk = mk[:, :, 0]
    mk_nz = (mk > 0).mean() * 100
    # 对齐
    if gt.shape[:2] != mk.shape[:2]:
        mk_r = np.array(Image.open(mp).resize((gt.shape[1], gt.shape[0])))
        if mk_r.ndim == 3:
            mk_r = mk_r[:, :, 0]
    else:
        mk_r = mk
    both = ((gt > 0) & (mk_r > 0)).sum()
    ov = both / max((mk_r > 0).sum(), 1) * 100
    # 另一个 depth 文件
    other = os.path.join(BENCH, "raw_data", os.path.basename(a["gt_depth_path"]).replace("gt_depth", "depth"))
    o_nz = "n/a"
    if os.path.exists(other) and other != gp:
        o = np.array(Image.open(other))
        o_nz = "%.2f" % ((o > 0).mean() * 100)
    print("%-6d %-26s %-22s %10.2f %10.2f %10.2f %10s" % (
        i, a["gt_depth_path"], os.path.basename(other), gt_nz, mk_nz, ov, o_nz))

print()
print("=== 目录里每个样本有哪些文件 ===")
d0 = os.path.join(BENCH, "raw_data", ans[0]["gt_depth_path"].split("/")[0])
for f in sorted(os.listdir(d0)):
    p = os.path.join(d0, f)
    try:
        a = np.array(Image.open(p))
        nz = (a > 0).mean() * 100
        print("  %-16s shape=%-14s dtype=%-8s 非零%%=%.2f" % (f, str(a.shape), str(a.dtype), nz))
    except Exception as e:
        print("  %-16s (%s)" % (f, e))
PY
