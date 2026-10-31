#!/usr/bin/env bash
set -uo pipefail
WS=/root/private_data/workspace/robo
cd "$WS"
echo "=== 96 样本兜底判分：4 个组合全量结果 ==="
python3 - <<'PY'
import json, os
WS = "/root/private_data/workspace/robo"
ROOT = WS + "/outputs/bench_public_3d96"
for src, label in (("bench_public", "基线 eager"), ("bench_public_noneager", "候选 非eager")):
    print("\n--- %s ---" % label)
    for v in ("official", "upstream"):
        p = os.path.join(ROOT, src, "tracespatial", v, "scores_3dchunk_3D.json")
        if not os.path.exists(p):
            print("  %-9s (缺)" % v)
            continue
        d = json.load(open(p, encoding="utf-8"))
        a = d.get("audit", {})
        print("  %-9s start=%.4f end=%.4f overall=%.4f no_collision=%.4f" % (
            v, d.get("start_success_rate") or 0, d.get("end_success_rate") or 0,
            d.get("overall_success_rate") or 0, d.get("no_collision_rate") or 0))
        print("            start_in_mask=%.4f end_in_bbox=%.4f start_dist=%.4f end_dist=%.4f" % (
            d.get("start_in_mask") or 0, d.get("end_in_bbox") or 0,
            d.get("start_distance") or 0, d.get("end_distance") or 0))
        print("            metrics_3d=%s" % json.dumps(d.get("metrics_3d"), ensure_ascii=False))
        print("            audit: n_eval=%d skip=%d excluded=%d collision=%d 3d_metric_n=%d" % (
            a.get("n_evaluated_total", 0), a.get("n_skipped_total", 0),
            d.get("n_excluded_samples", 0), a.get("collision_count", 0),
            a.get("n_metric3d_samples", 0)))
        warn3d = sum(c.get("counts", {}).get("warn3d_collision", 0)
                     for c in d.get("per_chunk", []))
        print("            warn3d_collision 合计=%d（=0 表示碰撞检查真实生效）" % warn3d)
PY
