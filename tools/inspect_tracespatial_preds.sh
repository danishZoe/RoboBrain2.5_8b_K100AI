#!/usr/bin/env bash
# TraceSpatial 预测内容体检：点数、坐标范围、深度分布（排查 3D 判分耗时）
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
python3 - <<'PY'
import json, os, re, statistics
B = "/root/private_data/workspace/robo/outputs/bench_public/tracespatial"
for variant in ("official", "upstream"):
    for task in ("2D", "3D"):
        p = os.path.join(B, variant, "predictions", f"{task}.jsonl")
        if not os.path.exists(p):
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        npts, xs, ys, ds, bad = [], [], [], [], 0
        empty = 0
        for r in rows:
            t = r.get("raw_output") or ""
            m = re.findall(r"\(([^()]*)\)", t)
            pts = []
            for s in m:
                parts = [x.strip() for x in s.split(",")]
                try:
                    vals = [float(x) for x in parts]
                except ValueError:
                    continue
                if len(vals) >= 2:
                    pts.append(vals)
            if not pts:
                empty += 1
                continue
            npts.append(len(pts))
            for v in pts:
                xs.append(v[0]); ys.append(v[1])
                if len(v) >= 3:
                    ds.append(v[2])
        print(f"--- {variant}/{task}  n={len(rows)} 空输出={empty} ---")
        if npts:
            print("   每题点数: min=%d max=%d mean=%.1f" %
                  (min(npts), max(npts), statistics.mean(npts)))
        if xs:
            print("   x: min=%.0f max=%.0f   y: min=%.0f max=%.0f" %
                  (min(xs), max(xs), min(ys), max(ys)))
        if ds:
            print("   深度 d: min=%.3f max=%.3f mean=%.3f" %
                  (min(ds), max(ds), statistics.mean(ds)))
        if xs and (max(xs) > 1000 or max(ys) > 1000):
            print("   ⚠ 存在 >1000 的坐标")
        if ds and max(ds) > 10:
            print("   ⚠ 存在 >10m 的深度（可能拖慢 TSDF 积分）")
        # 样例
        for r in rows[:2]:
            print("   样例:", (r.get("raw_output") or "")[:150])
PY
