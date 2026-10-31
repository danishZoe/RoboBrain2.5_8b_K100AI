#!/usr/bin/env bash
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
python3 - <<'PY'
import json
p = "outputs/bench_public_3d96/bench_public/tracespatial/official/scores_3dchunk_3D.json"
d = json.load(open(p, encoding="utf-8"))
print("top keys:", [k for k in d.keys()][:40])
for k in ("wall_s_total", "wall_s_all_jobs", "peak_rss_mb_max", "n_answers", "n_chunks", "n_jobs"):
    print(" ", k, "=", d.get(k))
pj = d.get("per_job") or d.get("jobs")
if isinstance(pj, list):
    for j in pj:
        print("  job:", {k: j.get(k) for k in ("name", "wall_s", "n_chunks", "n_answers", "returncode")})
PY
echo "=== ts_score_3d.log (中止的整集运行) ==="
head -3 logs/ts_score_3d.log
echo "  ... tail ..."
tail -6 logs/ts_score_3d.log
echo "=== mtime ==="
date -r logs/ts_score_3d.log '+%F %T'
