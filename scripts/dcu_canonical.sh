#!/bin/bash
W=/root/private_data/workspace/robo
cd "$W/outputs/bench_public/cvbench" 2>/dev/null || { echo "no cvbench dir"; exit 1; }
echo "=== canonical run dir ==="
ls -la | head -20
echo
echo "=== official result ==="
python3 - <<'PY'
import json, os
p = "/root/private_data/workspace/robo/outputs/bench_public/cvbench/cv_bench_official_result.json"
if os.path.exists(p):
    d = json.load(open(p))
    d.pop("scored_predictions", None)
    print(json.dumps(d, ensure_ascii=False, indent=2))
else:
    print("missing:", p)
PY
echo
echo "=== summary.json (key fields) ==="
python3 - <<'PY'
import json
p = "/root/private_data/workspace/robo/outputs/bench_public/cvbench/summary.json"
d = json.load(open(p))
for k in ("run_id","n_planned","n_returned","n_fallback_extraction","terminal_states",
          "scorer","dataset_source"):
    if k in d: print("%-24s %s" % (k, d[k]))
pr = d.get("protocol", {})
for k in ("decode","max_tokens","concurrency","image_protocol","prompt_style","task_filter"):
    if k in pr: print("%-24s %s" % (k, pr[k]))
sc = d.get("scores", {})
if "official_result" in sc:
    print("official_result:", json.dumps(sc["official_result"], ensure_ascii=False))
PY
