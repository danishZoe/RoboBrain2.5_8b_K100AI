#!/usr/bin/env bash
# RefSpatial 判分链路小样本验证（5 题/组，隔离输出目录，不污染正式结果）
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONPATH="$WS/envs/pylibs_core:$WS/third_party/upstream_eval/FlagEvalMM:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore
export HF_HOME="$WS/cache/hf"
export XDG_CACHE_HOME="$WS/cache/xdg"
export TMPDIR="$WS/tmp"
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

python3 "$WS/submission/bench_public/run_refspatial.py" \
  --url http://127.0.0.1:30000/v1/chat/completions \
  --limit 5 --splits Location \
  --variants upstream_scaled,official_raw \
  --out-dir "$WS/outputs/bench_public/_smoke_refspatial_v3" 2>&1 | tail -n 40
echo "=== exit=$? ==="

echo
echo "===== 抽样看预测与官方判分输出 ====="
python3 - <<'PY'
import json, os
base = "/root/private_data/workspace/robo/outputs/bench_public/_smoke_refspatial_v3"
for variant in ("upstream_scaled", "official_raw"):
    p = os.path.join(base, variant, "outputs", f"RoboBrain2.5-8B-NV_{variant}", "Location.jsonl")
    if not os.path.exists(p):
        print(variant, "-> 无输出"); continue
    print(f"--- {variant} ---")
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        print("  id=%s  text=%r" % (r["question_id"], r["text"]))
        print("     instruction=%r" % r["instruction"][:150])
    sp = os.path.join(base, variant, "score_Location.json")
    if os.path.exists(sp):
        d = json.load(open(sp, encoding="utf-8"))
        print("  score:", {k: d[k] for k in ("evaluated", "total", "mean_accuracy",
                                            "audit_score_full_denominator", "no_point_parsed")})
PY
