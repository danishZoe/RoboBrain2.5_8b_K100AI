#!/usr/bin/env bash
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
python3 - <<'PY'
import json
d = json.load(open("outputs/bench_public/refspatial/summary.json", encoding="utf-8"))
s = d["variants"]["upstream_scaled"]["splits"]["Location"]["scores"]["flagevalmm_norm1k"]
print("returncode:", s.get("returncode"))
print("stdout:", s.get("stdout"))
print("stderr:", s.get("stderr"))
PY
echo "--- 直接复现 ---"
export PYTHONPATH="$WS/envs/pylibs_core:$WS/third_party/upstream_eval/FlagEvalMM:${PYTHONPATH:-}"
set +u; source /opt/dtk/env.sh >/dev/null 2>&1; set -u
cd "$WS/submission/bench_public" && python3 - <<'PY'
import sys
sys.path.insert(0, ".")
import official_flagevalmm as OF
try:
    P = OF.load_point_utils()
    print("load_point_utils OK:", [n for n in dir(P) if "pts" in n or "mask" in n])
except Exception as e:
    import traceback
    print("FAIL:", type(e).__name__, e)
    traceback.print_exc()
PY
