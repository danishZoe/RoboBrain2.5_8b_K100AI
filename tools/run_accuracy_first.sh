#!/usr/bin/env bash
# 精度优先：RefSpatial 全量（4 变体 × 3 split）→ CV-Bench 全量（2638）
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

URL=http://127.0.0.1:30000/v1/chat/completions
LOGDIR="$WS/logs/public_bench_baseline-eager"
mkdir -p "$LOGDIR"

echo "===== [1/2] RefSpatial-Bench 全量 ====="
python3 "$WS/submission/bench_public/run_refspatial.py" \
  --url "$URL" --run-id baseline-eager-refspatial \
  --out-dir "$WS/outputs/bench_public/refspatial" 2>&1 | tee "$LOGDIR/refspatial.log"
rc1=${PIPESTATUS[0]}
echo "[refspatial] exit=$rc1"

echo
echo "===== [2/2] CV-Bench 全量（2638 题）====="
python3 "$WS/submission/bench_public/run_cv_bench.py" \
  --url "$URL" --run-id baseline-eager-cvbench \
  --out-dir "$WS/outputs/bench_public/cvbench" 2>&1 | tee "$LOGDIR/cvbench.log"
rc2=${PIPESTATUS[0]}
echo "[cvbench] exit=$rc2"

echo
echo "===== 汇总 ====="
python3 - <<'PY'
import json, os
base = "/root/private_data/workspace/robo/outputs/bench_public"
for name in ("refspatial", "cvbench"):
    p = os.path.join(base, name, "summary.json")
    if os.path.exists(p):
        d = json.load(open(p, encoding="utf-8"))
        print(f"--- {name} ---")
        if name == "refspatial":
            for v, vi in d["variants"].items():
                print("  %-18s 默认范围均值=%s" % (v, vi.get("default_scope_mean_accuracy")))
                for sp, si in vi["splits"].items():
                    cli = si.get("scores", {}).get("official_cli", {})
                    print("      %-10s n=%s acc=%s eval=%s total=%s" %
                          (sp, si.get("n"), cli.get("official_accuracy"),
                           cli.get("official_evaluated"), cli.get("official_total")))
        else:
            print("  ", json.dumps(d.get("scores", {}).get("official_result", {}),
                                   ensure_ascii=False))
PY
echo "[accuracy] done"
