#!/usr/bin/env bash
# 精度回归的**噪声底**测量：同一配置、同一提示词，对全量 RefSpatial 连跑两次。
#
# 本次在 **graph（非 eager）** 服务下跑第 2 次，与两套已有结果同时比对：
#   * vs 候选（非 eager，第 1 次）→ 同配置重跑差值 = **噪声底**
#   * vs 基线（eager，第 1 次）  → 配置变化 + 噪声的合计差值
#
# 依据：reports/17 §1.3.1 规则 R-2/R-3 —— 基线 vs 候选的差值要拿噪声底做参照，
#      差值落在噪声底以内就不能归因于服务配置变化。
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONPATH="$WS/envs/pylibs_core:$WS/third_party/upstream_eval/FlagEvalMM:$WS/scripts:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore
export HF_HOME="$WS/cache/hf"
export XDG_CACHE_HOME="$WS/cache/xdg"
export TMPDIR="$WS/tmp"
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

URL=http://127.0.0.1:30000/v1/chat/completions
OUT="$WS/outputs/bench_public_noisecheck"
LOGD="$WS/logs/public_bench_noneager-candidate"
mkdir -p "$OUT" "$LOGD"

echo "===== 服务状态 ====="
curl -s -o /dev/null -w 'health=%{http_code}\n' --max-time 5 http://127.0.0.1:30000/health

echo
echo "===== 同配置重跑（第 2 次）RefSpatial 全量（graph 配置）====="
python3 "$WS/submission/bench_public/run_refspatial.py" \
  --url "$URL" --run-id noisecheck-repeat-refspatial \
  --out-dir "$OUT/refspatial" --no-resume \
  2>&1 | tee "$LOGD/noisecheck_refspatial.log" | grep -E 'CLI acc|默认范围' || true

echo
echo "===== 噪声底对比 ====="
python3 - <<'PY'
import json, os

BASE = "/root/private_data/workspace/robo/outputs/bench_public/refspatial/summary.json"
CAND = "/root/private_data/workspace/robo/outputs/bench_public_noneager/refspatial/summary.json"
REPT = "/root/private_data/workspace/robo/outputs/bench_public_noisecheck/refspatial/summary.json"


def load(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def cells(d):
    """-> {(variant, split): acc}"""
    out = {}
    for v, vin in (d or {}).get("variants", {}).items():
        for s, sin in vin.get("splits", {}).items():
            acc = sin.get("scores", {}).get("official_cli", {}).get("official_accuracy")
            if acc is not None:
                out[(v, s)] = acc
    return out


def scope_mean(d):
    out = {}
    for v, vin in (d or {}).get("variants", {}).items():
        s = vin.get("splits", {})
        vals = [s[k]["scores"]["official_cli"]["official_accuracy"]
                for k in ("Location", "Placement") if k in s]
        if vals:
            out[v] = sum(vals) / len(vals)
    return out


b, c, r = load(BASE), load(CAND), load(REPT)
cb, cc, cr = cells(b), cells(c), cells(r)

print("%-18s %-10s %8s %8s %8s %8s %8s" % (
    "variant", "split", "基线eager", "候选non-e", "重跑non-e", "同配置差", "基线-重跑"))
rows = []
for k in sorted(set(cb) | set(cc) | set(cr)):
    x, y, z = cb.get(k), cc.get(k), cr.get(k)
    d_same = (z - y) if (y is not None and z is not None) else None
    d_base = (z - x) if (x is not None and z is not None) else None
    print("%-18s %-10s %8s %8s %8s %8s %8s" % (
        k[0], k[1],
        ("%.4f" % x) if x is not None else "—",
        ("%.4f" % y) if y is not None else "—",
        ("%.4f" % z) if z is not None else "—",
        ("%+.4f" % d_same) if d_same is not None else "—",
        ("%+.4f" % d_base) if d_base is not None else "—"))
    rows.append({"variant": k[0], "split": k[1], "baseline_eager": x,
                 "candidate_noneager": y, "repeat_noneager": z,
                 "diff_same_config": d_same, "diff_baseline_vs_repeat": d_base})

mb, mc, mr = scope_mean(b), scope_mean(c), scope_mean(r)
print("\n默认范围均值（Location+Placement，主口径 upstream_scaled / official_scaled）：")
for v in sorted(set(mb) | set(mc) | set(mr)):
    x, y, z = mb.get(v), mc.get(v), mr.get(v)
    d_same = (z - y) if (y is not None and z is not None) else None
    d_base = (z - x) if (x is not None and z is not None) else None
    print("  %-18s 基线=%-8s 候选=%-8s 重跑=%-8s 同配置差=%-9s 基线-重跑=%s" % (
        v,
        ("%.4f" % x) if x is not None else "—",
        ("%.4f" % y) if y is not None else "—",
        ("%.4f" % z) if z is not None else "—",
        ("%+.4f" % d_same) if d_same is not None else "—",
        ("%+.4f" % d_base) if d_base is not None else "—"))

json.dump({"rows": rows,
           "scope_mean_baseline_eager": mb,
           "scope_mean_candidate_noneager": mc,
           "scope_mean_repeat_noneager": mr,
           "note": "同配置差 = 候选(非eager,第1次) vs 重跑(非eager,第2次) = 噪声底"},
          open("/root/private_data/workspace/robo/outputs/bench_public_noisecheck/"
               "noise_floor.json", "w"), indent=2)
print("\n-> outputs/bench_public_noisecheck/noise_floor.json")
PY
echo "===== 噪声底测量结束 $(date '+%F %T') ====="
