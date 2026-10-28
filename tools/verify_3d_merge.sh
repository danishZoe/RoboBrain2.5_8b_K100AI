#!/usr/bin/env bash
# ============================================================================
# 分片合并精确性验证（TraceSpatial 3D 官方判分）
#
# 官方 summarize_acc.py 一字未改；这里只改变"喂给它的样本分组"，
# 比对两种分组方式是否给出同一个答案，以证明分片合并没有引入偏差。
#
# 阶段 A（对照）：--chunk-size 0 → **整集单进程**跑一次官方 CLI
# 阶段 B（被测）：--chunk-size 1 → 逐样本分片（每个样本一个官方进程）后合并
#                 （阶段 B 的产物由 tools/run_ts_score_3d.sh 产出，本脚本只读）
#
# ⚠️ 样本区间：官方 3D 判分在 **样本 56 / 78 / 82 / 98**（深景）上单条耗时 >45 min
#   （open3d 修复后 TSDF 真正生效 → 大点云 DBSCAN）。整集单进程会把它们串行化，
#   墙钟不可接受。因此阶段 A 取**前 N 条干净样本**，阶段 B 也只取同样这 N 条分片，
#   两者逐指标对拍 —— 合并口径的验证只需要覆盖正常样本，合并代码对所有样本同路径。
#
# ⚠️ work-dir 必须按 bench 根目录拆开：score_3d_chunked.py 的分片 tag 只取
#   out_dir 的 basename，outputs/bench_public/tracespatial 与
#   outputs/bench_public_noneager/tracespatial 会落到同一个 tag 目录。
# ============================================================================
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONPATH="$WS/envs/pylibs_core:$WS/third_party/upstream_eval/FlagEvalMM:$WS/scripts:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore
export TMPDIR="$WS/tmp"
export ROBO_O3D_PREFIX="${ROBO_O3D_PREFIX:-$WS/envs/pylibs_o3d_v2}"
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

LOG="$WS/logs/verify_3d_merge.log"
N=${N:-50}
OFFSET=${OFFSET:-0}
export N OFFSET
SKIP_A=${SKIP_A:-0}

run_stage () {   # $1=stage名 $2=chunk-size $3=concurrency $4=work-dir $5..=job specs
  local stage="$1" cs="$2" conc="$3" wd="$4"; shift 4
  local jargs=()
  for j in "$@"; do jargs+=(--job "$j"); done
  echo "" | tee -a "$LOG"
  echo "===== 阶段 $stage：chunk-size=$cs offset=$OFFSET limit=$N 并发=$conc work=$wd $(date '+%F %T') =====" | tee -a "$LOG"
  python3 "$WS/submission/bench_public/score_3d_chunked.py" \
    "${jargs[@]}" \
    --chunk-size "$cs" --concurrency "$conc" --mem-budget-gb 40 \
    --offset "$OFFSET" --limit "$N" \
    --work-dir "$wd" \
    2>&1 | tee -a "$LOG" | grep -E '任务|完成|->|audit'
}

# ---- 影子 bench root：只放输入 jsonl 的软链，让阶段A 的产物**不覆盖生产路径** ----
SHADOW="$WS/outputs/bench_public_verify"
for SRC in bench_public bench_public_noneager; do
  for V in official upstream; do
    D="$SHADOW/$SRC/tracespatial/$V/outputs"
    mkdir -p "$D"
    for F in "$WS/outputs/$SRC/tracespatial/$V/outputs"/*.jsonl; do
      [ -e "$F" ] || continue
      B=$(basename "$F")
      [ -e "$D/$B" ] || ln -s "$F" "$D/$B"
    done
  done
done

echo "===== 分片合并对拍 开始 $(date '+%F %T') =====" | tee -a "$LOG"
echo "样本区间：offset=$OFFSET limit=$N" | tee -a "$LOG"

# ---------------------------------------------------------------- 阶段 A ----
if [ "$SKIP_A" != "1" ]; then
  run_stage A 0 4 "$WS/tmp/ts3d_verify_full" \
    "$SHADOW/bench_public/tracespatial|official|3D" \
    "$SHADOW/bench_public/tracespatial|upstream|3D"
  run_stage A_ne 0 4 "$WS/tmp/ts3d_verify_full_ne" \
    "$SHADOW/bench_public_noneager/tracespatial|official|3D" \
    "$SHADOW/bench_public_noneager/tracespatial|upstream|3D"
else
  echo "" | tee -a "$LOG"
  echo "（SKIP_A=1：复用已存在的阶段A 产物，只重跑对拍）" | tee -a "$LOG"
fi

# -------------------------------------------------------------- 对拍 ----
echo "" | tee -a "$LOG"
echo "===== 逐指标对拍 =====" | tee -a "$LOG"
python3 - <<'PY' | tee -a "$LOG"
import json, os, sys

WS = "/root/private_data/workspace/robo"
sys.path.insert(0, WS + "/submission/bench_public")
import score_3d_chunked as S3

N = int(os.environ.get("N", "50"))
OFFSET = int(os.environ.get("OFFSET", "0"))

CASES = [
    ("基线 eager  official", "official",
     WS + "/outputs/bench_public/tracespatial/official",
     WS + "/outputs/bench_public_verify/bench_public/tracespatial/official"),
    ("基线 eager  upstream", "upstream",
     WS + "/outputs/bench_public/tracespatial/upstream",
     WS + "/outputs/bench_public_verify/bench_public/tracespatial/upstream"),
    ("候选 非eager official", "official",
     WS + "/outputs/bench_public_noneager/tracespatial/official",
     WS + "/outputs/bench_public_verify/bench_public_noneager/tracespatial/official"),
    ("候选 非eager upstream", "upstream",
     WS + "/outputs/bench_public_noneager/tracespatial/upstream",
     WS + "/outputs/bench_public_verify/bench_public_noneager/tracespatial/upstream"),
]

SCALARS = ("start_in_mask", "end_in_bbox", "start_distance", "end_distance",
           "no_collision_rate", "start_success_rate", "end_success_rate",
           "overall_success_rate")
DICTS = ("metrics_2d", "metrics_3d")
TOL_SCALAR = 1e-6
TOL_DICT = 2e-3
TOL_MEAN = 5e-3


def load(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def close(a, b, tol):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


all_ok = True
rows = []

for label, variant, vdir, fulldir in CASES:
    print("\n--- %s ---" % label)
    prod = load(os.path.join(vdir, "scores_3dchunk_3D.json"))
    prod_src = "生产判分（100 题口径）"
    if prod is None:
        # 生产判分若因深景病态样本未完成，退回「排除 4 个病态样本」的 96 题兜底口径。
        # 前 50 个分片对应原始样本 0..49，与阶段 A 的区间完全一致。
        alt_vdir = vdir.replace("/outputs/bench_public/tracespatial",
                                "/outputs/bench_public_3d96/bench_public/tracespatial")
        alt_vdir = alt_vdir.replace("/outputs/bench_public_noneager/tracespatial",
                                    "/outputs/bench_public_3d96/bench_public_noneager/tracespatial")
        prod = load(os.path.join(alt_vdir, "scores_3dchunk_3D.json"))
        prod_src = "96 题兜底口径（排除 4 个深景病态样本）"
    full = load(os.path.join(fulldir, "scores_3dchunk_3D.json"))
    if prod is None:
        print("  缺阶段B 产物（生产判分与 96 题兜底均未完成），跳过")
        all_ok = False
        continue
    if full is None or not full.get("per_chunk"):
        print("  缺阶段A 产物（整集单进程未完成），跳过")
        all_ok = False
        continue

    single_raw = full["per_chunk"][0]
    if single_raw.get("returncode") not in (0, None):
        print("  阶段A 官方 CLI 退出码=%s（判分未成功）" % single_raw.get("returncode"))
        all_ok = False
        continue
    print("  阶段A 整集单进程：样本=%d 墙钟=%.0fs 峰值RSS=%.0fMB" %
          (single_raw.get("n_answers", 0), single_raw.get("wall_s", 0),
           single_raw.get("peak_rss_mb", 0)))

    # 阶段B：取生产分片中与阶段A 同一区间的分片，用同一套 merge_chunks 合并
    per = prod.get("per_chunk") or []
    sel = per[OFFSET:OFFSET + N]
    chunked = S3.merge_chunks(sel, 1)
    print("  阶段B 逐样本分片：%s，参与合并的分片数=%d（该口径共 %d 片）" %
          (prod_src, len(sel), len(per)))

    for k in SCALARS:
        a = single_raw.get(k)
        b = chunked.get(k)
        tol = TOL_MEAN if k in ("start_distance", "end_distance") else TOL_SCALAR
        ok = close(a, b, tol)
        if not ok:
            all_ok = False
        print("    %-22s 整集A=%-14s 分片B=%-14s %s" %
              (k, a, b, "MATCH" if ok else "*** MISMATCH ***"))
        rows.append({"case": label, "metric": k, "single_full": a,
                     "chunk1_merged": b, "match": ok})

    for k in DICTS:
        a = single_raw.get(k) or {}
        b = chunked.get(k) or {}
        for kk in sorted(set(a) | set(b)):
            va, vb = a.get(kk), b.get(kk)
            ok = close(va, vb, TOL_DICT)
            if not ok:
                all_ok = False
            print("    %-26s 整集A=%-16s 分片B=%-16s %s" %
                  (k + "." + kk.split("_")[-1], va, vb,
                   "MATCH" if ok else "*** MISMATCH ***"))
            rows.append({"case": label, "metric": k + "." + kk,
                         "single_full": va, "chunk1_merged": vb, "match": ok})

    au = chunked.get("audit") or {}
    print("    -- 阶段B 审计分（区间内）--")
    for k in ("n_answers_total", "n_evaluated_total", "n_skipped_total",
              "start_success_count", "end_success_count", "collision_count",
              "overall_success_count", "n_start_distance_samples",
              "n_end_distance_samples", "n_metric3d_samples", "n_metric2d_samples",
              "n_ambiguous_allzero_metric3d", "n_ambiguous_allzero_metric2d"):
        print("    %-26s %s" % (k, au.get(k)))

print("\n===== 总结 =====")
print("  逐指标对拍：%s" % ("全部一致 ✅" if all_ok else "存在不一致 ❌"))
json.dump({"all_match": all_ok, "n_samples": N, "offset": OFFSET, "rows": rows},
          open(WS + "/outputs/bench_public/verify_3d_merge.json", "w"),
          ensure_ascii=False, indent=2)
print("  -> outputs/bench_public/verify_3d_merge.json")
PY

echo "===== 对拍结束 $(date '+%F %T') =====" | tee -a "$LOG"
