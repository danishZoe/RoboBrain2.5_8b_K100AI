#!/usr/bin/env bash
# TraceSpatial 3D 官方判分（**排除 4 个深景病态样本**的兜底口径）
#
# 官方 compute_accuracy_3d 对样本 56/78/82/98 的目标点云做 cluster_dbscan(eps=0.02)，
# 单条 >65 min 仍未结束。整集单进程不可行，逐样本分片也会被它们拖住。
# 本脚本排除这 4 条，其余 96 条按官方脚本**逐字**判分，分母如实标注为 96。
#
# 产物：outputs/bench_public_3d96/tracespatial/<variant>/scores_3dchunk_3D.json
# 官方代码零改动；work-dir 与生产判分完全隔离，避免互相覆盖。
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

EXCL="56,78,82,98"
ROOT="$WS/outputs/bench_public_3d96"
LOG="$WS/logs/ts_score_3d_96.log"

# 搭一个只含输入 jsonl 的影子 bench root（软链，避免复制大文件）
for SRC in bench_public bench_public_noneager; do
  for V in official upstream; do
    D="$ROOT/$SRC/tracespatial/$V/outputs"
    mkdir -p "$D"
    for F in "$WS/outputs/$SRC/tracespatial/$V/outputs"/*.jsonl; do
      [ -e "$F" ] || continue
      B=$(basename "$F")
      [ -e "$D/$B" ] || ln -s "$F" "$D/$B"
    done
  done
done

echo "===== TraceSpatial 3D 兜底判分（排除 $EXCL）$(date '+%F %T') =====" | tee "$LOG"
python3 "$WS/submission/bench_public/score_3d_chunked.py" \
  --job "$ROOT/bench_public/tracespatial|official|3D" \
  --job "$ROOT/bench_public/tracespatial|upstream|3D" \
  --job "$ROOT/bench_public_noneager/tracespatial|official|3D" \
  --job "$ROOT/bench_public_noneager/tracespatial|upstream|3D" \
  --chunk-size 1 --concurrency 16 --mem-budget-gb 45 \
  --exclude "$EXCL" \
  --work-dir "$WS/tmp/ts3d_96" \
  2>&1 | tee -a "$LOG" | grep -E '排除样本|任务|完成|->|audit'

echo
echo "===== 结果速览 ====="
python3 - <<'PY'
import json, os
WS = "/root/private_data/workspace/robo"
ROOT = WS + "/outputs/bench_public_3d96"
for src in ("bench_public", "bench_public_noneager"):
    print("\n--- %s ---" % src)
    for v in ("official", "upstream"):
        p = os.path.join(ROOT, src, "tracespatial", v, "scores_3dchunk_3D.json")
        if not os.path.exists(p):
            print("  %-9s (缺)" % v)
            continue
        d = json.load(open(p, encoding="utf-8"))
        a = d.get("audit", {})
        print("  %-9s start=%-8s end=%-8s overall=%-8s no_collision=%-8s | "
              "n_eval=%d skip=%d excluded=%d" % (
                  v, d.get("start_success_rate"), d.get("end_success_rate"),
                  d.get("overall_success_rate"), d.get("no_collision_rate"),
                  a.get("n_evaluated_total", 0), a.get("n_skipped_total", 0),
                  d.get("n_excluded_samples", 0)))
        print("            metrics_3d=%s" % json.dumps(d.get("metrics_3d"), ensure_ascii=False))
PY
echo "===== 兜底判分结束 $(date '+%F %T') =====" | tee -a "$LOG"
