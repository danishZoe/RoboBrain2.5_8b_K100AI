#!/usr/bin/env bash
# TraceSpatial 3D 官方判分：补跑 4 个深景病态样本（56/78/82/98），补全 100/100 口径
#
# 背景：官方 compute_accuracy_3d 对这 4 条 480x640 深景样本的目标点云做
# cluster_dbscan(eps=0.02)，单条 >65 min（近似二次复杂度）。96 条口径见
# tools/run_ts_score_3d_96.sh；本脚本只对这 4 条按官方脚本**逐字**判分。
#
# 官方代码零改动；work-dir 与生产判分完全隔离。
# 影子根里的 jsonl 只含原文件第 56/78/82/98 行（0-based），
# 分片 i == 原样本索引 [56,78,82,98][i]。
#
# 产物：outputs/bench_public_ts3d4/<src>/tracespatial/<variant>/scores_3dchunk_3D.json
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

IDX="56 78 82 98"
ROOT="$WS/outputs/bench_public_ts3d4"
LOG="$WS/logs/ts_score_3d_4.log"

# 影子 bench root：jsonl 只保留 4 条目标样本
for SRC in bench_public bench_public_noneager; do
  for V in official upstream; do
    SRCDIR="$WS/outputs/$SRC/tracespatial/$V/outputs"
    D="$ROOT/$SRC/tracespatial/$V/outputs"
    mkdir -p "$D"
    for F in "$SRCDIR"/*.jsonl; do
      [ -e "$F" ] || continue
      B=$(basename "$F")
      case "$B" in
        *_3D.jsonl)
          python3 - "$F" "$D/$B" $IDX <<PY
import sys
src, dst, idx = sys.argv[1], sys.argv[2], set(map(int, sys.argv[3:]))
lines = [l for l in open(src, encoding="utf-8").read().splitlines() if l.strip()]
kept = [l for i, l in enumerate(lines) if i in idx]
assert len(kept) == len(idx), (src, len(kept), len(idx))
open(dst, "w", encoding="utf-8").write("\n".join(kept) + "\n")
print("filtered %s -> %s (%d lines)" % (src, dst, len(kept)))
PY
          ;;
      esac
    done
  done
done

echo "===== TraceSpatial 3D 病态样本补跑（56/78/82/98）$(date +%F %T) =====" | tee "$LOG"
date +%s > "$WS/tmp/ts3d_4.start.ts"
python3 "$WS/submission/bench_public/score_3d_chunked.py" \
  --job "$ROOT/bench_public/tracespatial|official|3D" \
  --job "$ROOT/bench_public/tracespatial|upstream|3D" \
  --job "$ROOT/bench_public_noneager/tracespatial|official|3D" \
  --job "$ROOT/bench_public_noneager/tracespatial|upstream|3D" \
  --chunk-size 1 --concurrency 4 --mem-budget-gb 45 \
  --work-dir "$WS/tmp/ts3d_4" \
  2>&1 | tee -a "$LOG" | grep -E "排除样本|任务|完成|->|audit|Skip"
RC=$?
date +%s > "$WS/tmp/ts3d_4.end.ts"
echo "===== 补跑结束 rc=$RC $(date +%F %T) =====" | tee -a "$LOG"
exit $RC
