#!/usr/bin/env bash
# 3D 判分完成后的集成：回填 summary -> 汇总表 -> 精度回归对比
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONPATH="$WS/envs/pylibs_core:$WS/third_party/upstream_eval/FlagEvalMM:$WS/scripts:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore
export TMPDIR="$WS/tmp"
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

echo "===== ① 回填分片判分结果到 summary.json ====="
# 3D 判分主产物缺失时用「排除 4 个深景病态样本」的兜底口径兜底（分母会如实标注）
python3 "$WS/submission/bench_public/merge_ts_scores.py" \
  --bench-root "$WS/outputs/bench_public" \
  --alt-3d-root "$WS/outputs/bench_public_3d96"
python3 "$WS/submission/bench_public/merge_ts_scores.py" \
  --bench-root "$WS/outputs/bench_public_noneager" \
  --alt-3d-root "$WS/outputs/bench_public_3d96"

echo
echo "===== ② 汇总表 ====="
python3 "$WS/submission/bench_public/build_summary_table.py" \
  --bench-root "$WS/outputs/bench_public" --out "$WS/outputs/bench_public/summary_table.md"

echo
echo "===== ③ 精度回归对比（参考=eager基线 / 候选=非eager）====="
python3 "$WS/submission/bench_public/compare_accuracy_runs.py" \
  --reference "$WS/outputs/bench_public" --candidate "$WS/outputs/bench_public_noneager" \
  --out "$WS/outputs/bench_public/accuracy_regression.md" 2>&1 | tail -60

echo
echo "===== ④ TraceSpatial 结果速览 ====="
python3 - <<'PY'
import json, os
WS = "/root/private_data/workspace/robo"
for root, label in ((WS + "/outputs/bench_public", "基线 eager"),
                    (WS + "/outputs/bench_public_noneager", "候选 非eager")):
    print("\n--- %s ---" % label)
    for variant in ("official", "upstream"):
        for task in ("2D", "3D"):
            p = os.path.join(root, "tracespatial", variant,
                             "scores_3dchunk_%s.json" % task)
            if not os.path.exists(p):
                print("  %-9s/%-3s  (缺)" % (variant, task))
                continue
            d = json.load(open(p, encoding="utf-8"))
            a = d.get("audit", {})
            print("  %-9s/%-3s  start=%-8s end=%-8s overall=%-8s no_collision=%-8s | "
                  "skip=%d warn3d=%d n_eval=%d" % (
                      variant, task,
                      d.get("start_success_rate"), d.get("end_success_rate"),
                      d.get("overall_success_rate"), d.get("no_collision_rate"),
                      a.get("n_skipped_total", 0),
                      sum(c.get("counts", {}).get("warn3d_collision", 0)
                          for c in d.get("per_chunk", [])),
                      a.get("n_evaluated_total", 0)))
PY
echo "===== 集成完成 $(date '+%F %T') ====="
