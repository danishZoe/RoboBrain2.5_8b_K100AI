#!/usr/bin/env bash
# ============================================================================
# 公开基准精度收尾 + E1 候选复测（并行编排）
#
# 背景：官方 TraceSpatial 3D 判分（TSDF 体素化 + DBSCAN）在少数"病理样本"上
# 单核退化，100 条全量单进程要 ~90 分钟；而本机有 32 核。
# 官方 summarize_acc.py **一字未改**，我们只把「互不依赖的判分任务」并发执行：
#   - 4 个 (变体 × 任务) 判分并发
#   - 正式判分 与 离线重判分 并发
#   - 基线 与 候选 两轮并发（候选的推理用 GPU，与基线的判分用 CPU，互不抢资源）
#
# 阶段：
#   A 基线推理收尾（官方变体已有预测；补 upstream 变体推理）
#   B 稳定性检查（需要服务）
#   C 基线判分 8 路并发（4 判分 + 4 重判分）      ← CPU
#   D 候选：切服务 → RefSpatial/CV-Bench 全量 → TraceSpatial 推理   ← GPU
#   E 候选判分 8 路并发                          ← CPU
#   F 汇总：重判分比对 + 精度回归对比 + 汇总表
# ============================================================================
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
BASE_OUT="$WS/outputs/bench_public"
CAND_OUT="$WS/outputs/bench_public_noneager"
LOGD="$WS/logs/public_bench_baseline-eager"
CLOGD="$WS/logs/public_bench_noneager-candidate"
mkdir -p "$LOGD" "$CLOGD" "$CAND_OUT"

say() { echo; echo "########## $* ($(date '+%F %T')) ##########"; }

# --------------------------------------------------------------------------- #
say "A. 基线 TraceSpatial 推理收尾（官方变体断点续跑；upstream 变体补推理）"
python3 "$WS/submission/bench_public/run_tracespatial.py" \
  --url "$URL" --run-id baseline-eager-tracespatial \
  --out-dir "$BASE_OUT/tracespatial" --stage infer \
  2>&1 | tee "$LOGD/tracespatial_infer.log" | grep -E 'variant=|完成|断点' || true

# --------------------------------------------------------------------------- #
say "B. 稳定性检查（每基准 12 样例 × 2 次重复）"
python3 "$WS/submission/bench_public/stability.py" \
  --url "$URL" --out-dir "$BASE_OUT/stability" --n 12 \
  2>&1 | tee "$LOGD/stability.log" | tail -n 20

# --------------------------------------------------------------------------- #
say "C. 基线判分 8 路并发（4 判分 + 4 离线重判分）"
(
  python3 "$WS/submission/bench_public/run_tracespatial.py" \
    --url "$URL" --run-id baseline-eager-tracespatial \
    --out-dir "$BASE_OUT/tracespatial" --stage score \
    > "$LOGD/tracespatial_score.log" 2>&1
  echo "baseline score exit=$?" >> "$LOGD/parallel_score.status"
) &
P1=$!
(
  python3 "$WS/submission/bench_public/rescore.py" \
    --bench-root "$BASE_OUT" --stage rescore \
    > "$LOGD/rescore_raw.log" 2>&1
  echo "baseline rescore exit=$?" >> "$LOGD/parallel_score.status"
) &
P2=$!
echo "  基线判分 pid=$P1 / 重判分 pid=$P2（后台）"

# --------------------------------------------------------------------------- #
say "D. 候选配置（E1 非 eager）：切服务 → 三基准推理（GPU）"
if [ -f "$WS/outputs/.bench_server_eager.pid" ]; then
  PID=$(cat "$WS/outputs/.bench_server_eager.pid")
  echo "  停 eager 服务 pid=$PID"
  kill "$PID" 2>/dev/null || true
  for i in $(seq 1 60); do kill -0 "$PID" 2>/dev/null || break; sleep 2; done
  kill -9 "$PID" 2>/dev/null || true
fi
sleep 5
bash "$WS/tools/start_bench_service.sh" graph 30000 2>&1 | tail -n 3

echo "--- D1 RefSpatial 全量（候选）---"
python3 "$WS/submission/bench_public/run_refspatial.py" \
  --url "$URL" --run-id noneager-candidate-refspatial --out-dir "$CAND_OUT/refspatial" \
  2>&1 | tee "$CLOGD/refspatial.log" | grep -E 'CLI acc|默认范围' || true

echo "--- D2 CV-Bench 全量（候选）---"
python3 "$WS/submission/bench_public/run_cv_bench.py" \
  --url "$URL" --run-id noneager-candidate-cvbench --out-dir "$CAND_OUT/cvbench" \
  2>&1 | tee "$CLOGD/cvbench.log" | grep -E 'accuracy|题目数' || true

echo "--- D3 TraceSpatial 推理（候选）---"
python3 "$WS/submission/bench_public/run_tracespatial.py" \
  --url "$URL" --run-id noneager-candidate-tracespatial \
  --out-dir "$CAND_OUT/tracespatial" --stage infer \
  2>&1 | tee "$CLOGD/tracespatial_infer.log" | grep -E 'variant=|完成|断点' || true

# --------------------------------------------------------------------------- #
say "E. 候选判分 8 路并发"
(
  python3 "$WS/submission/bench_public/run_tracespatial.py" \
    --url "$URL" --run-id noneager-candidate-tracespatial \
    --out-dir "$CAND_OUT/tracespatial" --stage score \
    > "$CLOGD/tracespatial_score.log" 2>&1
  echo "candidate score exit=$?" >> "$CLOGD/parallel_score.status"
) &
P3=$!
(
  python3 "$WS/submission/bench_public/rescore.py" \
    --bench-root "$CAND_OUT" --stage rescore \
    > "$CLOGD/rescore_raw.log" 2>&1
  echo "candidate rescore exit=$?" >> "$CLOGD/parallel_score.status"
) &
P4=$!
echo "  候选判分 pid=$P3 / 重判分 pid=$P4（后台）"

# CV-Bench 图像输入协议因子实验：此时 GPU 空闲、CPU 被判分占满，正好用 GPU
# （协议实验内部 8 组串行，保证组间同批次条件一致）
(
  bash "$WS/tools/exp_cvbench_image_protocol.sh" \
    > "$WS/logs/exp_cvbench_protocol.log" 2>&1
  echo "protocol experiment exit=$?" >> "$CLOGD/parallel_score.status"
) &
P5=$!
echo "  协议实验 pid=$P5（后台，占 GPU）"

# --------------------------------------------------------------------------- #
say "F. 等待全部判分完成"
wait $P1 $P2 $P3 $P4 $P5
echo "  全部判分结束 $(date '+%F %T')"
cat "$LOGD/parallel_score.status" 2>/dev/null
cat "$CLOGD/parallel_score.status" 2>/dev/null

say "G. 离线重判分比对 + 精度回归对比 + 汇总表"
python3 "$WS/submission/bench_public/rescore.py" --bench-root "$BASE_OUT" --stage compare \
  2>&1 | tee "$LOGD/rescore.log" | tail -n 8
python3 "$WS/submission/bench_public/rescore.py" --bench-root "$CAND_OUT" --stage compare \
  2>&1 | tee "$CLOGD/rescore.log" | tail -n 8
python3 "$WS/submission/bench_public/compare_accuracy_runs.py" \
  --reference "$BASE_OUT" --candidate "$CAND_OUT" \
  --out "$BASE_OUT/accuracy_regression.md" 2>&1 | tail -n 45
python3 "$WS/submission/bench_public/build_summary_table.py" \
  --bench-root "$BASE_OUT" --out "$BASE_OUT/summary_table.md" 2>&1 | tail -n 6

say "全部完成"
