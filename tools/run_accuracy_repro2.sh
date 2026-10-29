#!/usr/bin/env bash
# 全量推理复现第二遍（验收要求 §6.8「全量重复运行证据」）
#
# 配置 = 候选配置（E1 非 eager / graph 服务），与 outputs/bench_public_noneager
# 第 1 次候选运行同服务参数、同提示词、同解码参数；唯一区别 = 运行时刻。
# 产物写入**新**根 outputs/bench_public_noneager_repro2/，不覆盖任何已有产物。
# 用途：三基准全量同配置重跑 -> 与第 1 次候选运行配对，量化全量运行间噪声，
#       补齐「全量推理可重复」证据。
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
OUT="$WS/outputs/bench_public_noneager_repro2"
LOGDIR="$WS/logs/public_bench_noneager-repro2"
mkdir -p "$LOGDIR" "$OUT"

echo "===== [0/5] 启动非 eager（graph）服务 $(date '+%F %T') ====="
bash "$WS/tools/start_bench_service.sh" graph 30000 2>&1 | tail -n 5
BASE="${URL%/v1/chat/completions}"
for i in $(seq 1 90); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$BASE/health" || echo 000)
  [ "$code" = "200" ] && break
  sleep 10
done
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$BASE/health" || echo 000)
[ "$code" = "200" ] || { echo "  非 eager 服务未就绪 ($code)"; exit 1; }
echo "  非 eager 服务就绪"

echo
echo "===== [1/5] RefSpatial 全量 ====="
python3 "$WS/submission/bench_public/run_refspatial.py" \
  --url "$URL" --run-id noneager-repro2-refspatial --out-dir "$OUT/refspatial" \
  2>&1 | tee "$LOGDIR/refspatial.log" | grep -E 'CLI acc|默认范围'
echo "[1/5] exit=${PIPESTATUS[0]}"

echo
echo "===== [2/5] CV-Bench 全量 ====="
python3 "$WS/submission/bench_public/run_cv_bench.py" \
  --url "$URL" --run-id noneager-repro2-cvbench --out-dir "$OUT/cvbench" \
  2>&1 | tee "$LOGDIR/cvbench.log" | tail -n 12
echo "[2/5] exit=${PIPESTATUS[0]}"

echo
echo "===== [3/5] TraceSpatial 全量 ====="
python3 "$WS/submission/bench_public/run_tracespatial.py" \
  --url "$URL" --run-id noneager-repro2-tracespatial --out-dir "$OUT/tracespatial" \
  2>&1 | tee "$LOGDIR/tracespatial.log" | tail -n 12
echo "[3/5] exit=${PIPESTATUS[0]}"

echo
echo "===== [4/5] 离线重判分复现检查（对本遍产物） ====="
python3 "$WS/submission/bench_public/rescore.py" \
  --bench-root "$OUT" 2>&1 | tee "$LOGDIR/rescore.log" | tail -n 6
echo "[4/5] exit=${PIPESTATUS[0]}"

echo
echo "===== [5/5] 与第 1 次候选运行配对比较 ====="
python3 "$WS/submission/bench_public/compare_accuracy_runs.py" \
  --reference "$WS/outputs/bench_public_noneager" \
  --candidate "$OUT" \
  --out "$OUT/repro2_vs_candidate1.md" 2>&1 | tee "$LOGDIR/compare.log" | tail -n 15
echo "[5/5] exit=${PIPESTATUS[0]}"

echo "===== 全量复现第二遍结束 $(date '+%F %T') ====="
date '+%s' > "$WS/tmp/repro2.end.ts"
