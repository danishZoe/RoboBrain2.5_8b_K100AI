#!/usr/bin/env bash
# 候选配置（E1：非 eager）精度回归：同样本、同协议、全量重跑
# 唯一变量 = 服务是否带 --enforce-eager
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
OUT="$WS/outputs/bench_public_noneager"
LOGDIR="$WS/logs/public_bench_noneager-candidate"
mkdir -p "$LOGDIR" "$OUT"

echo "===== [1/6] 停 eager 服务，切到非 eager（graph）服务 ====="
if [ -f "$WS/outputs/.bench_server_eager.pid" ]; then
  PID=$(cat "$WS/outputs/.bench_server_eager.pid")
  echo "  停 eager 服务 pid=$PID"
  kill "$PID" 2>/dev/null || true
  for i in $(seq 1 30); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 2
  done
  kill -9 "$PID" 2>/dev/null || true
fi
sleep 5
bash "$WS/tools/start_bench_service.sh" graph 30000 2>&1 | tail -n 5
BASE="${URL%/v1/chat/completions}"
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$BASE/health" || echo 000)
  [ "$code" = "200" ] && break
  sleep 10
done
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$BASE/health" || echo 000)
[ "$code" = "200" ] || { echo "  非 eager 服务未就绪 ($code)"; exit 1; }
echo "  非 eager 服务就绪"

echo
echo "===== [2/6] RefSpatial 全量 ====="
python3 "$WS/submission/bench_public/run_refspatial.py" \
  --url "$URL" --run-id noneager-candidate-refspatial --out-dir "$OUT/refspatial" \
  2>&1 | tee "$LOGDIR/refspatial.log" | grep -E 'CLI acc|默认范围'
echo "[2/6] exit=${PIPESTATUS[0]}"

echo
echo "===== [3/6] CV-Bench 全量 ====="
python3 "$WS/submission/bench_public/run_cv_bench.py" \
  --url "$URL" --run-id noneager-candidate-cvbench --out-dir "$OUT/cvbench" \
  2>&1 | tee "$LOGDIR/cvbench.log" | tail -n 12
echo "[3/6] exit=${PIPESTATUS[0]}"

echo
echo "===== [4/6] TraceSpatial 全量 ====="
python3 "$WS/submission/bench_public/run_tracespatial.py" \
  --url "$URL" --run-id noneager-candidate-tracespatial --out-dir "$OUT/tracespatial" \
  2>&1 | tee "$LOGDIR/tracespatial.log" | grep -E '^\[tracespatial\]|^    '
echo "[4/6] exit=${PIPESTATUS[0]}"

echo
echo "===== [5/6] 候选侧离线重判分复现验证 ====="
python3 "$WS/submission/bench_public/rescore.py" \
  --bench-root "$OUT" 2>&1 | tee "$LOGDIR/rescore.log" | tail -n 10
echo "[5/6] exit=${PIPESTATUS[0]}"

echo
echo "===== [6/6] 精度回归对比 ====="
python3 "$WS/submission/bench_public/compare_accuracy_runs.py" \
  --reference "$WS/outputs/bench_public" --candidate "$OUT" \
  --out "$WS/outputs/bench_public/accuracy_regression.md" 2>&1 | tail -n 40
echo "[6/6] exit=${PIPESTATUS[0]}"

echo
echo "===== 候选配置精度回归完成 ====="
