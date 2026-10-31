#!/usr/bin/env bash
# 精度收尾：重判 RefSpatial（补 FlagEvalMM 交叉核对）→ TraceSpatial → 稳定性检查 → 离线重判分
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

echo "===== [1/4] 重判 RefSpatial（resume：不再推理，只补 FlagEvalMM 交叉核对）====="
python3 "$WS/submission/bench_public/run_refspatial.py" \
  --url "$URL" --run-id baseline-eager-refspatial \
  --out-dir "$WS/outputs/bench_public/refspatial" 2>&1 | tee "$LOGDIR/refspatial_rescore.log"
echo "[1/4] exit=${PIPESTATUS[0]}"

echo
echo "===== [2/4] TraceSpatial-Bench（2D + 3D，官方几何判分）====="
python3 "$WS/submission/bench_public/run_tracespatial.py" \
  --url "$URL" --run-id baseline-eager-tracespatial \
  --out-dir "$WS/outputs/bench_public/tracespatial" 2>&1 | tee "$LOGDIR/tracespatial.log"
echo "[2/4] exit=${PIPESTATUS[0]}"

echo
echo "===== [3/4] 稳定性检查（每基准 12 样例 × 2 次重复）====="
python3 "$WS/submission/bench_public/stability.py" \
  --url "$URL" --out-dir "$WS/outputs/bench_public/stability" --n 12 \
  2>&1 | tee "$LOGDIR/stability.log"
echo "[3/4] exit=${PIPESTATUS[0]}"

echo
echo "===== [4/4] 离线重判分复现验证 ====="
python3 "$WS/submission/bench_public/rescore.py" \
  --bench-root "$WS/outputs/bench_public" 2>&1 | tee "$LOGDIR/rescore.log"
echo "[4/4] exit=${PIPESTATUS[0]}"

echo
echo "===== 全部收尾阶段结束 ====="
