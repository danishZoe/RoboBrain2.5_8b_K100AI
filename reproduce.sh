#!/usr/bin/env bash
# =============================================================================
# 一键复现入口（验收标准 §9「一键复现入口应能依次完成」）
#
#   run.sh → env_check.sh → smoke.sh → verify.py → BENCH_MODE=image bench.sh
#
# 每一步的真实退出码都会打印出来；整体退出码规则：
#   - run.sh / env_check.sh / smoke.sh / bench.sh 任一失败 → 整体失败
#   - verify.py 退出码 2 = 精度项 BLOCKED（官方基线未下发时的**预期**结果），
#     不算复现失败，但会在日志里明确写出；其它退出码才算失败。
#
# 用法:
#   bash reproduce.sh              # 完整六档压测（官方口径，约 4 小时/单卡）
#   BENCH_TIERS=4096:1 bash reproduce.sh   # 只跑指定档位（调试用）
# =============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
source ./env.sh
export BENCH_OUT_DIR="${BENCH_OUT_DIR:-$WORK_DIR/outputs/reproduce_bench_$(date +%Y%m%d_%H%M%S)_$$}"

RC=0
run_step() {
  local name="$1"; shift
  echo
  echo "=============================================================="
  echo "[reproduce] $name"
  echo "=============================================================="
  "$@"
  local rc=$?
  echo "[reproduce] $name -> exit=$rc"
  return $rc
}

run_step "1/5 run.sh（准备环境 + 启动服务 + 健康检查）" bash run.sh || RC=1

run_step "2/5 env_check.sh（环境采集 -> env.log）" bash env_check.sh || RC=1

run_step "3/5 smoke.sh（图文 Smoke -> smoke.log）" bash smoke.sh || RC=1

echo
echo "=============================================================="
echo "[reproduce] 4/5 verify.py（精度测评 -> verify_result.json）"
echo "=============================================================="
python3 verify.py
VRC=$?
echo "[reproduce] verify.py -> exit=$VRC"
if [ "$VRC" = "2" ]; then
  echo "[reproduce] verify.py 返回 2 = 精度项 BLOCKED：官方 GPU 基线 / 测试集 / 判分器未下发。"
  echo "[reproduce] 这是**预期**结果（验收标准 §2 不允许用自建数据顶替），不算复现失败。"
elif [ "$VRC" != "0" ]; then
  echo "[reproduce] verify.py 异常退出"; RC=1
fi

BENCH_MODE=${BENCH_MODE:-image}
echo
echo "[reproduce] 5/5 bench.sh（官方六档压测, BENCH_MODE=$BENCH_MODE）"
if [ "$BENCH_MODE" = "image" ]; then
  run_step "5/5 bench.sh（官方六档压测）" env BENCH_MODE=image bash bench.sh || RC=1
else
  run_step "5/5 bench.sh（官方六档压测）" env BENCH_MODE=text bash bench.sh || RC=1
fi

echo
echo "=============================================================="
echo "[reproduce] 产物："
for f in env.log server.log smoke.log verify_result.json "$BENCH_OUT_DIR/summary.md" "$BENCH_OUT_DIR/round1.jsonl" "$BENCH_OUT_DIR/round2.jsonl"; do
  if [ -s "$f" ]; then printf '  %-24s %10s B\n' "$f" "$(wc -c < "$f")"; else printf '  %-24s MISSING\n' "$f"; fi
done
echo "[reproduce] benchmark outputs: $BENCH_OUT_DIR"
echo "[reproduce] 整体退出码 = $RC"
exit $RC
