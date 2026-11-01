#!/usr/bin/env bash
# =============================================================================
# 16K/并发64 补跑链（把六档表填满）
#
# 背景（2026-11-01）：
#   SCNet 容器换卡后需在新卡上补齐「最优版 16384/并发64」与「首提基线 eager
#   16384/并发64」各一轮，使六档表两侧都是同一张卡。单档 128 请求 × 16K 上下文
#   ≈ 3.8 h，无法再压缩。
#
# 为什么不能用 chain_remaining.sh 重跑（已查明的三处缺陷）：
#   [缺陷1] 上一轮 scripts/run_full_bench.sh 在**运行中被改写**（12:16 的修复脚本
#           覆盖了正在执行的同一文件），bash 按旧字节偏移读新内容，报
#           "line 227: unexpected EOF"，在跑完最后一档后**没走到第 214 行的
#           "停止服务"步骤** → 服务进程存活至今（PID 5304）。
#   [缺陷2] chain_remaining.sh 只等 run_same_card_bench.sh 退出，**不等服务退出**，
#           接着调 run_full_bench.sh，被其第 95 行的 guard 判定"已有服务进程运行"
#           直接 exit 4 → 16K/c64 两轮各 1 秒内失败（rc=4）。
#   [缺陷3] 因此本脚本在每轮前**显式停掉残留服务并等待显存归零**，
#           在每轮后**确认服务确已停止**，再进入下一轮。
#
# 本脚本自身放在独立文件、运行期间**不做任何原地改写**，避免重蹈缺陷1。
#
# 流程：
#   0) 清理残留服务 + 等待显存归零（同时记录当前服务状态，供回退参考）
#   1) 最优版（candidate）16384/并发64 单轮
#   2) 停服务 + 等待显存归零
#   3) 首提基线（eager）16384/并发64 单轮
#   4) 停服务 + 等待显存归零
#   5) 收尾：回填报告表格（finalize_after_bench.sh）
#
# 任一步失败都记录 rc 并继续往下跑，不中断整条链。
# =============================================================================
set -uo pipefail

WS="${WS:-/root/private_data/workspace/robo}"
PORT="${PORT:-30000}"
TIER="${TIER:-16384:64:1}"
cd "$WS" || exit 1
mkdir -p logs tmp outputs

LOG="$WS/logs/chain_c64_$(date +%Y%m%d_%H%M%S).log"
exec >> "$LOG" 2>&1

echo "============================================================"
echo "=== 16K/c64 补跑链启动 $(date -Is) ==="
echo "============================================================"
echo "ws=$WS port=$PORT tier=$TIER"

# --- 卡指纹（决定这批数据能否与既有同卡数据相减）---------------------------
# shellcheck disable=SC1091
source "$WS/scripts/card_identity.sh" 2>/dev/null || true
echo "card_id=${CARD_ID:-unknown}"
echo "card_uid=${CARD_UID:-}"
echo "card_name=${CARD_NAME:-}"
echo "node_host=${CARD_HOST:-}"
echo "boot_id=${CARD_BOOT_ID:-}"

# --- 通用：停掉服务并等待显存归零 -------------------------------------------
# 返回 0 = 已确认无服务在跑；返回 1 = 仍有残留（调用方应放弃本轮）
stop_service_and_wait() {
  echo "--- 停止服务 ---"
  pkill -f "$WS/submission/scripts/serve_vllm_patched.py" 2>/dev/null
  pkill -f "VLLM::EngineCore" 2>/dev/null
  sleep 8
  pkill -9 -f "$WS/submission/scripts/serve_vllm_patched.py" 2>/dev/null
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null

  for i in $(seq 1 60); do
    if ! pgrep -f "$WS/submission/scripts/serve_vllm_patched.py" >/dev/null 2>&1 \
       && ! pgrep -f "VLLM::EngineCore" >/dev/null 2>&1; then
      echo "--- 服务已停止（等待 ${i}0s 内确认） ---"
      sleep 5
      return 0
    fi
    sleep 10
  done
  echo "!!! 仍有服务残留进程，拒绝继续（避免重演 guard rc=4）" >&2
  pgrep -af "serve_vllm_patched|EngineCore" >&2
  return 1
}

# --- 通用：跑一轮并记录 rc --------------------------------------------------
run_one() {
  cfg="$1"; label="$2"
  echo
  echo "============================================================"
  echo "[c64] 开始：$label（config=$cfg, tiers=$TIER）  $(date -Is)"
  echo "============================================================"
  if [ "$cfg" = "candidate" ]; then
    bash "$WS/scripts/run_same_card_bench.sh" --skip-eager --skip-baseline \
         --tiers "$TIER" --port "$PORT"
  else
    bash "$WS/scripts/run_same_card_bench.sh" --skip-baseline --skip-candidate \
         --tiers "$TIER" --port "$PORT"
  fi
  rc=$?
  echo "[c64] 结束：$label  rc=$rc  $(date -Is)"
  ls -dt "$WS/outputs/fullbench_${cfg}_"* 2>/dev/null | head -1 \
    > "$WS/outputs/.c64_${cfg}_rundir"
  echo "[c64] run_dir=$(cat "$WS/outputs/.c64_${cfg}_rundir" 2>/dev/null)"
  return $rc
}

RC_CAND=99
RC_EAGER=99
RC_FINAL=99

# --- 0) 清理残留服务（修复缺陷1/2 的直接后果）-------------------------------
echo
echo "[c64] === 步骤 0：清理残留服务 $(date -Is) ==="
if ! stop_service_and_wait; then
  echo "[c64] 步骤0 失败：服务无法停止，链终止" >&2
  exit 7
fi

# --- 1) 最优版 16K/c64 单轮 --------------------------------------------------
echo
echo "[c64] === 步骤 1：最优版 16K/并发64 单轮 $(date -Is) ==="
run_one candidate "最优版（提交配置）"
RC_CAND=$?
echo "[c64] 步骤1 rc=$RC_CAND"

# 无论成功失败都要停服务，避免下一轮撞 guard
echo
echo "[c64] === 步骤 1.5：停止服务 $(date -Is) ==="
if ! stop_service_and_wait; then
  echo "[c64] 步骤1.5 失败：服务无法停止，链终止（避免重演 guard rc=4）" >&2
  exit 7
fi

# --- 2) 首提基线 eager 16K/c64 单轮 ----------------------------------------
echo
echo "[c64] === 步骤 2：首提基线 eager 16K/并发64 单轮 $(date -Is) ==="
run_one eager "首提基线 eager（赛事首提口径）"
RC_EAGER=$?
echo "[c64] 步骤2 rc=$RC_EAGER"

echo
echo "[c64] === 步骤 2.5：停止服务 $(date -Is) ==="
if ! stop_service_and_wait; then
  echo "[c64] 步骤2.5 失败：服务无法停止，链终止" >&2
  exit 7
fi

# --- 3) 收尾：回填报告 ------------------------------------------------------
echo
echo "[c64] === 步骤 3：收尾（回填报告/归集/自检） $(date -Is) ==="
if [ -x "$WS/scripts/finalize_after_bench.sh" ] || [ -f "$WS/scripts/finalize_after_bench.sh" ]; then
  bash "$WS/scripts/finalize_after_bench.sh"
  RC_FINAL=$?
else
  echo "[c64] 未找到 finalize_after_bench.sh，跳过回填"
  RC_FINAL=0
fi

echo
echo "============================================================"
echo "=== 16K/c64 补跑链结束 $(date -Is) ==="
echo "最优版 rc=$RC_CAND   基线eager rc=$RC_EAGER   收尾 rc=$RC_FINAL"
echo "candidate run_dir = $(cat "$WS/outputs/.c64_candidate_rundir" 2>/dev/null)"
echo "eager     run_dir = $(cat "$WS/outputs/.c64_eager_rundir" 2>/dev/null)"
echo "============================================================"
