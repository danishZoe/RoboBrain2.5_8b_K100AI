#!/usr/bin/env bash
# =============================================================================
# ⚠️ **本脚本已被 `scripts/chain_16k64.sh` 取代，请勿直接使用。**
#
#   v1（本脚本）的两个 16384/并发64 长轮都会被 `run_full_bench.sh` 的安全闸
#   拒绝（rc=4，「[guard] 已有服务进程运行，拒绝启动」）：上一轮抽样用的
#   vLLM 服务实例没有退出，安全闸拒绝再起一个配置。
#   实际执行的是 v2（`scripts/chain_16k64.sh`）：每个长轮**之前**强制清场
#   （杀残留服务 + 确认进程退出 + 确认 30000 端口空闲），清不掉就中止，
#   不产生「看起来跑了其实没跑」的目录。
#
#   下面保留 v1 原文，仅作过程存档。
# =============================================================================
# 无人值守续跑链（节点侧，PPID=1 脱离会话）
#
# 背景：SCNet 容器换卡后需要在新卡上补齐「最优版 16384/并发64」与
#       「基线 eager 16384/并发64」各一轮，把六档表填满且两侧都是同一张卡。
#       单档 128 请求 × 16k 上下文 ≈ 3.8 h，无法再压缩。
#
# 流程：
#   1) 等当前抽样（最优版 5 个便宜档）结束
#   2) 最优版 16384/并发64 单轮
#   3) 基线 eager 16384/并发64 单轮
#   4) 收尾（finalize_after_bench.sh：回填报告 → 归集 → 自检 → 导出 tarball → DELIVERY_READY）
#
# 任一步失败都继续往下跑并记录 rc，不中断整条链。
# =============================================================================
set -uo pipefail
WS="${WS:-/root/private_data/workspace/robo}"
cd "$WS" || exit 1
mkdir -p logs tmp
exec >> "$WS/logs/chain_remaining.log" 2>&1

echo "============================================================"
echo "=== 续跑链启动 $(date -Is) ==="
echo "============================================================"

# --- 1) 等当前抽样结束 ---------------------------------------------------------
waited=0
while pgrep -f 'run_same_card_bench.sh' >/dev/null 2>&1; do
  sleep 30
  waited=$((waited + 30))
  if [ $((waited % 1800)) -eq 0 ]; then
    echo "[chain] 仍在等当前抽样（$((waited / 60)) min）$(date -Is)"
    tail -3 logs/candidate_probe2.log 2>/dev/null | sed 's/^/    /'
  fi
done
echo "[chain] 当前抽样结束（等待 $waited s）$(date -Is)"

# --- 2) 最优版 16384/并发64 单轮 -----------------------------------------------
echo ""
echo "[chain] === 最优版 16384/并发64 单轮 开始 $(date -Is) ==="
bash scripts/run_same_card_bench.sh --skip-eager --skip-baseline \
     --tiers "16384:64:1" --port 30000
rc1=$?
echo "[chain] === 最优版 16K/c64 结束 rc=$rc1 $(date -Is) ==="

# --- 3) 基线 eager 16384/并发64 单轮 -------------------------------------------
echo ""
echo "[chain] === 基线 eager 16384/并发64 单轮 开始 $(date -Is) ==="
bash scripts/run_same_card_bench.sh --skip-baseline --skip-candidate \
     --tiers "16384:64:1" --port 30000
rc2=$?
echo "[chain] === 基线 16K/c64 结束 rc=$rc2 $(date -Is) ==="

# --- 4) 收尾（此时无压测在跑，finalize_after_bench.sh 会立即执行）--------------
echo ""
echo "[chain] === 收尾开始 $(date -Is) ==="
bash scripts/finalize_after_bench.sh
rc3=$?
echo "[chain] === 收尾结束 rc=$rc3 $(date -Is) ==="
echo "[chain] 最优版16K/c64 rc=$rc1  基线16K/c64 rc=$rc2  收尾 rc=$rc3"
echo "[chain] 全链结束 $(date -Is)"
