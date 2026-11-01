#!/usr/bin/env bash
# =============================================================================
# 续跑链 v2：补齐「16384/并发64」两侧（最优版 + 首提基线 eager），各一轮
#
# 为什么要有 v2（2026-11-01）：
#   v1 链（chain_remaining.sh）两个长轮都被 run_full_bench.sh 的安全闸挡下：
#     [guard] 已有服务进程运行，拒绝启动  → rc=4
#   根因是上一轮抽样用的 vLLM 服务实例（PID 5304）在链路启动时**没有退出**，
#   安全闸（检测到 serve_vllm_patched / bench_serving 进程即拒绝启动）直接 exit 4。
#   → v2 在每个长轮**之前**强制清场，并确认端口与进程都已释放才允许开跑；
#     确认不了就中止（不产生「看起来跑了其实没跑」的目录）。
#
# 流程：
#   0) 清场：杀掉残留服务、清掉上一版不完整的 DELIVERY_READY / tarball
#   1) 最优版 16384/并发64 单轮（graph + GEMM 调优库 + ROCM_AITER_FA）
#   2) 清场
#   3) 首提基线 eager 16384/并发64 单轮（--enforce-eager + stock 库 + 默认 attention）
#   4) 清场
#   5) 收尾：finalize_after_bench.sh（回填报告 → 归集 → 自检 → 导出 tarball → DELIVERY_READY）
#
# 预估：单轮约 3.8 h（最优版）/ 4.0~4.4 h（eager），合计约 8 h。
# 任一步失败都记录 rc 并继续往下跑，不中断整条链。
#
# 用法（节点上 detached 启动）：
#   setsid nohup bash scripts/chain_16k64.sh > logs/chain_16k64.log 2>&1 &
# =============================================================================
set -uo pipefail
WS="${WS:-/root/private_data/workspace/robo}"
cd "$WS" || exit 1
mkdir -p logs tmp
exec >> "$WS/logs/chain_16k64.log" 2>&1

echo "============================================================"
echo "=== 续跑链 v2（16K/c64 两侧各一轮）启动 $(date -Is) ==="
echo "============================================================"

# --- 清场：杀掉残留服务，确认进程与端口都释放 --------------------------------
clear_service() {
  local tag="$1"
  echo "[chain] 清场（$tag）$(date -Is)"
  pkill -f "$WS/submission/scripts/serve_vllm_patched.py" 2>/dev/null
  pkill -f "VLLM::EngineCore" 2>/dev/null
  sleep 5
  pkill -9 -f "$WS/submission/scripts/serve_vllm_patched.py" 2>/dev/null
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  local i=0
  while [ $i -lt 60 ]; do
    if ! pgrep -f "serve_vllm_patched.py" >/dev/null 2>&1 \
       && ! pgrep -f "VLLM::EngineCore" >/dev/null 2>&1; then
      break
    fi
    sleep 2
    i=$((i + 2))
  done
  if pgrep -f "serve_vllm_patched.py" >/dev/null 2>&1 \
     || pgrep -f "VLLM::EngineCore" >/dev/null 2>&1; then
    echo "[chain] ERROR：残留服务清不掉（$tag），中止以免产生假数据" >&2
    return 9
  fi
  if curl -sf --noproxy '*' --max-time 3 http://127.0.0.1:30000/health >/dev/null 2>&1; then
    echo "[chain] ERROR：30000 端口仍有服务存活（$tag），中止" >&2
    return 9
  fi
  echo "[chain] 清场完成：进程已退出、30000 端口空闲（$tag）"
  return 0
}

# --- 0) 清场 + 清掉上一版「不完整」的交付标记 ---------------------------------
clear_service "起始" || exit 9
rm -f "$WS/tmp/DELIVERY_READY.txt" "$WS/tmp/publish_final.tar.gz"
# 上一版失败留下的空目录（jsonl=0、无 card_id），改名保留现场但不参与报告聚合
for d in "$WS"/outputs/fullbench_candidate_20261101_124641 \
         "$WS"/outputs/fullbench_eager_20261101_124642; do
  if [ -d "$d" ]; then
    mv "$d" "$WS/outputs/_failed_$(basename "$d")"
    echo "[chain] 失败空目录已改名保留现场：$(basename "$d")"
  fi
done
echo "[chain] 起始状态："
ls -1 "$WS"/outputs/fullbench_* -d 2>/dev/null | sed 's/^/    /'

# --- 1) 最优版 16384/并发64 单轮 ----------------------------------------------
echo ""
echo "[chain] === 最优版 16384/并发64 单轮 开始 $(date -Is) ==="
bash "$WS/scripts/run_same_card_bench.sh" --skip-eager --skip-baseline \
     --tiers "16384:64:1" --port 30000
rc1=$?
echo "[chain] === 最优版 16K/c64 结束 rc=$rc1 $(date -Is) ==="

# --- 2) 清场 ------------------------------------------------------------------
echo ""
clear_service "最优版之后" || { echo "[chain] 清场失败，中止"; exit 9; }

# --- 3) 首提基线 eager 16384/并发64 单轮 --------------------------------------
echo ""
echo "[chain] === 基线 eager 16384/并发64 单轮 开始 $(date -Is) ==="
bash "$WS/scripts/run_same_card_bench.sh" --skip-baseline --skip-candidate \
     --tiers "16384:64:1" --port 30000
rc2=$?
echo "[chain] === 基线 16K/c64 结束 rc=$rc2 $(date -Is) ==="

# --- 4) 清场 ------------------------------------------------------------------
echo ""
clear_service "基线之后" || { echo "[chain] 清场失败，中止"; exit 9; }

# --- 5) 收尾（finalize_after_bench.sh 内部会等 run_same_card_bench.sh 退出）----
echo ""
echo "[chain] === 收尾开始 $(date -Is) ==="
bash "$WS/scripts/finalize_after_bench.sh"
rc3=$?
echo "[chain] === 收尾结束 rc=$rc3 $(date -Is) ==="

echo ""
echo "[chain] 最优版16K/c64 rc=$rc1  基线16K/c64 rc=$rc2  收尾 rc=$rc3"
if [ "$rc1" != "0" ] || [ "$rc2" != "0" ] || [ "$rc3" != "0" ]; then
  echo "[chain] 存在失败项，请人工核查（不自动重跑）"
fi
echo "[chain] 全链结束 $(date -Is)"
