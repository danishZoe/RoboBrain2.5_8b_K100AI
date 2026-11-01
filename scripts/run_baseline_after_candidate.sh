#!/usr/bin/env bash
# =============================================================================
# 等最优版完整压测结束后，自动补测**同节点** graph 基线的缺失档位
#
# 背景：2026-10-31 节点上只测了 graph 基线的 4096/c1、4096/c8 两档，
#       为了让 16384/c1、16384/c8、4096/c64 三档也有**同节点**对照，
#       在候选压测结束后用 stock 库 + graph 模式补测一轮。
#
# 判定：这是基线补测，不改任何配置；跑完由 build_tuning_comparison.py 出表。
# =============================================================================
set -uo pipefail

WS=/root/private_data/workspace/robo
cd "$WS" || exit 1

LOG=logs/fullbench_baseline_fill_$(date +%Y%m%d_%H%M%S).log
echo "[chain] 等待候选压测结束… (log: $LOG)"

# 1) 等候选驱动进程退出
while pgrep -f 'run_full_bench.sh --config candidate' >/dev/null 2>&1; do
  sleep 30
done
echo "[chain] 候选压测已结束 $(date -Is)"

# 2) 等残留服务退出（候选驱动收尾会停服务）
for i in $(seq 1 30); do
  pgrep -f "serve_vllm_patched.py" >/dev/null 2>&1 || break
  sleep 10
done
if pgrep -f "serve_vllm_patched.py" >/dev/null 2>&1; then
  echo "[chain] 仍有残留服务，终止本次补测（不抢占）"
  exit 1
fi
sleep 10

# 3) 补测同节点基线（stock 库 + graph，单轮）
echo "[chain] 开始基线补测 $(date -Is)"
bash scripts/run_full_bench.sh --config baseline \
  --tiers "16384:1:1,16384:8:1,4096:64:1" --skip-smoke >> "$LOG" 2>&1
RC=$?
echo "[chain] 基线补测结束 rc=$RC $(date -Is)"

# 4) 重新出对比表
python3 scripts/build_tuning_comparison.py --out reports/tuning_comparison.md >/dev/null 2>&1
echo "[chain] 对比表已更新"
