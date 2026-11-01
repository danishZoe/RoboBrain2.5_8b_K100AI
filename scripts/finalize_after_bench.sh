#!/usr/bin/env bash
# =============================================================================
# 节点侧「等压测结束 → 自动收尾」守护脚本
#
# 为什么放在节点上：压测 + 收尾必须**完全脱离 agent 会话**。
#   agent / SSH 通道 / 本地机器中断，都不能影响远端流程；
#   远端自己把数据、报告、MANIFEST、自检、交付包全部产出并落盘，
#   剩下只等一次「取回 + 推送 GitLab」。
#
# 流程：
#   1) 等 run_same_card_bench.sh 退出（每 60 s 探一次，最长 18 h）
#   2) 自动执行 scripts/finalize_submission.sh
#        ├ 回填报告（reports/25 / submission/README / reports/05_final）
#        ├ 归集三套同卡原始数据到 submission/bench/
#        ├ 重建 submission/MANIFEST.sha256
#        ├ 跑 tools/check_submission.sh（FAIL>0 则中止导出）
#        └ 导出 tmp/publish_final.tar.gz
#   3) 写 tmp/DELIVERY_READY.txt：交付包路径、SHA256、三套运行目录、关键指标
#
# 用法（节点上 detached 启动）：
#   setsid nohup bash scripts/finalize_after_bench.sh > logs/finalize_after_bench.log 2>&1 &
# =============================================================================
set -uo pipefail

WS="${WS:-/root/private_data/workspace/robo}"
cd "$WS" || exit 1
mkdir -p logs tmp
export ROBO_WS="$WS"

LOG="$WS/logs/finalize_after_bench.log"
exec >> "$LOG" 2>&1

echo "=== finalize_after_bench 启动 $(date -Is) ==="
echo "ws=$WS"

# --- 1) 等压测结束 -------------------------------------------------------------
WAIT_MAX=$((18 * 3600))
waited=0
while [ "$waited" -lt "$WAIT_MAX" ]; do
  if ! pgrep -f "run_same_card_bench.sh" >/dev/null 2>&1; then
    echo "[after-bench] 压测进程已退出（等待 $waited s）"
    break
  fi
  sleep 60
  waited=$((waited + 60))
  if [ $((waited % 1800)) -eq 0 ]; then
    echo "[after-bench] 仍在等待（$((waited / 60)) min）$(date -Is)"
    tail -2 logs/same_card_bench_main.log 2>/dev/null | sed 's/^/    /'
  fi
done
if [ "$waited" -ge "$WAIT_MAX" ]; then
  echo "[after-bench] 等待超时，保留现场，不执行收尾"
  exit 2
fi

# --- 2) 收尾 ------------------------------------------------------------------
echo ""
echo "[after-bench] 开始 finalize_submission.sh $(date -Is)"
bash "$WS/scripts/finalize_submission.sh"
RC=$?
echo "[after-bench] finalize rc=$RC $(date -Is)"
if [ "$RC" != "0" ]; then
  echo "[after-bench] 收尾失败，保留现场；请人工检查 $WS/logs/finalize_after_bench.log"
  exit "$RC"
fi

# --- 3) 交付就绪标记 -----------------------------------------------------------
TARBALL="$WS/tmp/publish_final.tar.gz"
{
  echo "DELIVERY_READY"
  echo "ready_at=$(date -Is)"
  echo "tarball=$TARBALL"
  echo "tarball_sha256=$(sha256sum "$TARBALL" 2>/dev/null | cut -d' ' -f1)"
  echo "tarball_bytes=$(stat -c %s "$TARBALL" 2>/dev/null)"
  echo ""
  echo "=== 三套同卡运行目录 ==="
  for d in outputs/fullbench_eager_* outputs/fullbench_baseline_* outputs/fullbench_graph_* outputs/fullbench_candidate_*; do
    [ -d "$d" ] || continue
    cid=$(grep -m1 '^card_id=' "$d/run_config.txt" 2>/dev/null | cut -d= -f2-)
    echo "$d  jsonl=$(ls -1 "$d"/*.jsonl 2>/dev/null | wc -l)  card_id=${cid:-无}"
  done
  echo ""
  echo "=== 最优版各档输出吞吐（tok/s）==="
  CAND=$(ls -d outputs/fullbench_candidate_* 2>/dev/null | sort -r | head -1)
  for f in "$CAND"/round*.jsonl; do
    [ -f "$f" ] || continue
    python3 - "$f" <<'PY'
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as fh:
        r = json.loads(fh.readline())
    print(f"  {sys.argv[1].split('/')[-1]}: tp={r.get('output_throughput')} "
          f"ttft={r.get('mean_ttft_ms')} tpot={r.get('mean_tpot_ms')} "
          f"completed={r.get('completed')} failed={r.get('failed')}")
except Exception as e:
    print("  读取失败:", sys.argv[1], e)
PY
  done
  echo ""
  echo "=== 首提基线 eager 各档输出吞吐（tok/s）==="
  EAG=$(ls -d outputs/fullbench_eager_* 2>/dev/null | sort -r | head -1)
  for f in "$EAG"/round*.jsonl; do
    [ -f "$f" ] || continue
    python3 - "$f" <<'PY'
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as fh:
        r = json.loads(fh.readline())
    print(f"  {sys.argv[1].split('/')[-1]}: tp={r.get('output_throughput')} "
          f"ttft={r.get('mean_ttft_ms')} tpot={r.get('mean_tpot_ms')} "
          f"completed={r.get('completed')} failed={r.get('failed')}")
except Exception as e:
    print("  读取失败:", sys.argv[1], e)
PY
  done
} > "$WS/tmp/DELIVERY_READY.txt" 2>&1

echo ""
echo "[after-bench] 交付就绪标记已写：$WS/tmp/DELIVERY_READY.txt"
cat "$WS/tmp/DELIVERY_READY.txt"
echo "[after-bench] 完成 $(date -Is)"
