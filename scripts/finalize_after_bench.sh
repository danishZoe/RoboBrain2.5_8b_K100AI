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
  # 按**同卡 + 全部服务实例**汇总（单档耗时长时会被拆到多次服务实例，
  # 只看最新目录会漏档）。口径与报告脚本 bench_set.py 完全一致。
  python3 - <<'PY'
import sys
sys.path.insert(0, "scripts")
try:
    import bench_set
except Exception as e:
    print("  bench_set 导入失败:", e)
    raise SystemExit(0)

for cfg, label in (("candidate", "最优版"), ("eager", "首提基线 eager")):
    print(f"=== {label} 各档输出吞吐（tok/s）===")
    src, dirs = bench_set.pick_single(cfg)
    if not dirs:
        print("  （无同卡数据）")
        print("")
        continue
    print(f"  数据来源：{src}")
    for d in dirs:
        print(f"    - {d}")
    rows = bench_set.load_dirs(dirs)
    for key in bench_set.ORDER:
        rs = rows.get(key)
        name = bench_set.TIER_NAME.get(key, str(key))
        if not rs:
            print(f"  {name}: —")
            continue
        n = len(rs)
        tp = sum(float(r.get("output_throughput") or 0) for r in rs) / n
        ttft = sum(float(r.get("mean_ttft_ms") or 0) for r in rs) / n
        tpot = sum(float(r.get("mean_tpot_ms") or 0) for r in rs) / n
        done = sum(int(r.get("completed") or 0) for r in rs)
        fail = sum(int(r.get("failed") or 0) for r in rs)
        print(f"  {name}: tp={tp:.2f} ttft={ttft:.1f}ms tpot={tpot:.2f}ms "
              f"completed={done} failed={fail} rounds={n}")
    print("")
PY
} > "$WS/tmp/DELIVERY_READY.txt" 2>&1

echo ""
echo "[after-bench] 交付就绪标记已写：$WS/tmp/DELIVERY_READY.txt"
cat "$WS/tmp/DELIVERY_READY.txt"
echo "[after-bench] 完成 $(date -Is)"
