#!/usr/bin/env bash
# 等待收尾流水线结束 + 基线服务恢复健康后，执行 verify.py 官方路径机制演练
#
# 为什么要等：官方六档压测还在跑，往服务里塞额外请求会污染性能测量。
# 演练本身是"只读"的，但会真实调用服务，所以必须排在压测与基线恢复之后。
set -uo pipefail

W=/root/private_data/workspace/robo
LOG="$W/logs/dryrun_waiter.log"
: > "$LOG"

echo "[$(date -Is)] 等待收尾流水线结束..." | tee -a "$LOG"

# 1) 等 pipeline_after_bench.sh 退出
DEADLINE=$(( $(date +%s) + 21600 ))   # 最多等 6 小时
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  if ! pgrep -f 'pipeline_after_bench.sh' >/dev/null 2>&1; then
    echo "[$(date -Is)] 流水线已结束" | tee -a "$LOG"
    break
  fi
  sleep 30
done

# 2) 等服务在 30000 就绪
echo "[$(date -Is)] 等待服务 127.0.0.1:30000 就绪..." | tee -a "$LOG"
READY=0
for i in $(seq 1 120); do
  if curl -sf --noproxy '*' "http://127.0.0.1:30000/health" >/dev/null 2>&1; then
    READY=1
    echo "[$(date -Is)] 服务就绪（约 $((i*10))s）" | tee -a "$LOG"
    break
  fi
  sleep 10
done

if [ "$READY" != "1" ]; then
  echo "[$(date -Is)] 服务未就绪，放弃演练（不影响主流程）" | tee -a "$LOG"
  exit 0
fi

# 3) 确认压测确实已经结束（再查一次 jsonl 条数，避免误判）
echo "[$(date -Is)] 压测 jsonl 条数：" | tee -a "$LOG"
wc -l "$W"/submission/bench/round*.jsonl 2>/dev/null | tee -a "$LOG"

# 4) 跑演练
echo "[$(date -Is)] 开始 verify.py 官方路径机制演练" | tee -a "$LOG"
bash "$W/tools/verify_path_dryrun.sh" 2>&1 | tee -a "$LOG"
echo "[$(date -Is)] 演练结束" | tee -a "$LOG"
