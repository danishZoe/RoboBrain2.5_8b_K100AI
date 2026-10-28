#!/usr/bin/env bash
# 串行化：等 eager 基线收尾（finish_accuracy.sh）结束后，自动跑 E1 候选（非 eager）精度复测。
# 目的：避免两条评测链同时抢 CPU / 抢服务端口。
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1

BASE_PID="${1:-}"
echo "[chain] 等待基线收尾 pid=$BASE_PID ..."
if [ -n "$BASE_PID" ]; then
  while kill -0 "$BASE_PID" 2>/dev/null; do
    sleep 20
  done
fi
echo "[chain] 基线收尾结束于 $(date '+%F %T')，开始候选复测"
bash "$WS/tools/run_accuracy_candidate.sh"
echo "[chain] 候选复测结束于 $(date '+%F %T')"
