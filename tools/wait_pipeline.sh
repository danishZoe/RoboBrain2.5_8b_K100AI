#!/usr/bin/env bash
# wait_pipeline.sh - 等待远端流水线进程结束并打印日志尾部
# 用法: bash tools/wait_pipeline.sh <pattern> <tail_lines> [max_minutes]
# 例:   bash tools/wait_pipeline.sh pipeline_after_bench.sh 140 60
set -uo pipefail
W=/root/private_data/workspace/robo
PATTERN="${1:-pipeline_after_bench.sh}"
TAIL="${2:-120}"
MAXMIN="${3:-60}"
cd "$W" || exit 1
DEADLINE=$(( $(date +%s) + MAXMIN * 60 ))
echo "waiting for: $PATTERN (deadline ${MAXMIN}m)"
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  if ! pgrep -f "$PATTERN" >/dev/null 2>&1; then
    echo "DONE: $PATTERN finished at $(date -Is)"
    break
  fi
  sleep 20
done
if pgrep -f "$PATTERN" >/dev/null 2>&1; then
  echo "TIMEOUT: $PATTERN still running after ${MAXMIN}m"
fi
echo "=== tail ${TAIL} of logs/pipeline_after_bench.log ==="
tail -n "$TAIL" logs/pipeline_after_bench.log
