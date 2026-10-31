#!/usr/bin/env bash
# 启动 up2 全量看门狗（硬上限 2026-10-28 17:16:00）
set -uo pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
DL=$(date -d '2026-10-28 17:16:00' +%s)
echo "deadline=$(date -d "@$DL" '+%F %T')  now=$(date '+%F %T')"
nohup bash tools/watch_up2_full.sh "$DL" > logs/watch_up2_full_stdout.log 2>&1 &
sleep 2
ps -eo pid,etime,cmd | grep watch_up2_full | grep -v grep
