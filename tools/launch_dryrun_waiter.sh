#!/usr/bin/env bash
# 启动 verify 路径演练的等待器（脱离会话运行）
cd /root/private_data/workspace/robo || exit 1
mkdir -p logs
setsid nohup bash tools/run_dryrun_after_pipeline.sh > logs/dryrun_waiter_launch.log 2>&1 &
sleep 3
echo "--- 等待器进程 ---"
ps -eo pid,etime,args | grep 'run_dryrun_after_pipeline' | grep -v grep
echo "--- launcher log ---"
cat logs/dryrun_waiter_launch.log 2>/dev/null | head -5
