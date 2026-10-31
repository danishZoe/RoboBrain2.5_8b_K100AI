#!/usr/bin/env bash
# 诊断：3D 官方判分耗时构成 + 机器并行能力
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1

echo "===== CPU / 内存 ====="
echo "nproc      = $(nproc)"
echo "loadavg    = $(cat /proc/loadavg)"
free -g | head -2
echo "cpuinfo cores = $(grep -c ^processor /proc/cpuinfo)"

echo
echo "===== 当前 3D 判分进程 ====="
ps -o pid,etime,time,%cpu,rss,cmd -C python3 2>/dev/null | grep -E 'summarize_acc|PID' || \
  ps -ef | grep summarize_acc | grep -v grep

echo
echo "===== 官方参照打分的产物与时间戳（推断单变体 3D 判分耗时） ====="
find outputs/bench_public/tracespatial/_official_reference -name '*.json' -o -name '*.jsonl' 2>/dev/null \
  | head -20
find outputs/bench_public/tracespatial/_official_reference -type f -printf '%T+  %p\n' 2>/dev/null | sort | head -20

echo
echo "===== 本次 3D 判分进程已打开的文件（看它卡在哪一步） ====="
PID=$(pgrep -f 'summarize_acc.py .*--task_name 3D' | head -1)
if [ -n "$PID" ]; then
  echo "pid=$PID"
  ls -l /proc/"$PID"/fd 2>/dev/null | tail -15
  echo "--- wchan/stat ---"
  cat /proc/"$PID"/wchan 2>/dev/null; echo
  awk '{print "utime="$14" stime="$15}' /proc/"$PID"/stat
else
  echo "没有正在跑的 3D 判分进程"
fi
