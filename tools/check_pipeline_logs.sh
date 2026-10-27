#!/usr/bin/env bash
set -o pipefail
W=/root/private_data/workspace/robo
echo "===== RUNBOOK 备份与新文件 ====="
ls -l "$W"/reports/00_RUNBOOK.md*
echo
echo "===== 流水线 / 演练日志尾部 ====="
for f in "$W"/logs/pipeline_final.log "$W"/logs/pipeline_after_bench.log "$W"/logs/dryrun_waiter.log "$W"/logs/verify_path_dryrun.log "$W"/logs/dryrun_verify.log; do
  echo "== $(basename "$f")"
  tail -n 14 "$f" 2>/dev/null || echo "(缺失)"
  echo
done
echo "===== 关键进程 ====="
ps -eo pid,etime,cmd | grep -E 'bench_serving|pipeline|dryrun|verify' | grep -v grep | cut -c1-110 || echo "(none)"
echo
echo "===== verify_evidence / dryrun 产物 ====="
find "$W/tmp/dryrun_verify" -maxdepth 2 -type f 2>/dev/null | head -20
find "$W" -name 'verify_evidence*.json' -not -path '*/tmp/pypi_probe/*' 2>/dev/null | head
