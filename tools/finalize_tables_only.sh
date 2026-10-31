#!/usr/bin/env bash
# 压测进行中可安全执行的收尾（不向服务发任何请求，避免污染正在进行的测量）
set -uo pipefail
W=/root/private_data/workspace/robo
cd "$W/submission" || exit 1

echo "===== 1) summarize ====="
bash "$W/tools/summarize_official_bench.sh" 2>&1 | tail -30

echo
echo "===== 2) inject README §7 ====="
python3 "$W/tools/inject_bench_summary.py" README.md bench/summary.md 2>&1

echo
echo "===== 3) inject report 10 ====="
cd "$W" && python3 tools/inject_report10_bench.py reports/10_phase5_bench.md \
  submission/bench/summary.md submission/bench/summary_detailed.md 2>&1

echo
echo "===== 4) submission self-check ====="
cd "$W/submission" && bash "$W/tools/check_submission.sh" 2>&1 | tail -60
