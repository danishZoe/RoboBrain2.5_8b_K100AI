#!/usr/bin/env bash
set -o pipefail
W=/root/private_data/workspace/robo
echo "===== 演练沙箱 verify_evidence.json ====="
head -c 2500 "$W/tmp/dryrun_verify/verify_evidence.json" 2>/dev/null
echo
echo "===== 沙箱里的假基线长什么样 ====="
head -c 1200 "$W/tmp/dryrun_verify/baselines/BAAI__RoboBrain2.5-8B-NV.json" 2>/dev/null
echo
echo "===== 沙箱 data/internal_demo.jsonl (前 2 行) ====="
head -n 2 "$W/tmp/dryrun_verify/data/internal_demo.jsonl" 2>/dev/null | cut -c1-500
echo
echo "===== verify.py 里的基线/子集探测逻辑 ====="
grep -nE "baselines|BASELINE|subset|jsonl|judge|BLOCKED|def main|argparse|add_argument" "$W/submission/verify.py" | head -60
