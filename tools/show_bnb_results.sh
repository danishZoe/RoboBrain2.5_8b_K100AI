#!/usr/bin/env bash
# show_bnb_results.sh - 打印 E2 各档位结果（避免 shell 引号在 PowerShell 下被破坏）
set -uo pipefail
cd /root/private_data/workspace/robo || exit 1
for f in outputs/phase5_ab_bnb/*/*.jsonl; do
  echo "## ${f}"
  cat "${f}"
  echo
done
echo "=== run_config ==="
cat outputs/phase5_ab_bnb/*/run_config.txt 2>/dev/null
