#!/usr/bin/env bash
set -o pipefail
W=/root/private_data/workspace/robo

echo "===== E1/E2/E3/E4 结论（11_tuning_experiments.md 的结论段） ====="
grep -nE '^\|.*\|' "$W/reports/11_tuning_experiments.md" | head -24
echo "--- 结论行 ---"
grep -nE '结论|判定|收益|回退|无收益' "$W/reports/11_tuning_experiments.md" | head -24

echo
echo "===== pipeline_phase6 最新状态 ====="
tail -n 25 "$W/logs/pipeline_phase6.log" 2>/dev/null
ls -lt "$W/logs/" | head -12

echo
echo "===== 当前服务与 GPU ====="
ps -eo pid,etime,cmd | grep serve_vllm_patched | grep -v grep | cut -c1-120
command -v hy-smi >/dev/null && hy-smi 2>/dev/null | head -18 || echo "(hy-smi 不可用)"

echo
echo "===== clean_repro server_start.log ====="
cat "$W/logs/clean_repro/server_start.log" 2>/dev/null | tail -20
