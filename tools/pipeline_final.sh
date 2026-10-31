#!/usr/bin/env bash
# =============================================================================
# 收尾流水线（第二阶段）—— 排在 pipeline_after_bench.sh 之后自动执行
#
#   1) 等第一阶段流水线结束（它负责六档汇总、回填、E1/E2/E3、恢复基线服务）
#   2) 等基线服务（127.0.0.1:30000）健康
#   3) §3.8 对照实验：随机 token 提示词 vs 自然散文提示词的输出 token 计量
#   4) 把 §3.8 结果自动写进 reports/10_phase5_bench.md
#   5) 把 E1/E2/E3 的实测结果自动写进 reports/11_tuning_experiments.md
#   6) 重跑官方汇总并再次回填 README §7 与报告 10 的表格（幂等）
#   7) 刷新 env.log / smoke.log / verify_result.json
#   8) 提交包自检 + 最终检查
#
# 日志: /root/private_data/workspace/robo/logs/pipeline_final.log
# =============================================================================
set -uo pipefail
W=/root/private_data/workspace/robo
LOG="$W/logs/pipeline_final.log"
cd "$W" || exit 1
mkdir -p logs outputs/phase5_token_accounting

step() {
  echo ""
  echo "################ $(date -Is)  $* ################"
}

{
  step "1) wait for pipeline_after_bench.sh to finish"
  DEADLINE=$(( $(date +%s) + 36000 ))
  while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    if ! pgrep -f 'pipeline_after_bench.sh' >/dev/null 2>&1; then
      echo "pipeline_after_bench.sh finished at $(date -Is)"; break
    fi
    sleep 30
  done

  # 安全闸：官方压测必须已经真正跑完（进程退出 + round2 六条齐全），
  # 否则本阶段的实验请求会污染正在进行的测量。
  while pgrep -f 'bash bench.sh' >/dev/null 2>&1 || pgrep -f 'bench_serving' >/dev/null 2>&1; do
    echo "[$(date -Is)] 官方压测仍在运行，等待其结束（不向服务发任何请求）..."
    sleep 30
  done
  N=$(wc -l < "$W/submission/bench/round2.jsonl" 2>/dev/null || echo 0)
  echo "round2 entries = $N"
  if [ "$N" -lt 6 ]; then
    echo "WARNING: round2 只有 $N 条（期望 6），§3.8 实验仍会执行但汇总表可能不全"
  fi

  step "2) wait for baseline service on 127.0.0.1:30000"
  READY=0
  for i in $(seq 1 180); do
    if curl -sf --noproxy '*' "http://127.0.0.1:30000/health" >/dev/null 2>&1; then
      READY=1; echo "service ready after ~$((i*10))s"; break
    fi
    sleep 10
  done
  if [ "$READY" != "1" ]; then
    echo "WARNING: baseline service not up; §3.8 experiment will be skipped"
  fi

  # 运行环境：DTK + 隔离依赖
  export WORK_DIR="$W"
  export MODEL_PATH="/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV"
  export HF_HOME="$W/cache/hf"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export PYTHONPATH="$W/envs/pylibs:${PYTHONPATH:-}"
  _had_u=0; case "$-" in *u*) _had_u=1 ;; esac
  set +u
  # shellcheck disable=SC1091
  source /opt/dtk/env.sh
  if [ "$_had_u" = "1" ]; then set -u; fi
  unset _had_u

  step "3) token accounting controlled experiment (gibberish vs natural prompt)"
  if [ "$READY" = "1" ]; then
    python3 scripts/verify_output_token_accounting.py \
      --url http://127.0.0.1:30000 --model robobrain2.5-8b-nv \
      --tokenizer "$MODEL_PATH" --input-tokens 16384 --max-tokens 1024 \
      --out outputs/phase5_token_accounting/result.json 2>&1 | tail -40
  else
    echo "skipped: service not available"
  fi

  step "4) inject §3.8 into reports/10_phase5_bench.md"
  python3 tools/inject_token_accounting.py outputs/phase5_token_accounting/result.json \
    reports/10_phase5_bench.md 2>&1

  step "5) inject E1/E2/E3 into reports/11_tuning_experiments.md"
  python3 tools/inject_tuning_results.py reports/11_tuning_experiments.md 2>&1

  step "6) re-summarize official bench and re-inject tables"
  bash tools/summarize_official_bench.sh 2>&1 | tail -30
  ( cd submission && python3 ../tools/inject_bench_summary.py README.md bench/summary.md 2>&1 )
  python3 tools/inject_report10_bench.py reports/10_phase5_bench.md \
    submission/bench/summary.md submission/bench/summary_detailed.md 2>&1

  step "7) refresh env.log / smoke.log / verify_result.json"
  bash tools/refresh_artifacts.sh 2>&1 | tail -60

  step "8) submission self-check"
  bash tools/check_submission.sh 2>&1 | tail -40

  step "9) final check"
  bash tools/final_check.sh 2>&1 | tail -40

  step "FINAL PIPELINE DONE"
} >> "$LOG" 2>&1

echo "final pipeline finished, see $LOG"
