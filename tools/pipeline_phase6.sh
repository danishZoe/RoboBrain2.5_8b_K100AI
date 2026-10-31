#!/usr/bin/env bash
# =============================================================================
# pipeline_phase6.sh - 第三段收尾流水线（独占卡的实验最后跑）
#
#   1) 等前两段流水线（pipeline_after_bench / pipeline_final / clean_repro）全部结束
#   2) E1 确认实验：非 eager 重跑 4096/1、4096/8、16384/1、16384/8 四档 + 12 项 Smoke
#                   + 同口径探针逐坐标比对（登记要求「重跑确认」）
#   3) KV cache 容量按 --max-num-batched-tokens 取值采集（补 E2 的机制证据）
#   4) 恢复最终基线服务（eager / 32768 / 30000）
#   5) 把 E1 确认结果与 KV 容量写进 reports/11_tuning_experiments.md
#   6) 提交包自检 + 最终检查
#
# 这些实验都会独占整卡（会 pkill 服务），所以必须排在干净复现之后。
# 日志: /root/private_data/workspace/robo/logs/pipeline_phase6.log
# =============================================================================
set -uo pipefail
W=/root/private_data/workspace/robo
LOG="$W/logs/pipeline_phase6.log"
cd "$W" || exit 1
mkdir -p logs

step() {
  echo ""
  echo "################ $(date -Is)  $* ################"
}

{
  step "1) wait for previous pipelines"
  DEADLINE=$(( $(date +%s) + 36000 ))
  while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    BUSY=0
    for P in pipeline_after_bench.sh pipeline_final.sh clean_repro.sh; do
      if pgrep -f "$P" >/dev/null 2>&1; then BUSY=1; fi
    done
    if [ "$BUSY" = "0" ]; then
      echo "previous pipelines finished at $(date -Is)"; break
    fi
    sleep 30
  done
  # 安全闸：官方压测必须已真正结束
  while pgrep -f 'bash bench.sh' >/dev/null 2>&1 || pgrep -f 'bench_serving' >/dev/null 2>&1; do
    echo "[$(date -Is)] 官方压测仍在运行，等待其结束（不向服务发任何请求）..."
    sleep 30
  done
  N=$(wc -l < "$W/submission/bench/round2.jsonl" 2>/dev/null || echo 0)
  echo "round2 entries = $N"
  if [ "$N" -lt 6 ]; then
    echo "ABORT: round2 只有 $N 条（期望 6），不能继续（会破坏基准数据一致性）"
    exit 1
  fi

  step "2) E1 confirm experiment (non-eager re-run, 4 tiers + smoke + probe)"
  bash "$W/scripts/run_non_eager_confirm.sh" --port 8000 \
        --tiers "4096:1,4096:8,16384:1,16384:8" 2>&1 | tail -120

  step "3) KV cache by max-num-batched-tokens"
  bash "$W/scripts/collect_kv_by_bnt.sh" --values "4096,8192,16384" --port 8000 2>&1 | tail -40

  step "4) restore final baseline service"
  cd "$W/submission" || exit 1
  set +u
  source ./env.sh
  set -u
  setsid nohup bash run.sh > logs/run_stdout_phase6.log 2>&1 &
  for i in $(seq 1 120); do
    if curl -sf --noproxy '*' "http://127.0.0.1:30000/health" >/dev/null 2>&1; then
      echo "final service up after ~$((i*10))s"; break
    fi
    sleep 10
  done
  curl -s --noproxy '*' "http://127.0.0.1:30000/v1/models" | head -c 300; echo
  cd "$W" || exit 1

  step "5) inject E1-confirm + KV into reports/11_tuning_experiments.md"
  python3 tools/inject_tuning_results.py reports/11_tuning_experiments.md 2>&1

  step "5b) inject 验收 §8.2 优化门槛核算 into reports/05_final.md"
  python3 tools/inject_optimization_gate.py reports/05_final.md 2>&1
  python3 tools/inject_final_report.py reports/05_final.md 2>&1

  step "6) record delivery revision (git) + inject into reports/README"
  bash "$W/tools/record_revision.sh" 2>&1 | tail -20
  python3 tools/inject_revision.py 2>&1 | tail -10
  bash "$W/tools/record_revision.sh" --finalize 2>&1 | tail -10

  step "7) submission self-check"
  bash "$W/tools/check_submission.sh" 2>&1 | tail -40

  step "8) final check"
  bash "$W/tools/final_check.sh" 2>&1 | tail -40

  step "PHASE6 PIPELINE DONE"
} >> "$LOG" 2>&1

echo "phase6 pipeline finished, see $LOG"
