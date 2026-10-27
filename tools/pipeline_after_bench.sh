#!/usr/bin/env bash
# =============================================================================
# 压测收尾流水线（脱离会话运行，逐步失败不中断整体）
#   1) 等官方六档压测结束
#   2) 生成 bench/summary.md + summary_detailed.md
#   3) 回填 README 第 7 节的官方格式压测表
#   4) 刷新 env.log / smoke.log / verify_result.json（对齐最终配置）
#   5) 提交包自检
#   6) E1 非 eager A/B（带内存看门狗）
#   7) E2 max-num-batched-tokens A/B
#   8) E3 rocBLAS layer3 trace（诊断）
#   9) 恢复最终基线服务（eager, 32768, 30000）
# 日志: /root/private_data/workspace/robo/logs/pipeline_after_bench.log
# =============================================================================
set -uo pipefail
W=/root/private_data/workspace/robo
LOG="$W/logs/pipeline_after_bench.log"
cd "$W" || exit 1
mkdir -p logs

step() {
  echo ""
  echo "################ $(date -Is)  $* ################"
}

{
  step "1) wait for official bench to finish"
  # 等待条件（两条都要满足）：
  #   a) 压测进程全部退出（bench.sh + bench_serving 客户端）
  #   b) round2.jsonl 已有 6 条（官方六档两轮齐全）
  # 只靠进程退出不够：客户端可能在写 jsonl 之前先退出。
  # 超时保护：DEADLINE 必须大于「六档压测的最坏耗时」；
  #   首版用 4 h，而最后一档（16384/并发 64）单轮就要 ~2.2 h，
  #   六档两轮合计 > 4 h → 会在压测还在跑时提前进入 E1/E2，
  #   而 E2 会杀掉正在被压测的服务器，导致最后一档数据作废。
  #   现改为 10 h，且超时后**直接退出**而不是带着半个结果继续。
  DEADLINE=$(( $(date +%s) + 36000 ))
  BENCH_DONE=0
  while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    if ! pgrep -f 'bash bench.sh' >/dev/null 2>&1 && ! pgrep -f 'bench_serving' >/dev/null 2>&1; then
      N=$(wc -l < "$W/submission/bench/round2.jsonl" 2>/dev/null || echo 0)
      if [ "$N" -ge 6 ]; then
        echo "bench finished at $(date -Is) (round2 entries=$N)"; BENCH_DONE=1; break
      fi
      echo "bench processes gone but round2 has only $N entries; waiting for the writer to flush..."
    fi
    sleep 20
  done
  if [ "$BENCH_DONE" != "1" ]; then
    echo "ABORT: 官方压测未在截止时间内完成（round2 条数=$N）—— 为避免破坏正在进行的测量，本流水线退出。"
    exit 1
  fi

  step "2) summarize"
  bash "$W/tools/summarize_official_bench.sh" 2>&1 | tail -40

  step "3) inject bench summary into README"
  cd "$W/submission" && python3 "$W/tools/inject_bench_summary.py" README.md bench/summary.md 2>&1
  cd "$W" || exit 1

  step "4) refresh env.log / smoke.log / verify_result.json"
  bash "$W/tools/refresh_artifacts.sh" 2>&1 | tail -60

  step "5) submission self-check"
  bash "$W/tools/check_submission.sh" 2>&1 | tail -60

  step "6) E1 non-eager A/B (watchdog 70 GiB)"
  bash "$W/scripts/run_non_eager_ab.sh" --mem-limit-gib 70 --ctx 32768 \
        --port 8000 --tiers "4096:1,4096:8" 2>&1 | tail -80

  step "7) E2 max-num-batched-tokens A/B"
  bash "$W/scripts/run_batched_tokens_ab.sh" --values "4096,8192,16384" \
        --tiers "16384:1,16384:8" --port 8000 2>&1 | tail -80

  step "8) E3 rocBLAS layer3 trace (diagnostic)"
  bash "$W/tools/collect_rocblas_trace.sh" --port 8000 --requests 2 2>&1 | tail -50

  step "9) restore final baseline server (eager, 32768, port 30000)"
  cd "$W/submission" || exit 1
  source ./env.sh
  setsid nohup bash run.sh > logs/run_stdout_final.log 2>&1 &
  for i in $(seq 1 90); do
    if curl -sf --noproxy '*' "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo "final server up after ~$((i*10))s"; break
    fi
    sleep 10
  done
  curl -s --noproxy '*' "http://127.0.0.1:$PORT/v1/models" | head -c 400; echo

  step "PIPELINE DONE"
} >> "$LOG" 2>&1

echo "pipeline finished, see $LOG"
