#!/usr/bin/env bash
# 串行衔接：等上采样实验结束 -> 提示词口径实验 -> 全量 CV-Bench（up2 口径）
set -uo pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
LOG="$WS/logs/chain_cvbench.log"

echo "===== chain 开始 $(date '+%F %T') =====" | tee -a "$LOG"

# ① 等上采样实验结束
echo "--- 等 exp_cvbench_upscale 结束 ---" | tee -a "$LOG"
for i in $(seq 1 120); do
  if ! pgrep -f 'exp_cvbench_upscale.sh' >/dev/null 2>&1; then
    echo "  上采样实验已结束（等了 $((i*20)) 秒）" | tee -a "$LOG"
    break
  fi
  sleep 20
done

echo "--- 上采样实验结果 ---" | tee -a "$LOG"
grep -E 'protocol=|accuracy_2d_ade|accuracy_2d_coco|accuracy_2d"' "$WS/logs/exp_cvbench_upscale.log" | tail -40 | tee -a "$LOG"

# ② 提示词口径实验
echo "" | tee -a "$LOG"
echo "--- 提示词口径实验 ---" | tee -a "$LOG"
bash "$WS/tools/exp_cvbench_prompt_style.sh" >> "$LOG" 2>&1

# ③ 全量 CV-Bench（up2 口径，2638 题）
echo "" | tee -a "$LOG"
echo "--- 全量 CV-Bench（up2_png，2638 题）$(date '+%T') ---" | tee -a "$LOG"
export PYTHONPATH="$WS/envs/pylibs_core:$WS/third_party/upstream_eval/FlagEvalMM:$WS/scripts:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore
export TMPDIR="$WS/tmp"
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u
python3 "$WS/submission/bench_public/run_cv_bench.py" \
  --url http://127.0.0.1:30000/v1/chat/completions \
  --run-id "up2-cvbench-full" \
  --out-dir "$WS/outputs/bench_public/cvbench_up2_full" \
  --image-protocol up2_png --prompt-style flagevalmm --no-resume \
  2>&1 | tee -a "$LOG" | grep -E '题目数|accuracy_|图像输入协议|失败' || true

echo "===== chain 结束 $(date '+%F %T') =====" | tee -a "$LOG"
