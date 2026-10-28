#!/usr/bin/env bash
# 快评链：等提示词口径实验结束 -> 启动并发 48 的全量 CV-Bench（up2 口径）
#
# 背景：原 chain_cvbench.sh 第③步用默认并发 16，2638 题预计 ~40 min（L3 档）。
# 用户要求「优先做快的评测」。这里改为 tools/run_cvbench_up2_full.sh（并发 48），
# 预计 ~15 min，把剩余评测墙钟压掉约 25 min。
set -uo pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
LOG="$WS/logs/chain_cvbench_fast.log"

echo "===== fast chain 开始 $(date '+%F %T') =====" | tee -a "$LOG"

# ① 等提示词口径实验结束（最多等 40 min）
echo "--- 等 exp_cvbench_prompt_style 结束 ---" | tee -a "$LOG"
for i in $(seq 1 120); do
  if ! pgrep -f 'exp_cvbench_prompt_style.sh' >/dev/null 2>&1; then
    echo "  提示词实验已结束（等了 $((i*20)) 秒）" | tee -a "$LOG"
    break
  fi
  sleep 20
done

# ② 打印提示词实验结果
echo "" | tee -a "$LOG"
echo "--- 提示词口径实验结果 ---" | tee -a "$LOG"
grep -E 'prompt_style=|accuracy_2d_ade|accuracy_2d_coco|accuracy_2d"|题目数' \
  "$WS/logs/exp_cvbench_prompt_style.log" | tail -30 | tee -a "$LOG"

# ③ 全量 CV-Bench（up2_png，2638 题，并发 48）
echo "" | tee -a "$LOG"
echo "--- 全量 CV-Bench（up2_png，2638 题，并发 48）$(date '+%T') ---" | tee -a "$LOG"
bash "$WS/tools/run_cvbench_up2_full.sh" 2>&1 | tee -a "$LOG"

echo "===== fast chain 结束 $(date '+%F %T') =====" | tee -a "$LOG"
