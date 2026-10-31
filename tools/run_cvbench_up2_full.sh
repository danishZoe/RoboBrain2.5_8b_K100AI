#!/usr/bin/env bash
# 全量 CV-Bench（up2 第二口径），**提高并发以压缩墙钟**
#
# 上一轮用 --concurrency 16 跑 788 题要 12 min；这里用 48（服务 max_num_seqs=64）
# 跑 2638 题，把 L3 档（20~60 min）压到 ~15 min。
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONPATH="$WS/envs/pylibs_core:$WS/third_party/upstream_eval/FlagEvalMM:$WS/scripts:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore
export TMPDIR="$WS/tmp"
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

echo "===== 全量 CV-Bench（up2_png，2638 题，并发 48）$(date '+%F %T') ====="
python3 "$WS/submission/bench_public/run_cv_bench.py" \
  --url http://127.0.0.1:30000/v1/chat/completions \
  --run-id "up2-cvbench-full" \
  --out-dir "$WS/outputs/bench_public/cvbench_up2_full" \
  --image-protocol up2_png --prompt-style flagevalmm \
  --concurrency 48 --no-resume \
  2>&1 | tee -a "$WS/logs/cvbench_up2_full.log" | grep -E '题目数|accuracy_|图像输入协议|失败' || true
echo "===== 结束 $(date '+%F %T') ====="
