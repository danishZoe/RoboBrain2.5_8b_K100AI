#!/usr/bin/env bash
# 统计各日志里的时间锚点
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
for f in \
  logs/public_bench_baseline-eager/cvbench.log \
  logs/public_bench_baseline-eager/refspatial.log \
  logs/public_bench_baseline-eager/tracespatial.log \
  logs/public_bench_baseline-eager/tracespatial_infer.log \
  logs/public_bench_baseline-eager/tracespatial_score.log \
  logs/public_bench_noneager-candidate/cvbench.log \
  logs/public_bench_noneager-candidate/refspatial.log \
  logs/public_bench_noneager-candidate/tracespatial_infer.log \
  logs/public_bench_noneager-candidate/tracespatial_score.log \
  logs/public_bench_noneager-candidate/noisecheck_refspatial.log \
  logs/public_bench_baseline-eager/cvbench_protocol.log \
  logs/cvbench_upscale.log \
  logs/cvbench_prompt.log \
  logs/exp_cvbench_upscale.log \
  logs/exp_cvbench_protocol.log \
  logs/ts_score_3d_96.log \
  logs/verify_3d_merge.log
do
  [ -f "$f" ] || { echo "--- $f : MISSING"; continue; }
  n_time=$(grep -c '用时' "$f" 2>/dev/null || echo 0)
  n_eng=$(grep -c '\[engine\]' "$f" 2>/dev/null || echo 0)
  n_proto=$(grep -cE '^--- protocol=|^\[cvbench\] 图像输入协议' "$f" 2>/dev/null || echo 0)
  first=$(head -1 "$f" | cut -c1-60)
  last_use=$(grep '用时' "$f" 2>/dev/null | tail -3 | cut -c1-110)
  echo "--- $f | mtime=$(date -r "$f" '+%F %T') | 用时x$n_time | engine x$n_eng | proto x$n_proto"
  echo "    首行: $first"
  [ -n "$last_use" ] && echo "    用时行: $last_use"
done
