#!/usr/bin/env bash
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
echo "=== 3D 判分相关文件时间线 ==="
for f in logs/ts_score_3d.log logs/ts_score_2d.log logs/probe_3d_cost_mem.log \
         logs/probe_3d_fixed.log logs/diag_3d_scoring_cost.log logs/bisect_3d_slow.log \
         logs/ts_score_3d_96.log logs/verify_3d_merge.log logs/exp_cvbench_upscale.log \
         logs/cvbench_upscale.log; do
  [ -f "$f" ] && echo "$(date -r "$f" '+%F %T')  $f" || echo "MISSING  $f"
done
echo
echo "=== ts_score_3d.log 里是否记录过停止/超时 ==="
grep -nE '中止|停止|kill|timeout|超时|取消' logs/ts_score_3d.log | head -10
echo
echo "=== 相关产物目录 mtime ==="
for d in outputs/bench_public/tracespatial outputs/bench_public_3d96 outputs/bench_public_verify; do
  [ -d "$d" ] && echo "$(date -r "$d" '+%F %T')  $d"
done
