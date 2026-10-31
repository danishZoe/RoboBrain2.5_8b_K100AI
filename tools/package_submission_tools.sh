#!/usr/bin/env bash
# 把 submission/README.md 引用的 tools 脚本镜像进提交包，并重建 MANIFEST.sha256
set -uo pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1

TOOLS=(
  start_bench_service.sh
  run_ts_score_2d.sh
  run_ts_score_3d.sh
  run_ts_score_3d_96.sh
  verify_3d_merge.sh
  remerge_ts_3d.py
  exp_cvbench_upscale.sh
  exp_cvbench_prompt_style.sh
  paired_sign_test.py
  audit_prompt_format.py
  build_eval_timing.py
  watch_up2_full.sh
  launch_watch_up2.sh
  chain_cvbench_fast.sh
  test_verify_denominator.py
  analyze_server_throughput.py
)

mkdir -p submission/tools
miss=0
for f in "${TOOLS[@]}"; do
  if [ -f "tools/$f" ]; then
    cp -f "tools/$f" "submission/tools/$f"
  else
    echo "MISS: tools/$f"
    miss=$((miss+1))
  fi
done
echo "镜像完成，缺失 $miss 个"
ls -1 submission/tools

# 重建 MANIFEST.sha256（排除 __pycache__ 与 MANIFEST 自身）
cd submission || exit 1
find . -type f ! -path '*/__pycache__/*' ! -name 'MANIFEST.sha256' ! -name '*.pyc' \
  | sort | xargs sha256sum > MANIFEST.sha256
echo
echo "MANIFEST 条目数: $(wc -l < MANIFEST.sha256)"
head -3 MANIFEST.sha256
echo "..."
grep -c 'tools/' MANIFEST.sha256 | sed 's/^/tools 条目: /'
