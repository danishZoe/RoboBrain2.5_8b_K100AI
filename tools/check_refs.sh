#!/usr/bin/env bash
# 检查提交包内部引用一致性：README/run.sh/smoke.sh/verify.py/bench.sh 里提到的脚本是否都存在
set -uo pipefail
cd /root/private_data/workspace/robo/submission
echo "===== referenced scripts ====="
{
  grep -ohE '(scripts/[A-Za-z0-9_./-]+\.(py|sh))' README.md run.sh smoke.sh verify.py bench.sh env_check.sh 2>/dev/null
  grep -ohE '\b(serve_vllm_patched|bench_serving_vllm_image|image_canonical|answer_adapter|make_data_url|robo_infer|smoke_service|probe_http_request|bench_service|summarize_bench)\.(py|sh)' \
    README.md run.sh smoke.sh verify.py bench.sh 2>/dev/null | sed 's#^#scripts/#'
} | sort -u | while read -r f; do
  if [ -f "$f" ]; then echo "  OK      $f"; else echo "  MISSING $f"; fi
done
echo
echo "===== submission/scripts/ 实际内容 ====="
ls -1 scripts/ 2>/dev/null | sed 's/^/  /'
echo
echo "===== patches ====="
ls -1 patches/ 2>/dev/null | sed 's/^/  /'
echo
echo "===== samples ====="
find samples -type f 2>/dev/null | sed 's/^/  /' | head -20
echo
echo "===== envs ====="
find envs -type f 2>/dev/null | sed 's/^/  /' | head -10
