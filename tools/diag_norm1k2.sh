#!/usr/bin/env bash
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
RUN="$WS/outputs/bench_public/refspatial/upstream_scaled"
export PYTHONPATH="$WS/envs/pylibs_core:$WS/third_party/upstream_eval/FlagEvalMM:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore
set +u; source /opt/dtk/env.sh >/dev/null 2>&1; set -u

echo "===== 直接跑生成的判分子脚本，看完整错误栈 ====="
ls -la "$RUN"/_score_norm1k_*.py 2>/dev/null | head -n 3
( cd "$RUN" && python3 _score_norm1k_Location.py 2>&1 | tail -n 30 )

echo
echo "===== 该子脚本内容 ====="
head -n 20 "$RUN/_score_norm1k_Location.py" 2>/dev/null
