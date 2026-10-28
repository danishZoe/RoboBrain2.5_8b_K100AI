#!/usr/bin/env bash
# 用官方 summarize_acc.py 给上游自带参考预测打分：
#   RoboTracer_Intrinsics_Depth（官方 RoboTracer 模型）/ Gemini3Pro
# 目的：(1) 端到端验证官方判分链路在本节点可用；(2) 得到同一数据集的官方参照值
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
RT="$WS/third_party/upstream_eval/RoboTracer/Evaluation"
DATA="$WS/third_party/benchmarks/TraceSpatial-Bench"
RUN="$WS/outputs/bench_public/tracespatial/_official_reference"
mkdir -p "$RUN"

# 复现官方脚本的工作目录约定：./TraceSpatial-Bench 与 ./outputs
[ -e "$RUN/TraceSpatial-Bench" ] || ln -s "$DATA" "$RUN/TraceSpatial-Bench"
[ -e "$RUN/outputs" ] || ln -s "$RT/outputs" "$RUN/outputs"

export PYTHONPATH="$WS/envs/pylibs_o3d:$WS/envs/pylibs:$WS/envs/pylibs_core:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

for M in RoboTracer_Intrinsics_Depth Gemini3Pro; do
  for T in 2D 3D; do
    echo "########## model=$M task=$T ##########"
    ( cd "$RUN" && python3 "$RT/summarize_acc.py" --model_name "$M" --task_name "$T" 2>&1 | \
      grep -vE '^\s*$|it/s\]|Evaluating \(' | tail -n 22 )
    echo
  done
done
