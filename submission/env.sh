#!/usr/bin/env bash
# 模型、镜像、框架、卡型、并行方式与启动参数，其余脚本读这里的变量。
# 本文件由 `source ./env.sh` 使用；注意：/opt/dtk/env.sh 引用了未定义变量，
# 不能在 `set -u` 下 source。

export MODEL_ID="BAAI/RoboBrain2.5-8B-NV"
export MODEL_PATH="/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV"
export SERVED_NAME="robobrain2.5-8b-nv"
export IMAGE="jupyter-sglang5.12-ubuntu22.04-dtk26.04-py3.10-qwen3.6:v1.0"
export FRAMEWORK="vllm"
export CARD="K100-AI"          # K500SM_AI（gfx928）按官方口径计入 K100-AI
export PRECISION="bf16"
export PORT=30000

export TP=1                     # 本 Pod 仅分配到 1 张卡；官方脚手架默认 4
export EP=1
export PP=1
export MAX_LEN=32768            # 官方 env.sh 口径：须容纳 16384 输入 + 2040 视觉 token + 1024 输出
export MEM_FRAC=0.85
export NNODES=1
export NODE_RANK=0
export DIST_ADDR="10.0.0.1:50000"
export LOAD_TIMEOUT=1800

# site-packages 在本容器内只读，patch 改用 sitecustomize 垫片注入；
# 该变量仅为兼容官方 run.sh 的 patch 重放逻辑。
export PATCH_ROOT="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null || echo /usr/local/lib/python3.10/dist-packages)"

# 其余启动参数（报告第 4 节逐条说明）
#   --enforce-eager        跳过 torch.compile / graph capture：容器 cgroup 上限 115 GiB，
#                          非 eager 启动会在 inductor 编译期触顶并被 OOM killer 杀掉（Pod 重建）
#   --no-enable-prefix-caching  压测时避免相同 prompt 的 prefill 成本被缓存抹掉
# 注意：EXTRA_ARGS 以**空格分隔**后原样传给启动命令，因此其中的 JSON 不能带空格
#       （`{"image":8}` 而不是 `{"image": 8}`），否则会被拆成两个参数。
export EXTRA_ARGS="--enforce-eager --no-enable-prefix-caching --max-num-batched-tokens 4096 --max-num-seqs 64 --limit-mm-per-prompt={\"image\":8}"

# 工作目录：所有产物、缓存、临时文件
export WORK_DIR="${WORK_DIR:-/root/private_data/workspace/robo}"
export REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="$WORK_DIR/cache/hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$WORK_DIR/cache"
export TMPDIR="$WORK_DIR/tmp"
# 客户端图像规范化依赖（官方 qwen_vl_utils，隔离安装，不进 site-packages）
export PYTHONPATH="$WORK_DIR/envs/pylibs:${PYTHONPATH:-}"

# 运行时（DTK）必须在每个脚本里显式加载：source /opt/dtk/env.sh
# 注意：/opt/dtk/env.sh 引用了未定义变量，在 `set -u` 下 source 会直接报错退出，
# 因此这里临时关掉 nounset，source 完再按调用方的原状态恢复。
if [ -f /opt/dtk/env.sh ]; then
  _robo_had_u=0
  case "$-" in *u*) _robo_had_u=1 ;; esac
  set +u
  # shellcheck disable=SC1091
  source /opt/dtk/env.sh
  if [ "$_robo_had_u" = "1" ]; then set -u; fi
  unset _robo_had_u
fi
