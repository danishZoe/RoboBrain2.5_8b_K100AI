#!/usr/bin/env bash
# 模型、镜像、框架、卡型、并行方式与启动参数，其余脚本读这里的变量。
# 本文件由 `source ./env.sh` 使用；注意：/opt/dtk/env.sh 引用了未定义变量，
# 不能在 `set -u` 下 source。
#
# ---------------------------------------------------------------------------
# 2026-10-31 起：提交配置由「eager 首提基线」切换为「最优版」
#
#   最优版 = graph(非 eager) + GEMM 算子调优库 + ROCM_AITER_FA
#
#   证据：reports/25_tuning_delivery_20261031.md（完整六档压测）
#         reports/23_tuning_progress_20261030.md（E5/E7 采纳记录）
#         reports/22_gemm_tuning.md（GEMM 选优过程与算子级对比）
#
#   回退（回到 eager 首提基线）：
#     USE_TUNED=0 bash run.sh
#   或手工：
#     unset ROCBLAS_TENSILE_LIBPATH VLLM_ATTENTION_BACKEND
# ---------------------------------------------------------------------------

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

# 工作目录：所有产物、缓存、临时文件（须在调优库路径之前定义）
export WORK_DIR="${WORK_DIR:-/root/private_data/workspace/robo}"
export REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- 最优版调优开关 -------------------------------------------------------------
# USE_TUNED=1（默认）：启用 GEMM 调优库 + aiter FA。
# USE_TUNED=0：回到 eager 首提基线（只跑 --enforce-eager、stock 库、默认 attention）。
export USE_TUNED="${USE_TUNED:-1}"

if [ "$USE_TUNED" = "1" ]; then
  # 1) GEMM 算子调优库（rocBLAS Tensile 选优表）。
  #    候选库只存在于工作目录：run.sh 先用 scripts/make_tuned_rocblas_lib.sh
  #    把 stock 库复制到该目录并覆盖 6 个差异文件（4 改写 + 2 新增），
  #    公共 /opt/dtk 库保持只读、不被改动。哈希不符即启动失败。
  export TUNED_ROCBLAS_DIR="$WORK_DIR/rocblas_tuned/library_gpu4"
  export ROCBLAS_TENSILE_LIBPATH="$TUNED_ROCBLAS_DIR"
  # 2) Attention 后端：长上下文 decode 走 aiter FA kernel（10-30 同轮 E7 gate 隔离测得：16384/并发1 +17.9%；本节点未单独隔离该项）。
  #    不要设置 VLLM_ROCM_USE_AITER=1：本镜像下会在 ViT 的
  #    mm_encoder_attention flash op 崩溃（reports/23 §2 E8）。
  export VLLM_ATTENTION_BACKEND=ROCM_AITER_FA
fi

# site-packages 在本容器内只读，patch 改用 sitecustomize 垫片注入；
# 该变量仅为兼容官方 run.sh 的 patch 重放逻辑。
export PATCH_ROOT="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null || echo /usr/local/lib/python3.10/dist-packages)"

# 其余启动参数（报告第 4 节逐条说明）
#   --no-enable-prefix-caching  压测时避免相同 prompt 的 prefill 成本被缓存抹掉
#   --max-num-batched-tokens 4096 / --max-num-seqs 64  首次提交基线取值，
#                             E2 扩大 token 预算的取值未同时满足「吞吐>3% + TTFT下降
#                             + 零失败」三条件，已回退（reports/11 §3）
# 注意：EXTRA_ARGS 以**空格分隔**后原样传给启动命令，因此其中的 JSON 不能带空格
#       （`{"image":8}` 而不是 `{"image": 8}`），否则会被拆成两个参数。
#
# 最优版**不再**带 --enforce-eager：启用 torch.compile + graph capture 后，
# 4096/并发1 输出吞吐 首提 eager 21.74 → graph 基线 22.38 tok/s（+2.9%，2026-10-31 节点同卡实测（换卡后未重跑 graph 基线，属跨卡存档））。
# 单看 graph 的收益有限，主体收益来自 GEMM 调优库与 aiter FA（见 reports/25 §2）。
# 2026-10-27 首次非 eager 启动曾在 inductor 编译期触顶容器 cgroup 上限被 OOM killer
# 杀掉；当前实测启动稳定（ready_after 170~240 s），run.sh 仍保留 70 GiB 内存看门狗
# 作为保护。
#
# USE_TUNED=0 时回到「首提基线」：--enforce-eager + stock rocBLAS 库 + 默认 attention。
export ROBO_BASE_ARGS="--no-enable-prefix-caching --max-num-batched-tokens 4096 --max-num-seqs 64 --limit-mm-per-prompt={\"image\":8}"
if [ "$USE_TUNED" = "1" ]; then
  export EXTRA_ARGS="$ROBO_BASE_ARGS"
else
  export EXTRA_ARGS="--enforce-eager $ROBO_BASE_ARGS"
fi

# 缓存与临时目录
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
