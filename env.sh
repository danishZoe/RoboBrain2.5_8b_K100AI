# RoboBrain2.5-8B-NV on Iluvatar K500/K100 (DTK 26.04)
# usage: source /root/private_data/workspace/robo/env.sh
set -o pipefail

# --- vendor runtime (K100AI/HCU) ---
source /opt/dtk/env.sh
source /opt/hyhal/env.sh

# --- fixed paths ---
export WORK_DIR="/root/private_data/workspace/robo"
export MODEL_PATH="/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV"
# isolated vendored deps (qwen_vl_utils, matplotlib) + our scripts
export PYTHONPATH="${WORK_DIR}/envs/pylibs:${WORK_DIR}/scripts:${PYTHONPATH}"

# --- redirect every cache/temp path into the work dir ---
export HF_HOME="${WORK_DIR}/cache/hf"
export HF_HUB_CACHE="${WORK_DIR}/cache/hf/hub"
export TORCH_HOME="${WORK_DIR}/cache/torch"
export TRITON_CACHE_DIR="${WORK_DIR}/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="${WORK_DIR}/cache/inductor"
export VLLM_CACHE_ROOT="${WORK_DIR}/cache/vllm"
export SGLANG_CACHE_DIR="${WORK_DIR}/cache/sglang"
export XDG_CACHE_HOME="${WORK_DIR}/cache/xdg"
export PIP_CACHE_DIR="${WORK_DIR}/cache/pip"
export TMPDIR="${WORK_DIR}/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

for d in scripts configs samples logs reports outputs envs cache tmp third_party; do mkdir -p "${WORK_DIR}/${d}"; done
for d in "${HF_HOME}" "${TORCH_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${SGLANG_CACHE_DIR}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}" "${TMPDIR}"; do mkdir -p "${d}"; done