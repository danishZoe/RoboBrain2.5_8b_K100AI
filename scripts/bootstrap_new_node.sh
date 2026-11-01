#!/usr/bin/env bash
# =============================================================================
# 新容器重建脚本（在 SCNet 新申请的 K100AI 容器上运行）
#
# 背景：2026-11-01 05:08 起原容器 dcu_1026 被平台回收/重建。
#       本脚本在新容器上把工作目录重建到「可压测 + 可收尾」的状态。
#
# 输入（由调用方通过 scp 放到 tmp/ 下）：
#   tmp/publish_final.tar.gz    交付包（代码 + 调优库 6 个差异文件 + 报告 + 脚本）
#   tmp/transfer_scripts.tar.gz 收尾脚本包（fill/build/patch/finalize）
#
# 用法：bash scripts/bootstrap_new_node.sh
# 退出码：0 = 环境就绪且服务可启动
# =============================================================================
set -uo pipefail

WS="${WS:-/root/private_data/workspace/robo}"
cd "$WS" 2>/dev/null || { mkdir -p "$WS"; cd "$WS" || exit 1; }

echo "=================== 0. 基本信息 ==================="
echo "ws=$WS"
echo "date=$(date -Is)"
echo "hostname=$(hostname)"

echo
echo "=================== 1. 环境自检 ==================="
# DTK 运行时（/opt/dtk/env.sh 引用未定义变量，必须先关 nounset）
set +u
# shellcheck disable=SC1091
[ -f /opt/dtk/env.sh ] && source /opt/dtk/env.sh
set -u
export HF_HOME="$WS/cache/hf"
export HUGGINGFACE_HUB_CACHE="$WS/cache/hf/hub"
export TRANSFORMERS_CACHE="$WS/cache/hf/transformers"
export TORCH_HOME="$WS/cache/torch"
export XDG_CACHE_HOME="$WS/cache/xdg"
export TMPDIR="$WS/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$WS/tmp" "$WS/logs" "$WS/outputs"

python3 - <<'PY'
import json, sys
try:
    import torch
    print(f"torch={torch.__version__} hip={torch.version.hip} "
          f"cuda_avail={torch.cuda.is_available()} devices={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  device {i}: {p.name} mem={p.total_memory/2**30:.2f} GiB")
except Exception as e:
    print(f"torch 检查失败：{e}")
try:
    import vllm
    print(f"vllm={vllm.__version__}")
except Exception as e:
    print(f"vllm 检查失败：{e}")
PY

echo "--- 模型只读路径 ---"
MODEL=/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV
if [ -d "$MODEL" ]; then
  echo "模型可见：$(ls "$MODEL" | head -5 | tr '\n' ' ')"
  echo "shards: $(ls "$MODEL"/*.safetensors 2>/dev/null | wc -l)"
else
  echo "ERROR: 模型目录不可见 $MODEL" >&2
  exit 2
fi

echo
echo "=================== 2. 恢复交付包 ==================="
# git 是收尾链的硬依赖（record_revision.sh / export_publish_tarball.sh 都要用）
if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: 容器内没有 git，收尾链无法运行" >&2
  exit 5
fi
echo "git=$(git --version)"
echo "bash=$(bash --version | head -1)"
echo "tar=$(tar --version | head -1)"
echo "python=$(python3 --version 2>&1)"

if [ -f tmp/publish_final.tar.gz ]; then
  tar -xzf tmp/publish_final.tar.gz -C .
  echo "已展开交付包：$(find . -type f -not -path './tmp/*' | wc -l) 个文件"
else
  echo "ERROR: 缺少 tmp/publish_final.tar.gz" >&2
  exit 3
fi
if [ -f tmp/transfer_scripts.tar.gz ]; then
  tar -xzf tmp/transfer_scripts.tar.gz -C .
  echo "已展开收尾脚本包"
fi

echo
echo "=================== 3. 组装 GEMM 调优库 ==================="
bash submission/scripts/make_tuned_rocblas_lib.sh
RC=$?
if [ "$RC" -ne 0 ]; then
  echo "ERROR: 调优库组装失败（哈希不符），拒绝启动" >&2
  exit 4
fi
echo "--- 调优库目录文件数 ---"
ls -1 "$WS/rocblas_tuned/library_gpu4" 2>/dev/null | wc -l

echo
echo "=================== 3b. 卡指纹（换卡后同卡判定的依据）==================="
# shellcheck disable=SC1091
if [ -f "$WS/scripts/card_identity.sh" ]; then
  source "$WS/scripts/card_identity.sh" 2>/dev/null || true
fi
echo "card_id=${CARD_ID:-unknown}"
echo "card_id_source=${CARD_ID_SOURCE:-—}"
echo "card_uid=${CARD_UID:-}"
echo "card_name=${CARD_NAME:-}"
echo "card_gfx=${CARD_GFX:-}"
echo "node_host=${CARD_HOST:-}"
echo "指纹文件：${CARD_ID_FILE:-$WS/outputs/CARD_ID.txt}"

echo
echo "=================== 4. 确认工作目录继承情况 ==================="
if [ -d outputs/fullbench_candidate_20261031_231152 ]; then
  echo "GOOD: 原压测数据仍在（$(ls outputs/fullbench_candidate_20261031_231152/*.jsonl 2>/dev/null | wc -l) 个 jsonl）"
  echo "      只需重跑收尾链即可，无需重新压测"
  HAVE_OLD=1
else
  echo "NOTE: 原压测数据不在（卷未随容器继承）——需要重新压测"
  HAVE_OLD=0
fi
if [ -d outputs/fullbench_baseline_20261101_043223 ]; then
  echo "GOOD: 基线补测数据仍在（$(ls outputs/fullbench_baseline_20261101_043223/*.jsonl 2>/dev/null | wc -l) 个 jsonl）"
fi

echo
echo "=================== 5. 服务可启动性检查 ==================="
echo "（只做 env.sh 解析与参数自检，不启动服务；正式压测由调用方决定何时开跑）"
( cd submission && set +u && source ./env.sh && set -u
  printf 'framework=%s\ncard=%s\nprecision=%s\ntp=%s\nmax_len=%s\nmem_frac=%s\nextra_args=%s\n' \
    "$FRAMEWORK" "$CARD" "$PRECISION" "$TP" "$MAX_LEN" "$MEM_FRAC" "$EXTRA_ARGS"
  printf 'use_tuned=%s\nrocblas_libpath=%s\nattention_backend=%s\n' \
    "${USE_TUNED:-unset}" "${ROCBLAS_TENSILE_LIBPATH:-unset}" "${VLLM_ATTENTION_BACKEND:-unset}" )

echo
echo "=================== 引导完成 ==================="
echo "HAVE_OLD=$HAVE_OLD"
echo "下一步（三套同卡压测，eager → graph 基线 → 最优版）："
echo "  bash scripts/run_same_card_bench.sh --tiers \"4096:1:2,4096:8:2,16384:1:2,16384:8:2,4096:64:2,16384:64:1\" \\"
echo "                              --baseline-tiers \"4096:1:2,4096:8:2,16384:1:2,16384:8:2\""
echo "下一步（收尾）：bash scripts/finalize_submission.sh"
