#!/usr/bin/env bash
# =============================================================================
# 组装「最优版」rocBLAS 算子调优库到工作目录
#
# 背景（reports/22_gemm_tuning.md、reports/23_tuning_progress_20261030.md）：
#   10-30 用厂商 auto_select_tools 在本机实测 GEMM shape 上做了选优，
#   产出与 stock 不同的 Tensile 选优表。只在 ECO 硬件类型上生效
#   （hardware_type=ECO，card0 → library_gpu4）。
#
# 做法：
#   1) 把 stock 库 /opt/dtk/lib/rocblas/library_gpu4 **复制**到工作目录
#      （只读公共环境不动，候选库只存在于 $WORK_DIR 内）
#   2) 用仓库内 rocblas_tuned/library_gpu4/ 的 6 个差异文件覆盖
#      （4 个被改写 + 2 个新增；其余文件与 stock 逐字节相同）
#   3) 校验 SHA256；不匹配直接失败退出，不允许带病启动
#
# 幂等：目标目录已存在且 6 个文件哈希一致时跳过复制。
#
# 用法：bash scripts/make_tuned_rocblas_lib.sh
# 产物：$WORK_DIR/rocblas_tuned/library_gpu4
# 回退：不调用本脚本、不设 ROCBLAS_TENSILE_LIBPATH 即回到 stock 库。
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${WORK_DIR:-/root/private_data/workspace/robo}"
STOCK_DIR="${STOCK_DIR:-/opt/dtk/lib/rocblas/library_gpu4}"
OVERLAY_DIR="$REPO_DIR/rocblas_tuned/library_gpu4"
TARGET_DIR="$WORK_DIR/rocblas_tuned/library_gpu4"

# 差异文件清单：文件名 SHA256
# 唯一真源 = rocblas_tuned/library_gpu4/DIFF_FILES.sha256
# （早期本脚本与 scripts/start_aiterfa_service.sh 各抄一份清单，已漂移过 1 个字符，
#   导致「组装自检通过但启动校验失败」，最优版服务起不来）
MANIFEST_FILE="$REPO_DIR/rocblas_tuned/library_gpu4/DIFF_FILES.sha256"
if [ -f "$MANIFEST_FILE" ]; then
  MANIFEST="$(grep -vE '^[[:space:]]*(#|$)' "$MANIFEST_FILE" || true)"
else
  echo "[tuned-lib] ERROR: 缺少差异文件清单：$MANIFEST_FILE" >&2
  exit 1
fi
if [ -z "$MANIFEST" ]; then
  echo "[tuned-lib] ERROR: 差异文件清单为空：$MANIFEST_FILE" >&2
  exit 1
fi

echo "[tuned-lib] stock=$STOCK_DIR"
echo "[tuned-lib] overlay=$OVERLAY_DIR"
echo "[tuned-lib] target=$TARGET_DIR"

if [ ! -d "$STOCK_DIR" ]; then
  echo "[tuned-lib] ERROR: stock 库不存在：$STOCK_DIR" >&2
  exit 1
fi
if [ ! -d "$OVERLAY_DIR" ]; then
  echo "[tuned-lib] ERROR: 仓库内差异文件缺失：$OVERLAY_DIR" >&2
  exit 1
fi

# --- 幂等检查 -----------------------------------------------------------------
need_build=1
if [ -d "$TARGET_DIR" ]; then
  need_build=0
  while read -r fname fsha; do
    [ -z "${fname:-}" ] && continue
    if [ ! -f "$TARGET_DIR/$fname" ]; then need_build=1; break; fi
    cur=$(sha256sum "$TARGET_DIR/$fname" | cut -d' ' -f1)
    if [ "$cur" != "$fsha" ]; then need_build=1; break; fi
  done <<< "$MANIFEST"
fi

if [ "$need_build" = "0" ]; then
  echo "[tuned-lib] 目标目录已是最新，跳过构建"
else
  echo "[tuned-lib] 构建候选库：复制 stock + 覆盖差异文件"
  mkdir -p "$(dirname "$TARGET_DIR")"
  rm -rf "$TARGET_DIR.tmp"
  mkdir -p "$TARGET_DIR.tmp"
  cp -a "$STOCK_DIR"/. "$TARGET_DIR.tmp"/
  while read -r fname fsha; do
    [ -z "${fname:-}" ] && continue
    cp -f "$OVERLAY_DIR/$fname" "$TARGET_DIR.tmp/$fname"
  done <<< "$MANIFEST"
  rm -rf "$TARGET_DIR"
  mv "$TARGET_DIR.tmp" "$TARGET_DIR"
fi

# --- 校验 ---------------------------------------------------------------------
rc=0
while read -r fname fsha; do
  [ -z "${fname:-}" ] && continue
  if [ ! -f "$TARGET_DIR/$fname" ]; then
    echo "[tuned-lib] ERROR: 缺失 $fname" >&2
    rc=1
    continue
  fi
  cur=$(sha256sum "$TARGET_DIR/$fname" | cut -d' ' -f1)
  if [ "$cur" != "$fsha" ]; then
    echo "[tuned-lib] ERROR: 哈希不符 $fname 期望=$fsha 实际=$cur" >&2
    rc=1
  else
    echo "[tuned-lib] OK  $fname  $cur"
  fi
done <<< "$MANIFEST"

# stock 原库保持只读、未被改动
if [ -f "$STOCK_DIR/GemmGenerality_AI_Eco.dat" ]; then
  stock_sha=$(sha256sum "$STOCK_DIR/GemmGenerality_AI_Eco.dat" | cut -d' ' -f1)
  echo "[tuned-lib] stock AI DAT 未改动：$stock_sha"
fi

if [ "$rc" != "0" ]; then
  echo "[tuned-lib] 校验失败" >&2
  exit 1
fi
echo "[tuned-lib] 就绪：$TARGET_DIR"
echo "[tuned-lib] 请在启动服务前 export ROCBLAS_TENSILE_LIBPATH=$TARGET_DIR"
