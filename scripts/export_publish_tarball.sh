#!/usr/bin/env bash
# =============================================================================
# 导出「可发布文件清单」压缩包：用于把远端工作目录的交付物同步到代码托管平台
#
# 规则：
#   * 只包含 git 已跟踪的文件 + 未跟踪但未被 .gitignore 忽略的文件
#   * 排除运行产物（outputs/ logs/ tmp/ cache/ envs/）、大体积第三方镜像、
#     崩溃转储（core.*）、运行时组装的调优库目录（/rocblas_tuned/）
#   * 显式排除厂商 GEMM 调优现场目录（tools/auto_select_tools*，8 GB+）：
#     .gitignore 对「已被 git 跟踪」的文件无效，故此处按目录前缀再兜底一次，
#     避免交付包被 8 GB 非交付物撑爆
#   * 不修改工作区，不提交任何东西（提交由调用方决定）
#
# 用法：bash scripts/export_publish_tarball.sh [输出路径]
#        默认输出 tmp/publish-$(date +%Y%m%d_%H%M%S).tar.gz
#        DRY_RUN=1 时只统计清单、不打包
# =============================================================================
set -euo pipefail

WS="${WS:-/root/private_data/workspace/robo}"
cd "$WS"

OUT="${1:-tmp/publish-$(date +%Y%m%d_%H%M%S).tar.gz}"
DRY_RUN="${DRY_RUN:-0}"
mkdir -p "$(dirname "$OUT")"

# 显式排除前缀（按目录边界匹配，避免误伤 tools/auto_select_tools_helper 之类）
EXCLUDE_PREFIXES=(
  "tools/auto_select_tools"
)

is_excluded() {
  local path="$1" p
  for p in "${EXCLUDE_PREFIXES[@]}"; do
    case "$path" in
      "$p"/*) return 0 ;;
    esac
  done
  return 1
}

RAW="$(mktemp)"
LIST="$(mktemp)"
trap 'rm -f "$RAW" "$LIST"' EXIT

{
  git ls-files
  git ls-files --others --exclude-standard
} | sort -u > "$RAW"

EXCLUDED_COUNT=0
EXCLUDED_BYTES=0
: > "$LIST"
while IFS= read -r f; do
  [ -f "$f" ] || continue
  if is_excluded "$f"; then
    EXCLUDED_COUNT=$((EXCLUDED_COUNT + 1))
    sz=$(stat -c '%s' "$f" 2>/dev/null || echo 0)
    EXCLUDED_BYTES=$((EXCLUDED_BYTES + sz))
    continue
  fi
  printf '%s\n' "$f"
done < "$RAW" > "$LIST"

COUNT=$(wc -l < "$LIST")
TOTAL_BYTES=$(while IFS= read -r f; do stat -c '%s' "$f" 2>/dev/null || echo 0; done < "$LIST" \
             | awk '{s+=$1} END {printf "%.0f", s+0}')

echo "[export] 文件数：$COUNT"
echo "[export] 原始体积：$((TOTAL_BYTES / 1024 / 1024)) MB"
echo "[export] 显式排除：${EXCLUDED_COUNT} 个文件 / $((EXCLUDED_BYTES / 1024 / 1024)) MB（厂商调优现场目录）"
echo "[export] 输出：$WS/$OUT"
echo "[export] 清单前 10 行："
head -10 "$LIST" | sed 's/^/    /'

if [ "$DRY_RUN" = "1" ]; then
  echo "[export] DRY_RUN=1，跳过打包"
  exit 0
fi

tar -czf "$OUT" -T "$LIST"
echo "[export] 大小：$(du -h "$OUT" | cut -f1)"
echo "[export] SHA256：$(sha256sum "$OUT" | cut -d' ' -f1)"
