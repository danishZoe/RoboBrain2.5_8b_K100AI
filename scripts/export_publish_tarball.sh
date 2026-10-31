#!/usr/bin/env bash
# =============================================================================
# 导出「可发布文件清单」压缩包：用于把远端工作目录的交付物同步到代码托管平台
#
# 规则：
#   * 只包含 git 已跟踪的文件 + 未跟踪但未被 .gitignore 忽略的文件
#   * 排除运行产物（outputs/ logs/ tmp/ cache/ envs/）、大体积第三方镜像、
#     崩溃转储（core.*）、运行时组装的调优库目录（/rocblas_tuned/）
#   * 不修改工作区，不提交任何东西（提交由调用方决定）
#
# 用法：bash scripts/export_publish_tarball.sh [输出路径]
#        默认输出 tmp/publish-<时间戳>.tar.gz
# =============================================================================
set -euo pipefail

WS="${WS:-/root/private_data/workspace/robo}"
cd "$WS"

OUT="${1:-tmp/publish-$(date +%Y%m%d_%H%M%S).tar.gz}"
mkdir -p "$(dirname "$OUT")"

LIST="$(mktemp)"
trap 'rm -f "$LIST"' EXIT

{
  git ls-files
  git ls-files --others --exclude-standard
} | sort -u | while IFS= read -r f; do
  [ -f "$f" ] || continue
  printf '%s\n' "$f"
done > "$LIST"

COUNT=$(wc -l < "$LIST")
echo "[export] 文件数：$COUNT"
echo "[export] 输出：$WS/$OUT"

tar -czf "$OUT" -T "$LIST"
echo "[export] 大小：$(du -h "$OUT" | cut -f1)"
echo "[export] SHA256：$(sha256sum "$OUT" | cut -d' ' -f1)"
echo "[export] 清单前 10 行："
head -10 "$LIST" | sed 's/^/    /'
