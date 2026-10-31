#!/usr/bin/env bash
# 从带 ROCBLAS_LAYER=3 的服务日志提取唯一 GEMM size，生成 fc_size.log
# 用法: extract_fc_size.sh <serve_log> <out_fc_size.log>
set -euo pipefail
LOG="$1"
OUT="$2"
# layer3 行形如: ./rocblas-bench -f gemm_ex --transposeA N --transposeB N -m ... 
grep -o '\./rocblas-bench .*' "$LOG" | sort | uniq -c | sort -nr > "$OUT"
echo "[extract] unique sizes: $(wc -l < "$OUT")"
head -20 "$OUT"
