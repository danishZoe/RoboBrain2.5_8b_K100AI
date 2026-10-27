#!/usr/bin/env bash
set -o pipefail
W=/root/private_data/workspace/robo

echo "===== 下载进度 ====="
du -sh "$W/third_party/benchmarks"/* 2>/dev/null
find "$W/third_party/benchmarks" -maxdepth 2 -type d 2>/dev/null | head -20
echo "--- 文件数/体积 ---"
for d in "$W/third_party/benchmarks"/*/; do
  [ -d "$d" ] || continue
  printf '%-24s files=%-6s size=%s\n' "$(basename "$d")" \
    "$(find "$d" -type f -not -path '*/.cache/*' | wc -l)" \
    "$(du -sh "$d" 2>/dev/null | cut -f1)"
done

echo
echo "===== RoboBrain2.5-8B-NV 模型卡里的评测表 ====="
MC="$W/third_party/upstream_eval/cards/BAAI__RoboBrain2.5-8B-NV.md"
if [ -s "$MC" ]; then
  grep -niE -B3 -A25 "benchmark|evaluation|CV-?Bench|RefSpatial|TraceSpatial" "$MC" | head -70
else
  echo "(模型卡尚未取到)"
fi

echo
echo "===== 各数据集关键数据文件 ====="
for d in CV-Bench RefSpatial-Bench TraceSpatial-Bench; do
  echo "--- $d"
  find "$W/third_party/benchmarks/$d" -type f -not -path '*/.cache/*' \
       \( -name '*.json' -o -name '*.jsonl' -o -name '*.parquet' -o -name '*.csv' -o -name '*.tsv' \) \
       2>/dev/null | head -12
done
