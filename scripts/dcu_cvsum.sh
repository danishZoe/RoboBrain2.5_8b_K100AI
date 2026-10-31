#!/bin/bash
W=/root/private_data/workspace/robo
cd "$W/third_party/benchmarks/CV-Bench" || exit 1
echo "=== CV-Bench top level ==="
ls -la
echo "=== du ==="
du -sh . .cache 2>/dev/null
echo "=== .cache tree (depth 4) ==="
find .cache -maxdepth 4 -type d 2>/dev/null | head -30
echo "=== file counts ==="
echo "jsonl:"; wc -l test_2d.jsonl test_3d.jsonl 2>/dev/null
echo "=== sha256 of data files ==="
find . -maxdepth 2 -type f \( -name "*.parquet" -o -name "*.jsonl" -o -name "*.json" -o -name "*.py" \) -printf "%s " -exec sha256sum {} \; 2>/dev/null
echo "=== image dir listing ==="
find .cache -type f 2>/dev/null | head -10
echo "n_images=$(find .cache -type f 2>/dev/null | wc -l)"
echo "=== README head ==="
head -40 README.md 2>/dev/null
