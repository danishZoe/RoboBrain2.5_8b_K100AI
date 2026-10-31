#!/usr/bin/env bash
# 探查 RefSpatial-Bench 目录结构与 question.json 字段
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
B=third_party/benchmarks/RefSpatial-Bench

echo "===== 1) 顶层 ====="
ls -la "$B" | head -n 30

echo
echo "===== 2) 各 split 目录 ====="
for s in Location Placement Unseen; do
  echo "--- $s ---"
  ls "$B/$s" 2>/dev/null | head -n 12
done

echo
echo "===== 3) question.json 结构 ====="
python3 - <<'PY'
import json, os
B = "third_party/benchmarks/RefSpatial-Bench"
for s in ("Location", "Placement", "Unseen"):
    p = os.path.join(B, s, "question.json")
    if not os.path.exists(p):
        print(f"[{s}] MISSING {p}")
        continue
    d = json.load(open(p, encoding="utf-8"))
    print(f"[{s}] n={len(d)} type={type(d).__name__}")
    if d:
        print("  keys:", list(d[0].keys()))
        print("  item0:", json.dumps(d[0], ensure_ascii=False)[:900])
PY

echo
echo "===== 4) CV-Bench 目录 ====="
C=third_party/benchmarks/CV-Bench
ls -la "$C" | head -n 20
echo "--- CV-Bench README 的 accuracy 段 ---"
grep -n -A 30 -i 'accuracy' "$C/README.md" 2>/dev/null | head -n 60
