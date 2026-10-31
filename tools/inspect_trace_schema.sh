#!/usr/bin/env bash
set -o pipefail
W=/root/private_data/workspace/robo
B="$W/third_party/benchmarks/TraceSpatial-Bench"

echo "===== 模型卡头部（上游公布的评测表） ====="
sed -n '1,30p' "$W/third_party/upstream_eval/cards/BAAI__RoboBrain2.5-8B-NV.md"

echo
echo "===== TraceSpatial parquet schema ====="
python3 - <<'PY'
import json, os
B = "/root/private_data/workspace/robo/third_party/benchmarks/TraceSpatial-Bench"
p = os.path.join(B, "data", "train-00000-of-00001.parquet")
try:
    import pyarrow.parquet as pq
    t = pq.read_table(p)
    print("rows =", t.num_rows)
    print("schema:")
    print(t.schema)
    row = t.slice(0, 1).to_pylist()[0]
    for k, v in row.items():
        s = str(v)
        print("  %-22s %s" % (k, s[:160]))
except Exception as e:
    print("读取失败:", type(e).__name__, e)
PY

echo
echo "===== 上游参考预测（决定期望输出格式） ====="
for f in Model_Prediction_TraceSpatial_2D.jsonl Model_Prediction_TraceSpatial_3D.jsonl; do
  p="$B/$f"
  [ -e "$p" ] || { echo "$f 缺失"; continue; }
  echo "--- $f  行数=$(wc -l < "$p")"
  head -n 1 "$p" | cut -c1-700
  echo
done

echo
echo "===== robo_infer 的提示词模板（我们模型自己的 trajectory/pointing 模板） ====="
grep -nE "TASK_PROMPTS|trajectory|pointing|grounding|general" \
     "$W/third_party/RoboBrain2.5/inference.py" 2>/dev/null | head -25
