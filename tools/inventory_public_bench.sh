#!/usr/bin/env bash
set -o pipefail
W=/root/private_data/workspace/robo
B="$W/third_party/benchmarks"

echo "===== RoboBrain2.5-8B-NV 模型卡全文（上游公布评测） ====="
sed -n '1,200p' "$W/third_party/upstream_eval/cards/BAAI__RoboBrain2.5-8B-NV.md" | sed -n '30,160p'

echo
echo "===== CV-Bench 样本数与字段 ====="
python3 - <<'PY'
import json, os
B = "/root/private_data/workspace/robo/third_party/benchmarks/CV-Bench"
for f in ("test_2d.jsonl", "test_3d.jsonl"):
    p = os.path.join(B, f)
    if not os.path.exists(p):
        print(f, "缺失"); continue
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    print(f"{f}: {len(rows)} 条, 字段={sorted(rows[0].keys())}")
    from collections import Counter
    for k in ("source", "type", "dataset"):
        if k in rows[0]:
            print("   ", k, dict(Counter(r.get(k) for r in rows)))
    print("    示例 prompt:", str(rows[0].get("prompt"))[:220])
    print("    示例 answer:", rows[0].get("answer"))
PY

echo
echo "===== RefSpatial-Bench 各 split 题量与字段 ====="
python3 - <<'PY'
import json, os
B = "/root/private_data/workspace/robo/third_party/benchmarks/RefSpatial-Bench"
for split in ("Location", "Placement", "Unseen"):
    p = os.path.join(B, split, "question.json")
    if not os.path.exists(p):
        print(split, "缺失"); continue
    d = json.load(open(p, encoding="utf-8"))
    n = len(d) if isinstance(d, list) else len(d.get("questions", d))
    print(f"{split}: {n} 题, 字段={sorted((d[0] if isinstance(d, list) else list(d.values())[0]).keys())}")
    first = d[0] if isinstance(d, list) else list(d.values())[0]
    print("    prompt:", str(first.get("prompt"))[:160])
    print("    suffix:", str(first.get("suffix"))[:160])
    imgs = os.path.join(B, split, "image")
    masks = os.path.join(B, split, "mask")
    print("    image:", len(os.listdir(imgs)) if os.path.isdir(imgs) else "无",
          " mask:", len(os.listdir(masks)) if os.path.isdir(masks) else "无")
PY

echo
echo "===== TraceSpatial-Bench 结构 ====="
python3 - <<'PY'
import json, os
B = "/root/private_data/workspace/robo/third_party/benchmarks/TraceSpatial-Bench"
print("顶层:", sorted(os.listdir(B)))
p = os.path.join(B, "trajectory_dataset.json")
print("trajectory_dataset.json 存在:", os.path.exists(p))
if os.path.exists(p):
    d = json.load(open(p, encoding="utf-8"))
    print("  条数:", len(d), "字段:", sorted(d[0].keys()))
    print("  示例:", json.dumps({k: (str(v)[:90]) for k, v in d[0].items()}, ensure_ascii=False))
raw = os.path.join(B, "raw_data")
if os.path.isdir(raw):
    subs = sorted(os.listdir(raw))
    print("raw_data 子目录数:", len(subs))
    if subs:
        s = os.path.join(raw, subs[0])
        print("  ", subs[0], sorted(os.listdir(s)))
PY
