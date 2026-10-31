#!/usr/bin/env bash
# 诊断：CV-Bench 判分链路（answer 字段格式 / evaluate_multiple_choice 比对逻辑 / 选项构造）
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONPATH="$WS/envs/pylibs_core:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore

echo "########## 1. parquet 的 answer / choices 字段形态 ##########"
python3 - <<'PY'
import os, collections, json
import pyarrow.parquet as pq
D = "/root/private_data/workspace/robo/third_party/benchmarks/CV-Bench"
forms = collections.Counter()
nopt = collections.Counter()
srcs = collections.Counter()
sample = []
for name in ("test_2d.parquet", "test_3d.parquet"):
    pf = pq.ParquetFile(os.path.join(D, name))
    for b in pf.iter_batches(batch_size=128,
                             columns=["idx","type","source","answer","choices","filename","question"]):
        for row in b.to_pylist():
            a = row["answer"]
            forms[(type(a).__name__, (a[:3] if isinstance(a, str) else str(a)[:3]),
                   len(a) if isinstance(a, str) else -1)] += 1
            nopt[len(row["choices"] or [])] += 1
            srcs[row["source"]] += 1
            if len(sample) < 3:
                sample.append({"idx": row["idx"], "answer": a, "choices": row["choices"],
                               "question": row["question"][:160], "filename": row["filename"]})
print("  answer 形态 (类型, 前3字符, 长度) -> 计数")
for k, v in forms.most_common(12):
    print("   ", k, "->", v)
print("  choices 长度分布:", dict(nopt))
print("  source 分布:", dict(srcs))
print("  样例:")
for s in sample:
    print("   ", json.dumps(s, ensure_ascii=False)[:400])
PY

echo
echo "########## 2. 官方 evaluate_multiple_choice ##########"
sed -n '1,120p' third_party/upstream_eval/FlagEvalMM/flagevalmm/evaluator/common_types.py | grep -n 'def evaluate_multiple_choice' -A 60

echo
echo "########## 3. 本次已跑出的 CV-Bench 逐题结果：错题分布 ##########"
python3 - <<'PY'
import json, os, collections
p = "/root/private_data/workspace/robo/outputs/bench_public/cvbench/per_sample.json"
if not os.path.exists(p):
    print("  缺 per_sample.json")
else:
    d = json.load(open(p, encoding="utf-8"))
    items = d if isinstance(d, list) else d.get("samples", [])
    c = collections.Counter()
    nopt_err = collections.Counter()
    for r in items:
        ok = r.get("correct")
        ext = (r.get("extracted_answer") or "").strip()
        src = r.get("source") or r.get("split")
        c[(src, bool(ok))] += 1
        if not ok:
            nopt_err[(src, ext)] += 1
    print("  (source, correct) ->", dict(c))
    print("  错误答案 Top20:")
    for k, v in nopt_err.most_common(20):
        print("   ", k, "->", v)
PY
