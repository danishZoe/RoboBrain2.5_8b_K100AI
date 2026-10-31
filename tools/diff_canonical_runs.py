#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对拍两次「同配置」CV-Bench Count 运行：
   A = cvbench_upscale/canonical_png     （上采样实验的 canonical 对照）
   B = cvbench_prompt/flagevalmm         （提示词实验的主口径对照）
两者都是 canonical_png + flagevalmm 提示词 + 788 题，配置应当逐字相同。
差异必须能被解释，否则 +2.9~3.3 pt 的结论就不成立。
"""
import json
import os
import sys

WS = "/root/private_data/workspace/robo/outputs/bench_public"
A = os.path.join(WS, "cvbench_upscale", "canonical_png")
B = os.path.join(WS, "cvbench_prompt", "flagevalmm")


def load(d):
    for name in ("per_sample.json",):
        p = os.path.join(d, name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
    return None


def as_map(obj):
    """兼容 list[dict] / dict 两种形态。"""
    if isinstance(obj, dict):
        # 形如 {"predictions": [...]} 或 {qid: rec}
        if "predictions" in obj and isinstance(obj["predictions"], list):
            obj = obj["predictions"]
        else:
            out = {}
            for k, v in obj.items():
                if isinstance(v, dict):
                    rec = dict(v)
                    rec.setdefault("question_id", k)
                    out[str(rec["question_id"])] = rec
            return out
    out = {}
    for rec in obj:
        qid = str(rec.get("question_id", rec.get("id", rec.get("index"))))
        out[qid] = rec
    return out


a = as_map(load(A))
b = as_map(load(B))
print("A(canonical_png) 条数 =", len(a))
print("B(flagevalmm)  条数 =", len(b))

common = sorted(set(a) & set(b), key=lambda x: int(x) if x.isdigit() else x)
print("共同题号 =", len(common), " 仅A =", len(set(a) - set(b)), " 仅B =", len(set(b) - set(a)))

diff = []
for q in common:
    ra, rb = a[q], b[q]
    ka = (ra.get("answer"), ra.get("correct"), ra.get("label"))
    kb = (rb.get("answer"), rb.get("correct"), rb.get("label"))
    if ka != kb:
        diff.append((q, ra, rb))

print("\n解析结果不同的题数 =", len(diff), " / ", len(common),
      " = %.2f%%" % (100.0 * len(diff) / max(1, len(common))))

# 原始模型输出是否也不同（区分「模型不确定」与「我们解析不一致」）
raw_same = 0
raw_diff = 0
raw_missing = 0
for q, ra, rb in diff:
    ta = ra.get("raw_answer")
    tb = rb.get("raw_answer")
    if ta is None or tb is None:
        raw_missing += 1
    elif ta == tb:
        raw_same += 1
    else:
        raw_diff += 1
print("  其中原始输出也不同的 =", raw_diff, " 原始输出相同(解析差异) =", raw_same,
      " 无原始输出 =", raw_missing)

for q, ra, rb in diff[:20]:
    print("  q%-6s A: ans=%-4s correct=%-5s label=%-3s | B: ans=%-4s correct=%-5s label=%-3s"
          % (q, ra.get("answer"), ra.get("correct"), ra.get("label"),
             rb.get("answer"), rb.get("correct"), rb.get("label")))

# 汇总核对
for tag, d in (("A", A), ("B", B)):
    p = os.path.join(d, "cv_bench_official_result.json")
    if os.path.exists(p):
        o = json.load(open(p, encoding="utf-8"))
        print("%s 官方结果: %s" % (tag, json.dumps(o, ensure_ascii=False)[:300]))
