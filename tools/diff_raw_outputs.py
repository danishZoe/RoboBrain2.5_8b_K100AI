#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""决定性检查：同配置两次运行，官方判分输入差 10 条 —— 原始模型输出到底差在哪？
A = cvbench_upscale/canonical_png
B = cvbench_prompt/flagevalmm
"""
import json
import os

WS = "/root/private_data/workspace/robo/outputs/bench_public"
A = os.path.join(WS, "cvbench_upscale", "canonical_png")
B = os.path.join(WS, "cvbench_prompt", "flagevalmm")

TARGETS = ["30", "52", "221", "635", "665"]


def load_jsonl(p):
    out = {}
    if not os.path.exists(p):
        return out
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            qid = str(d.get("question_id", d.get("id")))
            out[qid] = d
    return out


def find_pred(d):
    """找出原始预测文本，兼容多种字段名。"""
    for k in ("raw_prediction", "prediction", "response", "content",
              "model_output", "text", "answer_text", "raw_answer"):
        if k in d and isinstance(d[k], str):
            return k, d[k]
    return None, None


for tag, d in (("A", A), ("B", B)):
    print("=== %s 目录内容 ===" % tag)
    for root, dirs, files in os.walk(os.path.join(d, "predictions")):
        for f in files:
            p = os.path.join(root, f)
            print("   %s  (%d B)" % (os.path.relpath(p, d), os.path.getsize(p)))
    print()

pa = load_jsonl(os.path.join(A, "predictions", "cvbench.jsonl"))
pb = load_jsonl(os.path.join(B, "predictions", "cvbench.jsonl"))
print("原始记录数 A=%d B=%d" % (len(pa), len(pb)))

common = set(pa) & set(pb)
print("共同题号 =", len(common))

n_diff = 0
samples = []
for q in sorted(common, key=lambda x: int(x) if x.isdigit() else 0):
    ka, ta = find_pred(pa[q])
    kb, tb = find_pred(pb[q])
    if ta != tb:
        n_diff += 1
        if len(samples) < 12:
            samples.append((q, ka, ta, kb, tb))
print("原始文本不同的题数 = %d / %d = %.2f%%" % (n_diff, len(common),
                                                100.0 * n_diff / max(1, len(common))))
for q, ka, ta, kb, tb in samples:
    print("  q%-6s [%s] A=%r" % (q, ka, (ta or "")[:90]))
    print("          [%s] B=%r" % (kb, (tb or "")[:90]))

print("\n=== 目标题号逐条 ===")
for q in TARGETS:
    if q in pa and q in pb:
        ka, ta = find_pred(pa[q])
        kb, tb = find_pred(pb[q])
        print("  q%-6s A(%s)=%r" % (q, ka, (ta or "")[:70]))
        print("         B(%s)=%r" % (kb, (tb or "")[:70]))

print("\n=== A/B 原始记录里的其它字段（取一条样例） ===")
if common:
    q0 = sorted(common, key=lambda x: int(x) if x.isdigit() else 0)[0]
    print("  A keys:", sorted(pa[q0].keys()))
    print("  B keys:", sorted(pb[q0].keys()))
