#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CV-Bench Count 子任务失分模式诊断（只读产物，不发请求）。"""
import collections
import json
import os
import re
import sys

WS = "/root/private_data/workspace/robo"
ROOT = sys.argv[1] if len(sys.argv) > 1 else WS + "/outputs/bench_public/cvbench"

anns = json.load(open(ROOT + "/annotations.json", encoding="utf-8"))
per = json.load(open(ROOT + "/per_sample.json", encoding="utf-8"))

cnt = [r for r in per if r.get("task") == "Count"]
print("Count 样本数 =", len(cnt))

ok = 0
gt_hist = collections.Counter()
pred_hist = collections.Counter()
conf = collections.Counter()
# 模型输出的「形态」统计
shape = collections.Counter()
opt_val_hist = collections.Counter()
nopts = collections.Counter()

def shape_of(raw):
    t = (raw or "").strip()
    if not t:
        return "empty"
    if re.fullmatch(r"[A-H]", t):
        return "pure_letter"
    if re.fullmatch(r"\(?[A-H]\)?[\.\:\)]?", t):
        return "letter_with_punct"
    if re.fullmatch(r"\d+", t):
        return "pure_number"
    m = re.search(r"[A-H]", t)
    if m:
        return "contains_letter"
    if re.search(r"\d", t):
        return "contains_number"
    return "other"

def num_of(s):
    m = re.search(r"\d+", str(s))
    return int(m.group(0)) if m else None

wrong_samples = []
for r in cnt:
    qid = r["question_id"]
    g = anns.get(qid, {})
    gt = (r.get("gt_answer") or "").strip()[:1].upper()
    pred = (r.get("extracted_answer") or "").strip()[:1].upper()
    raw = r.get("raw_output") or ""
    sh = shape_of(r.get("extracted_answer"))
    shape[sh] += 1
    nopts[len(g.get("options") or [])] += 1
    good = (gt == pred)
    ok += good
    gt_hist[gt] += 1
    pred_hist[pred] += 1
    conf[(gt, pred)] += 1
    opts = g.get("options") or []
    # 选项值分布：正确答案对应的数值
    if opts:
        idx = ord(gt) - 65
        if 0 <= idx < len(opts):
            opt_val_hist[num_of(opts[idx])] += 1
    if not good and len(wrong_samples) < 12:
        wrong_samples.append({
            "qid": qid, "task": g.get("task"), "source": g.get("source"),
            "question": (r.get("question") or g.get("question") or "")[:220],
            "options": opts, "gt": gt, "pred": pred,
            "shape": sh,
            "raw_tail": raw[-400:],
        })

print("正确 = %d / %d = %.2f%%" % (ok, len(cnt), 100.0 * ok / len(cnt)))
print("GT 字母分布 :", dict(sorted(gt_hist.items())))
print("预测字母分布:", dict(sorted(pred_hist.items())))
print("抽取形态分布:", dict(shape.most_common()))
print("选项数分布  :", dict(sorted(nopts.items())))
print("GT 对应数值分布:", dict(sorted(opt_val_hist.items(), key=lambda kv: (kv[0] is None, kv[0]))))
print("\n混淆对 top12:")
for (g, p), c in conf.most_common(12):
    print("   gt=%s pred=%s  n=%d" % (g, p, c))

# 模型输出的是数字时，数字是否等于某个选项值？等于则字母映射应能救回
num_vs_opt = collections.Counter()
for r in cnt:
    qid = r["question_id"]
    g = anns.get(qid, {})
    opts = g.get("options") or []
    ex = (r.get("extracted_answer") or "").strip()
    m = re.search(r"\d+", ex)
    if not m or not opts:
        continue
    v = int(m.group(0))
    hit = [chr(65 + i) for i, o in enumerate(opts) if num_of(o) == v]
    num_vs_opt["num_in_options" if hit else "num_not_in_options"] += 1
    if hit:
        pred = (r.get("extracted_answer") or "").strip()[:1].upper()
        gt = (r.get("gt_answer") or "").strip()[:1].upper()
        num_vs_opt["num_hit_correct" if hit[0] == gt else "num_hit_wrong"] += 1
print("\n数字型输出与选项的关系:", dict(num_vs_opt))

print("\n===== 失分样例 =====")
for s in wrong_samples:
    print("-" * 70)
    print("qid=%s source=%s gt=%s pred=%s shape=%s" %
          (s["qid"], s["source"], s["gt"], s["pred"], s["shape"]))
    print("Q:", s["question"])
    print("options:", s["options"])
    print("raw_tail:", repr(s["raw_tail"]))
