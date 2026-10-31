#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配对符号检验：把「聚合分差」拆成逐题的 对→错 / 错→对 计数。

对 CV-Bench Count（788 题）：
  基线对拍 A = upscale/canonical_png  vs  B = prompt/flagevalmm   → 同配置两次运行 = 噪声基线
  实验对拍 C = upscale/canonical_png  vs  D = upscale/up2_png      → 上采样效应
  实验对拍 C = upscale/canonical_png  vs  E = upscale/up2_jpeg95   → 上采样效应
  实验对拍 C = upscale/canonical_png  vs  F = prompt/dataset       → 提示词口径效应

符号检验：只看 n_10（错→对）与 n_01（对→错），用精确二项检验算 p 值。
"""
import json
import os
from math import comb

WS = "/root/private_data/workspace/robo/outputs/bench_public"


def per_sample(sub):
    """per_sample.json：元素含 gt_answer / extracted_answer（没有 correct 字段）。"""
    p = os.path.join(WS, sub, "per_sample.json")
    with open(p, encoding="utf-8") as fh:
        d = json.load(fh)
    out = {}
    if isinstance(d, list):
        for r in d:
            out[str(r.get("question_id"))] = {
                "correct": r.get("extracted_answer") == r.get("gt_answer"),
                "gt": r.get("gt_answer"), "pred": r.get("extracted_answer"),
                "source": r.get("source"), "task": r.get("task"),
            }
    else:
        for k, v in d.items():
            out[str(v.get("question_id", k))] = {
                "correct": v.get("extracted_answer") == v.get("gt_answer"),
                "gt": v.get("gt_answer"), "pred": v.get("extracted_answer"),
                "source": v.get("source"), "task": v.get("task"),
            }
    return out


def two_sided_binom(k, n):
    """精确双侧二项检验（零假设 p=0.5）。
    k = 观测到的少数方向计数，n = 翻转总数。
    p = 2 * P(X <= k),  X ~ Bin(n, 0.5)，并截断到 1.0。
    """
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def official(sub):
    p = os.path.join(WS, sub, "cv_bench_official_result.json")
    with open(p, encoding="utf-8") as fh:
        d = json.load(fh)
    return d


def compare(name, ref_sub, test_sub):
    a, b = per_sample(ref_sub), per_sample(test_sub)
    common = sorted(set(a) & set(b), key=lambda x: int(x) if x.isdigit() else x)
    n10 = n01 = n00 = n11 = 0
    flips = []
    for q in common:
        if not a[q]["correct"] and b[q]["correct"]:
            n10 += 1
            flips.append({"question_id": q, "dir": "wrong->right",
                          "gt": a[q]["gt"], "ref": a[q]["pred"], "test": b[q]["pred"]})
        elif a[q]["correct"] and not b[q]["correct"]:
            n01 += 1
            flips.append({"question_id": q, "dir": "right->wrong",
                          "gt": a[q]["gt"], "ref": a[q]["pred"], "test": b[q]["pred"]})
        elif a[q]["correct"]:
            n11 += 1
        else:
            n00 += 1
    n = n10 + n01
    p = two_sided_binom(min(n10, n01), n) if n else 1.0
    oa, ob = official(ref_sub), official(test_sub)
    d2 = (ob.get("accuracy_2d", 0) - oa.get("accuracy_2d", 0))
    d_ade = (ob.get("accuracy_2d_ade", 0) - oa.get("accuracy_2d_ade", 0))
    d_coco = (ob.get("accuracy_2d_coco", 0) - oa.get("accuracy_2d_coco", 0))
    print("\n### %s" % name)
    print("  共同题数            = %d" % len(common))
    print("  错->对 (n10)        = %d" % n10)
    print("  对->错 (n01)        = %d" % n01)
    print("  同对 (n11)          = %d   同错 (n00) = %d" % (n11, n00))
    print("  翻转总数            = %d  (%.2f%%)" % (n, 100.0 * n / max(1, len(common))))
    print("  符号检验 p          = %.5g" % p)
    print("  官方 acc_2d         = %.2f -> %.2f  (Δ = %+.2f pt)" % (oa.get("accuracy_2d"), ob.get("accuracy_2d"), d2))
    print("  官方 acc_2d_ade     = %.2f -> %.2f  (Δ = %+.2f pt)" % (oa.get("accuracy_2d_ade"), ob.get("accuracy_2d_ade"), d_ade))
    print("  官方 acc_2d_coco    = %.2f -> %.2f  (Δ = %+.2f pt)" % (oa.get("accuracy_2d_coco"), ob.get("accuracy_2d_coco"), d_coco))
    for f in flips[:6]:
        print("    %-10s q%-6s gt=%s ref=%s test=%s" % (f["dir"], f["question_id"],
                                                        f["gt"], f["ref"], f["test"]))
    return {"name": name, "ref": ref_sub, "test": test_sub, "n_common": len(common),
            "n_wrong_to_right": n10, "n_right_to_wrong": n01,
            "n_both_right": n11, "n_both_wrong": n00,
            "flip_rate_pct": round(100.0 * n / max(1, len(common)), 3),
            "sign_test_p": p,
            "accuracy_2d_ref": oa.get("accuracy_2d"), "accuracy_2d_test": ob.get("accuracy_2d"),
            "delta_accuracy_2d": round(d2, 2),
            "delta_accuracy_2d_ade": round(d_ade, 2),
            "delta_accuracy_2d_coco": round(d_coco, 2),
            "flips": flips}


rows = []
rows.append(compare("噪声基线：同配置两次运行（canonical_png vs flagevalmm 口径复现）",
                     "cvbench_upscale/canonical_png", "cvbench_prompt/flagevalmm"))
rows.append(compare("上采样 up2_png vs canonical_png",
                     "cvbench_upscale/canonical_png", "cvbench_upscale/up2_png"))
rows.append(compare("上采样 up2_jpeg95 vs canonical_png",
                     "cvbench_upscale/canonical_png", "cvbench_upscale/up2_jpeg95"))
rows.append(compare("提示词 dataset 口径 vs flagevalmm 口径",
                     "cvbench_prompt/flagevalmm", "cvbench_prompt/dataset"))

out = os.path.join(WS, "cvbench_paired_tests.json")
with open(out, "w", encoding="utf-8") as fh:
    json.dump(rows, fh, ensure_ascii=False, indent=2)
print("\n写出:", out)

# 紧凑版协议对比（与 cvbench_prompt/prompt_compare.json 同构，便于镜像）
compact = []
for name, sub in (("canonical_png", "cvbench_upscale/canonical_png"),
                  ("up2_png", "cvbench_upscale/up2_png"),
                  ("up2_jpeg95", "cvbench_upscale/up2_jpeg95"),
                  ("up4_png", "cvbench_upscale/up4_png")):
    try:
        ps = per_sample(sub)
    except OSError:
        continue
    if not ps:
        continue
    src = {}
    for q, v in ps.items():
        d = src.setdefault(v["source"], {"n": 0, "c": 0})
        d["n"] += 1
        d["c"] += bool(v["correct"])
    allok = sum(1 for v in ps.values() if v["correct"])
    try:
        off = official(sub)
    except (OSError, ValueError):
        off = {}
    off = {k: v for k, v in off.items() if not isinstance(v, (list, dict))}
    compact.append({
        "protocol": name,
        "ade": [src.get("ADE20K", {}).get("c", 0), src.get("ADE20K", {}).get("n", 0)],
        "coco": [src.get("COCO", {}).get("c", 0), src.get("COCO", {}).get("n", 0)],
        "all": [allok, len(ps)],
        "official": off,
        "note": "协议在评测集上按分数选定 -> 数值有选择偏差，仅用于归因，不作上报口径",
    })

out2 = os.path.join(WS, "cvbench_upscale", "upscale_compare.json")
with open(out2, "w", encoding="utf-8") as fh:
    json.dump(compact, fh, ensure_ascii=False, indent=2)
print("写出:", out2)
