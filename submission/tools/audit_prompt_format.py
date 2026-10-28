#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""审计 dataset 提示词口径的答案格式：
   模型在 dataset 口径下输出 "(C) 1" 这类「字母+数值」格式，
   若抽取器不归一化，会把**答对**的题判成错 —— 必须查清 §5.11 结论是否成立。"""
import json
import os
import re

WS = "/root/private_data/workspace/robo/outputs/bench_public"

RE_PAREN = re.compile(r"^\s*\(?\s*([A-F])\s*\)?\s*[\.\):,]?\s*\d*\s*$")
RE_LETTER_ONLY = re.compile(r"^\s*([A-F])\s*[\.\)]?\s*$")


def load(sub):
    p = os.path.join(WS, sub, "per_sample.json")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def norm_letter(s):
    """把 '(C) 1' / 'C.' / 'C' 统一成 'C'；无法归一化返回 None。"""
    if s is None:
        return None
    s = str(s).strip()
    m = RE_PAREN.match(s)
    if m:
        return m.group(1)
    m = RE_LETTER_ONLY.match(s)
    if m:
        return m.group(1)
    return None


def official_result(sub):
    p = os.path.join(WS, sub, "cv_bench_official_result.json")
    try:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return {}
    return {k: d.get(k) for k in ("accuracy_2d_ade", "accuracy_2d_coco",
                                  "accuracy_2d", "accuracy_3d_omni", "accuracy")}


for sub in ("cvbench_prompt/dataset", "cvbench_prompt/flagevalmm",
            "cvbench_upscale/canonical_png", "cvbench_upscale/up2_png"):
    recs = load(sub)
    n = len(recs)
    raw_match = letter_match = 0
    fmt_counts = {}
    rescued = []
    for r in recs:
        gt, pr = r.get("gt_answer"), r.get("extracted_answer")
        if pr == gt:
            raw_match += 1
        L = norm_letter(pr)
        kind = "letter_only" if RE_LETTER_ONLY.match(str(pr or "").strip()) else (
            "paren+num" if RE_PAREN.match(str(pr or "").strip()) else "other")
        fmt_counts[kind] = fmt_counts.get(kind, 0) + 1
        if L is not None and L == gt:
            letter_match += 1
            if pr != gt:
                rescued.append({"q": r.get("question_id"), "gt": gt, "raw_pred": pr,
                                "norm": L, "source": r.get("source")})
    print("\n=== %s ===" % sub)
    print("  题数                       = %d" % n)
    print("  抽取值与 GT 逐字相等       = %d  (%.2f%%)" % (raw_match, 100.0 * raw_match / n))
    print("  归一化成字母后与 GT 相等   = %d  (%.2f%%)" % (letter_match, 100.0 * letter_match / n))
    print("  答案形态分布               = %s" % fmt_counts)
    print("  因格式被判错（可救回）     = %d" % len(rescued))
    for x in rescued[:5]:
        print("     q%-6s gt=%s raw=%r -> %s  (%s)" % (x["q"], x["gt"], x["raw_pred"],
                                                      x["norm"], x["source"]))

# 若把 dataset 口径按归一化重算，官方分数会是多少
print("\n=== dataset 口径按字母归一化后重算 ===")
recs = load("cvbench_prompt/dataset")
by_src = {}
for r in recs:
    src = r.get("source")
    gt, pr = r.get("gt_answer"), r.get("extracted_answer")
    L = norm_letter(pr)
    ok_raw = (pr == gt)
    ok_norm = (L is not None and L == gt)
    d = by_src.setdefault(src, {"n": 0, "raw": 0, "norm": 0})
    d["n"] += 1
    d["raw"] += ok_raw
    d["norm"] += ok_norm
for src, d in sorted(by_src.items()):
    print("  %-8s n=%d  逐字口径=%.2f%%  归一化口径=%.2f%%"
          % (src, d["n"], 100.0 * d["raw"] / d["n"], 100.0 * d["norm"] / d["n"]))

# ---- 落盘产物 ----
summary = {"note": "答案格式审计：dataset 口径模型输出 '(C) 1' 形式，官方判分器对其的"
                   "归一化比严格字母匹配更宽松；flagevalmm 口径 19 条 'A.' 官方已认对。",
           "subsets": {}}
for sub in ("cvbench_prompt/dataset", "cvbench_prompt/flagevalmm",
            "cvbench_upscale/canonical_png", "cvbench_upscale/up2_png"):
    recs = load(sub)
    n = len(recs)
    raw_match = letter_match = 0
    fmt_counts = {}
    src_stat = {}
    for r in recs:
        gt, pr = r.get("gt_answer"), r.get("extracted_answer")
        if pr == gt:
            raw_match += 1
        L = norm_letter(pr)
        kind = "letter_only" if RE_LETTER_ONLY.match(str(pr or "").strip()) else (
            "paren+num" if RE_PAREN.match(str(pr or "").strip()) else "other")
        fmt_counts[kind] = fmt_counts.get(kind, 0) + 1
        ok_norm = (L is not None and L == gt)
        if ok_norm:
            letter_match += 1
        s = src_stat.setdefault(r.get("source"), {"n": 0, "raw": 0, "norm": 0})
        s["n"] += 1
        s["raw"] += (pr == gt)
        s["norm"] += ok_norm
    off = official_result(sub)
    summary["subsets"][sub] = {
        "n": n,
        "literal_match": raw_match,
        "literal_match_pct": round(100.0 * raw_match / n, 2),
        "letter_normalized_match": letter_match,
        "letter_normalized_match_pct": round(100.0 * letter_match / n, 2),
        "answer_format_counts": fmt_counts,
        "by_source": src_stat,
        "official_result": off,
    }

out = os.path.join(WS, "cvbench_answer_format_audit.json")
with open(out, "w", encoding="utf-8") as fh:
    json.dump(summary, fh, ensure_ascii=False, indent=2)
print("\n写出:", out)
