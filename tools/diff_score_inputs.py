#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定位：同配置两次运行，预测逐题一致，但官方判分结果不同。
逐项对比判分输入（predictions_official.json / annotations.json / 判分脚本）。"""
import hashlib
import json
import os

WS = "/root/private_data/workspace/robo/outputs/bench_public"
A = os.path.join(WS, "cvbench_upscale", "canonical_png")
B = os.path.join(WS, "cvbench_prompt", "flagevalmm")


def sha(p):
    try:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            h.update(fh.read())
        return h.hexdigest()[:16]
    except OSError:
        return "MISSING"


def load(p):
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as e:
        return ("ERR", str(e))


print("=== 文件指纹 ===")
for name in ("predictions_official.json", "annotations.json", "per_sample.json",
             "_score_cv_bench.py", "cv_bench_official_result.json"):
    print("  %-28s A=%s  B=%s" % (name, sha(os.path.join(A, name)), sha(os.path.join(B, name))))

for name in ("predictions_official.json", "annotations.json"):
    pa, pb = os.path.join(A, name), os.path.join(B, name)
    da, db = load(pa), load(pb)
    print("\n=== %s ===" % name)
    if isinstance(da, dict) and isinstance(db, dict):
        ka, kb = set(da), set(db)
        print("  A 顶层键 %d 个, B 顶层键 %d 个; 仅A=%s 仅B=%s"
              % (len(ka), len(kb), sorted(ka - kb)[:5], sorted(kb - ka)[:5]))
        for k in sorted(ka & kb):
            va, vb = da[k], db[k]
            same = (json.dumps(va, sort_keys=True, ensure_ascii=False)
                    == json.dumps(vb, sort_keys=True, ensure_ascii=False))
            if not same:
                if isinstance(va, list) and isinstance(vb, list):
                    print("  键 %r: 列表长度 A=%d B=%d" % (k, len(va), len(vb)))
                    for i, (x, y) in enumerate(zip(va, vb)):
                        sx = json.dumps(x, sort_keys=True, ensure_ascii=False)
                        sy = json.dumps(y, sort_keys=True, ensure_ascii=False)
                        if sx != sy:
                            print("    [idx %d] A=%s" % (i, sx[:220]))
                            print("    [idx %d] B=%s" % (i, sy[:220]))
                            if i > 4:
                                print("    ...")
                                break
                else:
                    print("  键 %r 不同:\n    A=%s\n    B=%s"
                          % (k, json.dumps(va, ensure_ascii=False)[:400],
                             json.dumps(vb, ensure_ascii=False)[:400]))
    elif isinstance(da, list) and isinstance(db, list):
        print("  A 列表 len=%d, B 列表 len=%d" % (len(da), len(db)))
        n = 0
        for i, (x, y) in enumerate(zip(da, db)):
            sx = json.dumps(x, sort_keys=True, ensure_ascii=False)
            sy = json.dumps(y, sort_keys=True, ensure_ascii=False)
            if sx != sy:
                n += 1
                if n <= 5:
                    print("    [idx %d] A=%s" % (i, sx[:240]))
                    print("    [idx %d] B=%s" % (i, sy[:240]))
        print("  差异条目数 =", n)

print("\n=== 判分脚本差异 ===")
pa, pb = os.path.join(A, "_score_cv_bench.py"), os.path.join(B, "_score_cv_bench.py")
try:
    la = open(pa, encoding="utf-8").read().splitlines()
    lb = open(pb, encoding="utf-8").read().splitlines()
    import difflib
    d = list(difflib.unified_diff(la, lb, "A", "B", lineterm="", n=1))
    print("\n".join(d[:60]) if d else "  脚本逐字相同")
except OSError as e:
    print("  ", e)

print("\n=== 官方结果里的 scored_predictions 计数 ===")
for tag, d in (("A", A), ("B", B)):
    o = load(os.path.join(d, "cv_bench_official_result.json"))
    if isinstance(o, dict):
        sp = o.get("scored_predictions") or []
        ntrue = sum(1 for x in sp if x.get("correct"))
        print("  %s: scored_predictions=%d  correct=%d  acc=%.4f"
              % (tag, len(sp), ntrue, (ntrue / len(sp)) if sp else -1))
