#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出 Count 失分样例的图像 + 元数据到本地目录，供人工目视核对。"""
import json
import os
import shutil
import sys

WS = "/root/private_data/workspace/robo"
ROOT = WS + "/outputs/bench_public/cvbench"
OUT = WS + "/outputs/bench_public/cvbench_count_samples"
MANIFEST = WS + "/outputs/bench_public/manifest"

anns = json.load(open(ROOT + "/annotations.json", encoding="utf-8"))
per = json.load(open(ROOT + "/per_sample.json", encoding="utf-8"))
mapfile = os.path.join(MANIFEST, "cvbench_image_map.json")
imgmap = {r["question_id"]: r["file"] for r in json.load(open(mapfile, encoding="utf-8"))}

os.makedirs(OUT, exist_ok=True)


def num_of(s):
    try:
        return int(str(s).strip())
    except Exception:
        return None


def pick(src, task, pred_delta, want, limit):
    out = []
    for r in per:
        g = anns.get(r["question_id"], {})
        if g.get("source") != src or g.get("task") != task:
            continue
        gt = (r.get("gt_answer") or "").strip()[:1].upper()
        pr = (r.get("extracted_answer") or "").strip()[:1].upper()
        opts = g.get("options") or []
        gi, pi = ord(gt) - 65, ord(pr) - 65
        if not (0 <= gi < len(opts) and 0 <= pi < len(opts)):
            continue
        gv, pv = num_of(opts[gi]), num_of(opts[pi])
        if gv is None or pv is None:
            continue
        if pv - gv != pred_delta:
            continue
        out.append((r, g, gv, pv))
        if len(out) >= limit:
            break
    return out


manifest = []
for src, delta, tag in [("ADE20K", -1, "ade_m1"), ("ADE20K", 1, "ade_p1"),
                         ("ADE20K", -2, "ade_m2"),
                         ("COCO", -1, "coco_m1"), ("COCO", 1, "coco_p1")]:
    got = pick(src, "Count", delta, 0, 3)
    for r, g, gv, pv in got:
        qid = r["question_id"]
        rel = imgmap.get(qid)
        if not rel:
            continue
        srcp = rel if os.path.isabs(rel) else os.path.join(WS, rel)
        if not os.path.exists(srcp):
            continue
        name = "%s_%s.png" % (tag, qid)
        shutil.copy(srcp, os.path.join(OUT, name))
        manifest.append({
            "file": name, "src_file": rel, "question_id": qid,
            "source": src, "task": "Count",
            "question": g.get("question"), "options": g.get("options"),
            "gt_letter": (r.get("gt_answer") or "")[:1].upper(),
            "gt_value": gv, "pred_letter": (r.get("extracted_answer") or "")[:1].upper(),
            "pred_value": pv,
            "target_class": g.get("target_class"), "target_size": g.get("target_size"),
            "bbox": g.get("bbox"), "filename": g.get("filename"),
            "raw_output": r.get("raw_output"),
        })
        print("%-14s qid=%-6s gt=%s(%s) pred=%s(%s)  %s" %
              (tag, qid, manifest[-1]["gt_letter"], gv,
               manifest[-1]["pred_letter"], pv, (g.get("question") or "")[:70]))

json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n导出 %d 张 -> %s" % (len(manifest), OUT))
