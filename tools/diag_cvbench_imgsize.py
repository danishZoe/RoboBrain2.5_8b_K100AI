#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核对：parquet 里的原始图像尺寸 vs manifest 里落盘的图像尺寸。"""
import io
import json
import os
import sys

WS = "/root/private_data/workspace/robo"
sys.path.insert(0, WS + "/envs/pylibs_core")
import pyarrow.parquet as pq
from PIL import Image

CVDIR = WS + "/third_party/benchmarks/CV-Bench"
MAPJ = WS + "/outputs/bench_public/manifest/cvbench_image_map.json"
man = json.load(open(MAPJ, encoding="utf-8"))
manmap = {str(r["question_id"]): r for r in man}

cols = ["idx", "type", "task", "source", "filename", "image"]
for name in ("test_2d.parquet", "test_3d.parquet"):
    path = os.path.join(CVDIR, name)
    if not os.path.exists(path):
        print("!! 缺", path)
        continue
    pf = pq.ParquetFile(path)
    print("=== %s ===" % name)
    n = 0
    sizes = {}
    for batch in pf.iter_batches(batch_size=64, columns=cols):
        for row in batch.to_pylist():
            qid = str(row["idx"])
            img = row["image"]
            try:
                im = Image.open(io.BytesIO(img["bytes"]))
                sz = im.size
                fmt = im.format
                mode = im.mode
            except Exception as e:
                sz, fmt, mode = None, None, repr(e)
            m = manmap.get(qid)
            msz = m.get("original_size") if m else None
            mf = m.get("format") if m else None
            agree = (msz == list(sz)) if (sz and msz) else None
            if n < 8:
                print("  idx=%-6s parquet_size=%-12s fmt=%-5s mode=%-5s | manifest=%-12s fmt=%-5s agree=%s  file=%s" %
                      (qid, sz, fmt, mode, msz, mf, agree, row["filename"]))
            sizes[(tuple(sz) if sz else None)] = sizes.get((tuple(sz) if sz else None), 0) + 1
            n += 1
    print("  样本数=%d  parquet 原图尺寸分布:" % n)
    for k, v in sorted(sizes.items(), key=lambda kv: -kv[1])[:12]:
        print("     %-18s %d" % (str(k), v))
    # manifest 落盘尺寸分布
    ms = {}
    for r in man:
        ms[tuple(r.get("original_size") or [])] = ms.get(tuple(r.get("original_size") or []), 0) + 1
    print("  manifest original_size 分布:")
    for k, v in sorted(ms.items(), key=lambda kv: -kv[1])[:12]:
        print("     %-18s %d" % (str(k), v))
    print()
