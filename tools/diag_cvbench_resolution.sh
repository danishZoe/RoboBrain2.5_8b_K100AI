#!/usr/bin/env bash
# 诊断：CV-Bench Count 子任务的图像尺寸 / 视觉 token 预算（当前路径 vs 官方 FlagEvalMM 路径）
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONPATH="$WS/envs/pylibs_core:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore

python3 - <<'PY'
import os, io, json, math, collections, statistics
from PIL import Image
import pyarrow.parquet as pq

sys_path = None
try:
    import sys
    sys.path.insert(0, "/root/private_data/workspace/robo/scripts")
    from image_canonical import canonicalize_bytes, _smart_resize
except Exception as e:
    print("image_canonical 导入失败:", e)
    raise SystemExit

D = "/root/private_data/workspace/robo/third_party/benchmarks/CV-Bench"
FACTOR_PROC = 32           # patch_size 16 * merge_size 2

def proc_size(w, h):
    """processor 的 smart_resize（factor=32）—— 官方 Qwen2VLImageProcessor 的行为"""
    hb = max(FACTOR_PROC, round(h / FACTOR_PROC) * FACTOR_PROC)
    wb = max(FACTOR_PROC, round(w / FACTOR_PROC) * FACTOR_PROC)
    return wb, hb

rows = []
for name in ("test_2d.parquet", "test_3d.parquet"):
    pf = pq.ParquetFile(os.path.join(D, name))
    for b in pf.iter_batches(batch_size=64,
                             columns=["idx","type","task","source","choices","answer","image","filename"]):
        for r in b.to_pylist():
            img = r.get("image") or {}
            raw = img.get("bytes") or b""
            if not raw:
                continue
            rows.append((r["task"], r["source"], r["filename"], raw))
            if len(rows) > 3000:
                break
        if len(rows) > 3000:
            break
    if len(rows) > 3000:
        break

by_task = collections.defaultdict(list)
for task, source, fn, raw in rows:
    by_task[task].append((source, fn, raw))

print("=== 各 task 的图像尺寸与两条路径的视觉 token 预算 ===")
for task, items in sorted(by_task.items()):
    ours_tok, offi_tok = [], []
    dims, nbig = [], 0
    for source, fn, raw in items[:120]:
        try:
            im = Image.open(io.BytesIO(raw))
            w, h = im.size
        except Exception:
            continue
        dims.append((w, h))
        # 官方路径：原始 PIL -> (encode_image JPEG q95) -> processor
        ow, oh = proc_size(w, h)
        offi_tok.append((ow // FACTOR_PROC) * (oh // FACTOR_PROC))
        # 当前路径：qwen_vl_utils canonical(28) -> processor
        try:
            can = canonicalize_bytes(raw)
            cim = Image.open(io.BytesIO(can))
            cw, ch = cim.size
        except Exception:
            cw, ch = w, h
        pw, ph = proc_size(cw, ch)
        ours_tok.append((pw // FACTOR_PROC) * (ph // FACTOR_PROC))
        if w * h > 501760:
            nbig += 1
    if not dims:
        continue
    px = sorted(w * h for w, h in dims)
    print("\n[%s]  n=%d (抽样 %d)" % (task, len(items), len(dims)))
    print("   原图像素 p50=%d p90=%d max=%d ; >501760 占比 %.1f%%" % (
        int(px[len(px)//2]), int(px[int(len(px)*0.9)]), px[-1],
        100.0*nbig/len(dims)))
    print("   官方路径视觉token p50=%d p90=%d max=%d" % (
        int(statistics.median(offi_tok)), int(sorted(offi_tok)[int(len(offi_tok)*0.9)]), max(offi_tok)))
    print("   当前路径视觉token p50=%d p90=%d max=%d" % (
        int(statistics.median(ours_tok)), int(sorted(ours_tok)[int(len(ours_tok)*0.9)]), max(ours_tok)))
    print("   尺寸样例:", dims[:4])
PY
