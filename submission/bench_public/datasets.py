#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三个公开基准的数据装载层。

只做**读取与字段映射**，不改写任何标注内容；判分所需的官方字段由各 runner
按官方脚本期望的结构拼装。

数据选择（验收要求 §3.3「Parquet 与原始目录二选一作为正式输入格式」）：

* **CV-Bench**：正式输入 = 本地 parquet（``test_2d.parquet`` + ``test_3d.parquet``）。
  图像内嵌在 parquet 的 ``image`` struct 里，按行组流式读取，不落盘副本。
* **RefSpatial-Bench**：正式输入 = 原始目录（``<split>/question.json`` +
  ``image/`` + ``mask/``）。parquet 只是同一份数据的另一种封装，不重复计题。
* **TraceSpatial-Bench**：正式输入 = 原始目录（``trajectory_dataset.json`` +
  ``raw_data/``）。HF 仓库里的 ``train-00000-of-00001.parquet`` 与
  ``trajectory_dataset.json`` 是同一份 100 条数据，不重复计题。
"""
from __future__ import annotations

import io
import json
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import common as C

BENCH_ROOT = os.path.join(C.WORK_DIR, "third_party/benchmarks")
CVBENCH_DIR = os.path.join(BENCH_ROOT, "CV-Bench")
REFSPATIAL_DIR = os.path.join(BENCH_ROOT, "RefSpatial-Bench")
TRACESPATIAL_DIR = os.path.join(BENCH_ROOT, "TraceSpatial-Bench")

# 官方数据卡上的来源标识（用于逐题记录与 manifest）
DATASET_SOURCES = {
    "CV-Bench": "hf://nyu-visionx/CV-Bench",
    "RefSpatial-Bench": "hf://BAAI/RefSpatial-Bench",
    "TraceSpatial-Bench": "hf://JingkunAn/TraceSpatial-Bench",
}


# --------------------------------------------------------------------------- #
# RefSpatial-Bench（RoboRefer）
# --------------------------------------------------------------------------- #
def load_refspatial(split: str) -> List[Dict[str, Any]]:
    """读取 ``<split>/question.json``，返回题目列表（原字段不改写）。"""
    qpath = os.path.join(REFSPATIAL_DIR, split, "question.json")
    with open(qpath, encoding="utf-8") as fh:
        questions = json.load(fh)
    out = []
    for q in questions:
        img = os.path.join(REFSPATIAL_DIR, split, q["rgb_path"])
        mask = os.path.join(REFSPATIAL_DIR, split, q["mask_path"])
        out.append({
            "sample_id": f"{split}/{q['id']}",
            "split": split,
            "image_paths": [img],
            "mask_path": mask,
            "fields": {
                "question_id": q["id"],
                "prompt": q["prompt"],
                "object_name": q["object"],
                "suffix": q["suffix"],
                "model_id": "RoboBrain2.5-8B-NV",
                "rgb_path": q["rgb_path"],
                "mask_path": q["mask_path"],
                "category": q.get("category"),
                "step": q.get("step"),
            },
            "task": "pointing",
        })
    return out


# --------------------------------------------------------------------------- #
# TraceSpatial-Bench（RoboTracer）
# --------------------------------------------------------------------------- #
def load_tracespatial(task_name: str) -> List[Dict[str, Any]]:
    """读取 ``trajectory_dataset.json``（官方 eval 的输入），返回题目列表。"""
    dpath = os.path.join(TRACESPATIAL_DIR, "trajectory_dataset.json")
    with open(dpath, encoding="utf-8") as fh:
        items = json.load(fh)
    out = []
    for it in items:
        img = os.path.join(TRACESPATIAL_DIR, "raw_data", it["image_path"])
        out.append({
            "sample_id": f"TraceSpatial/{it['id']}",
            "split": task_name,
            "image_paths": [img],
            "fields": {
                "question_id": it["id"],
                "final_prompt": None,          # 由 runner 填
                "image_path": it["image_path"],
                "gt_depth_path": it["gt_depth_path"],
                "mask_path": it["mask_path"],
                "gt_depth_intrinsics": it["gt_depth_intrinsics"],
                "prompt": it["prompt"],
                "target_object": it["target_object"],
                "destination|direction": it["destination|direction"],
                "trajectory": it["trajectory"],
                "bbox_center": it["bbox_center"],
                "bbox_extent": it["bbox_extent"],
                "bbox_rotation": it["bbox_rotation"],
                "model_prediction": None,      # 由 runner 填
            },
            "task": "trajectory",
            "target_object": it["target_object"],
            "destination": it["destination|direction"],
            "raw_prompt": it["prompt"],
        })
    return out


# --------------------------------------------------------------------------- #
# CV-Bench（FlagEvalMM）
# --------------------------------------------------------------------------- #
def iter_cvbench() -> Iterator[Tuple[Dict[str, Any], bytes]]:
    """流式产出 (annotation, image_bytes)。

    annotation 字段按 FlagEvalMM ``tasks/cv_bench/process.py`` 的映射构造：
      ``question_id`` = ``idx``；``answer`` = ``data["answer"][1]``（"(C)" -> "C"）；
      ``options`` = ``choices``；其余原字段保留。
    """
    import pyarrow.parquet as pq

    cols = ["idx", "type", "task", "question", "choices", "answer", "source",
            "source_dataset", "source_filename", "target_class", "target_size",
            "bbox", "filename", "image"]
    for name in ("test_2d.parquet", "test_3d.parquet"):
        path = os.path.join(CVBENCH_DIR, name)
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=64, columns=cols):
            for row in batch.to_pylist():
                ann = {
                    "question_id": str(row["idx"]),
                    "type_ori": row["type"],
                    "question_type": "multiple-choice",
                    "answer": (row["answer"][1] if isinstance(row["answer"], str)
                               and row["answer"].startswith("(")
                               else row["answer"]),
                    "options": list(row["choices"] or []),
                    "task": row["task"],
                    "question": row["question"],
                    "source": row["source"],
                    "source_dataset": row.get("source_dataset"),
                    "source_filename": row.get("source_filename"),
                    "target_class": row.get("target_class"),
                    "target_size": row.get("target_size"),
                    "bbox": row.get("bbox"),
                    "filename": row["filename"],
                    "parquet_file": name,
                }
                img = row.get("image") or {}
                yield ann, img.get("bytes") or b""


def load_cvbench(limit: int = 0) -> List[Dict[str, Any]]:
    """把 CV-Bench 流式读成 job 列表（图像字节留在内存里逐个发送）。"""
    out = []
    for ann, data in iter_cvbench():
        out.append({
            "sample_id": f"CVBench/{ann['question_id']}",
            "split": ann["type_ori"],                 # "2D" / "3D"
            "image_paths": [],
            "image_bytes": data,
            "fields": ann,
            "task": None,
        })
        if limit and len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------- #
# 图像几何（原图尺寸 vs 实际送入模型的尺寸）
# --------------------------------------------------------------------------- #
def image_geometry(path: Optional[str] = None,
                   data: Optional[bytes] = None) -> Dict[str, Any]:
    """返回原图尺寸与 smart_resize 后的实际送入尺寸（不编码、不重采样）。

    ``smart_resize`` 的参数与 ``image_canonical`` 保持一致（factor=28，
    max_pixels=16384*28^2），因此这里算出的就是服务实际会看到的尺寸。
    """
    from PIL import Image
    from image_canonical import (_smart_resize, IMAGE_MAX_TOKEN_NUM,
                                 SPATIAL_MERGE_SIZE, DEFAULT_PATCH_SIZE)

    if path is not None:
        with Image.open(path) as im:
            w, h = im.size
    else:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
    factor = DEFAULT_PATCH_SIZE * SPATIAL_MERGE_SIZE
    h_bar, w_bar = _smart_resize(h, w, factor=factor,
                                 max_pixels=IMAGE_MAX_TOKEN_NUM * factor ** 2,
                                 min_pixels=4 * factor ** 2)
    return {"original": [w, h], "sent": [w_bar, h_bar],
            "scale_x": round(w_bar / w, 6) if w else None,
            "scale_y": round(h_bar / h, 6) if h else None}
