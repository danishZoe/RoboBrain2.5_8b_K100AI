#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核对 CV-Bench parquet 自带的 `prompt` 字段与我们构造的提示词是否一致。"""
import io
import json
import os
import sys

WS = "/root/private_data/workspace/robo"
sys.path.insert(0, WS + "/envs/pylibs_core")
sys.path.insert(0, WS + "/submission/bench_public")
import pyarrow.parquet as pq
import common as C

CVDIR = WS + "/third_party/benchmarks/CV-Bench"
cols = ["idx", "type", "task", "question", "choices", "answer", "prompt"]

n = 0
same = 0
diff = 0
samples = []
for name in ("test_2d.parquet", "test_3d.parquet"):
    path = os.path.join(CVDIR, name)
    if not os.path.exists(path):
        continue
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=64, columns=cols):
        for row in batch.to_pylist():
            n += 1
            p_field = row.get("prompt")
            ours = C.prompt_cv_bench_flag_eval_mm(row["question"], row["choices"])
            if p_field is None:
                diff += 1
                if len(samples) < 3:
                    samples.append(("None", row, ours))
                continue
            # ours 末尾带 "<think></think><answer>"（官方适配器后缀），去掉后与数据卡字段比
            ours_core = ours.replace("<think></think><answer>", "")
            if p_field.strip() == ours_core.strip():
                same += 1
            else:
                diff += 1
                if len(samples) < 6:
                    samples.append(("DIFF", row, ours))

print("总条数 = %d   与数据卡 prompt 字段一致 = %d   不一致/缺失 = %d" % (n, same, diff))
for tag, row, ours in samples:
    print("\n===== %s  idx=%s task=%s =====" % (tag, row["idx"], row.get("task")))
    print("question : %r" % row["question"])
    print("choices  : %r" % row["choices"])
    print("answer   : %r" % row["answer"])
    print("数据卡 prompt :\n%s" % row.get("prompt"))
    print("我们构造的（去适配器后缀）:\n%s" % ours.replace("<think></think><answer>", ""))
