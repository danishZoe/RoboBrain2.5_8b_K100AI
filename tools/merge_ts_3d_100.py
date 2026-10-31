#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合并 96 条口径 + 4 条病态样本补跑 -> TraceSpatial 3D 全量 100 分母指标。

两个来源的 scores_3dchunk_3D.json 都带 per_chunk（每片官方 stdout 的解析值），
直接拼接后调用 score_3d_chunked.merge_chunks(..., chunk_size=1) 重算。
96 条口径的精确性此前已由「整集单进程 vs 分片合并」48/48 对拍锁死；
chunk-size=1 时逐样本分片合并是精确的（见 score_3d_chunked.py 文档）。

产物：outputs/bench_public_ts3d100/<src>/tracespatial/<variant>/scores_3dchunk_3D.json
      （含逐分片 sample_index 标注，原 96/4 产物不动）
"""
from __future__ import annotations

import json
import os
import sys

WS = "/root/private_data/workspace/robo"
sys.path.insert(0, os.path.join(WS, "submission/bench_public"))

import score_3d_chunked as S3  # noqa: E402

EXCLUDED = [56, 78, 82, 98]
SRC96 = os.path.join(WS, "outputs/bench_public_3d96")
SRC4 = os.path.join(WS, "outputs/bench_public_ts3d4")
DST = os.path.join(WS, "outputs/bench_public_ts3d100")


def main() -> int:
    for src in ("bench_public", "bench_public_noneager"):
        for variant in ("official", "upstream"):
            p96 = os.path.join(SRC96, src, "tracespatial", variant,
                               "scores_3dchunk_3D.json")
            p4 = os.path.join(SRC4, src, "tracespatial", variant,
                              "scores_3dchunk_3D.json")
            for p in (p96, p4):
                if not os.path.exists(p):
                    print("[merge100] 缺输入 %s" % p)
                    return 1
            d96 = json.load(open(p96, encoding="utf-8"))
            d4 = json.load(open(p4, encoding="utf-8"))
            pc96, pc4 = list(d96["per_chunk"]), list(d4["per_chunk"])
            if len(pc96) != 96 or len(pc4) != 4:
                print("[merge100] %s/%s 分片数异常: 96-run=%d 4-run=%d"
                      % (src, variant, len(pc96), len(pc4)))
                return 1
            # 标注样本下标：96-run 为 0..99 去掉 EXCLUDED 的升序；4-run 为 EXCLUDED 顺序
            idx96 = [i for i in range(100) if i not in EXCLUDED]
            for c, i in zip(pc96, idx96):
                c["sample_index"] = i
            for c, i in zip(pc4, EXCLUDED):
                c["sample_index"] = i
            combined = sorted(pc96 + pc4, key=lambda c: c["sample_index"])
            merged = S3.merge_chunks(combined, 1)
            merged["per_chunk"] = combined
            merged["n_excluded_samples"] = 0
            merged["excluded_sample_indices"] = []
            merged["merge_method"] = (merged.get("merge_method", "")
                                      + " + 病态样本补跑合并（100 分母）")
            a = merged.get("audit", {})
            assert a.get("n_answers_total") == 100, (src, variant, a)
            out = os.path.join(DST, src, "tracespatial", variant)
            os.makedirs(out, exist_ok=True)
            op = os.path.join(out, "scores_3dchunk_3D.json")
            json.dump(merged, open(op, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            print("[merge100] %-25s %-9s start=%-8s end=%-8s overall=%-8s "
                  "no_coll=%-8s eval=%d skip=%d ambiguous3d=%d"
                  % (src, variant,
                     merged.get("start_success_rate"),
                     merged.get("end_success_rate"),
                     merged.get("overall_success_rate"),
                     merged.get("no_collision_rate"),
                     a.get("n_evaluated_total", -1),
                     a.get("n_skipped_total", -1),
                     a.get("n_ambiguous_allzero_metric3d", -1)))
            print("           metrics_3d=%s"
                  % json.dumps(merged.get("metrics_3d"), ensure_ascii=False))
    print("[merge100] 全部完成 -> %s" % DST)
    return 0


if __name__ == "__main__":
    sys.exit(main())
