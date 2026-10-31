#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用当前 score_3d_chunked.merge_chunks 重算已落盘分片结果的合并指标。

背景
----
生产跑（``tools/run_ts_score_3d.sh``）用的是当时的 merge 实现；本工具从 JSON 里
已保存的 ``per_chunk``（每片的官方 stdout 原样解析值）**重算**合并指标，
使「产物口径」与「验证口径」使用同一份 merge 代码。

同时保留原合并值为 ``merged_legacy``，并逐指标对比差异 —— 若有差异会显式打印，
不做静默替换。

用法::

    python3 tools/remerge_ts_3d.py --bench-root outputs/bench_public
"""
from __future__ import annotations

import argparse
import json
import os
import sys

WS = "/root/private_data/workspace/robo"
sys.path.insert(0, os.path.join(WS, "submission/bench_public"))

import score_3d_chunked as S3  # noqa: E402

SCALARS = ("start_in_mask", "end_in_bbox", "start_distance", "end_distance",
           "no_collision_rate", "start_success_rate", "end_success_rate",
           "overall_success_rate")


def close(a, b, tol=1e-9):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root", default=os.path.join(WS, "outputs/bench_public"))
    ap.add_argument("--apply", action="store_true",
                    help="把重算结果写回 JSON（默认只对比不落盘）")
    args = ap.parse_args()

    n_changed = 0
    for variant in ("official", "upstream"):
        for task in ("2D", "3D"):
            p = os.path.join(args.bench_root, "tracespatial", variant,
                             "scores_3dchunk_%s.json" % task)
            if not os.path.exists(p):
                continue
            d = json.load(open(p, encoding="utf-8"))
            per_chunk = d.get("per_chunk") or []
            if not per_chunk:
                print("[remerge] %s/%s: 无 per_chunk，跳过" % (variant, task))
                continue
            cs = int(d.get("merge_method", "").split("=")[-1].split("（")[0]
                     .replace("（逐样本分片，官方 CLI 逐字调用）", "")
                     .split("（")[0]) if "chunk_size=" in d.get("merge_method", "") else 1
            new = S3.merge_chunks(per_chunk, cs)
            old = {k: d.get(k) for k in SCALARS}
            old.update({"metrics_2d": d.get("metrics_2d"),
                        "metrics_3d": d.get("metrics_3d")})
            diffs = []
            for k in SCALARS:
                if not close(old.get(k), new.get(k), 1e-9):
                    diffs.append((k, old.get(k), new.get(k)))
            for k in ("metrics_2d", "metrics_3d"):
                a, b = old.get(k) or {}, new.get(k) or {}
                for kk in sorted(set(a) | set(b)):
                    if not close(a.get(kk), b.get(kk), 1e-6):
                        diffs.append(("%s.%s" % (k, kk), a.get(kk), b.get(kk)))
            tag = "%s/%s" % (variant, task)
            if diffs:
                n_changed += 1
                print("[remerge] %s: **%d 项差异**" % (tag, len(diffs)))
                for k, a, b in diffs:
                    print("    %-28s 旧=%-14s 新=%-14s" % (k, a, b))
            else:
                print("[remerge] %s: 全部一致（%d 项标量 + 2 组字典）" % (tag, len(SCALARS)))
            if args.apply and diffs:
                d["merged_legacy"] = {k: d.get(k) for k in SCALARS}
                d["merged_legacy"]["metrics_2d"] = d.get("metrics_2d")
                d["merged_legacy"]["metrics_3d"] = d.get("metrics_3d")
                for k in SCALARS:
                    d[k] = new.get(k)
                d["metrics_2d"] = new.get("metrics_2d")
                d["metrics_3d"] = new.get("metrics_3d")
                d["audit"] = new.get("audit")
                d["merge_method"] = new.get("merge_method")
                with open(p, "w", encoding="utf-8") as fh:
                    json.dump(d, fh, ensure_ascii=False, indent=2)
                print("    -> 已写回 %s" % p)
    print("\n[remerge] 有差异的组合数 = %d" % n_changed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
