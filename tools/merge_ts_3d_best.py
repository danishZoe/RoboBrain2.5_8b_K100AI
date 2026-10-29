#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TraceSpatial 3D 「最大可行覆盖」合并：96 口径 + 可完成补跑分片。

根因（2026-10-29  py-spy + 复现实验证实）：
官方 summarize_acc.py 的 3D 路径对**含 NaN/inf 的插值轨迹**调用递归版
discrete_frechet_distance；其备忘录判据 ``ca[i,j] > -0.5`` 对 NaN 永假 ->
指数级递归 -> **永不终止**（非 DBSCAN 慢；DBSCAN/目标点云链路实测 0.0s）。
NaN 来源：模型对深景样本输出**恒定退化轨迹**（如 56/78：9 点仅 2 个唯一点），
距离均匀插值时重复节点使 interp1d 除零产生 NaN。
2D 路径有 NaN 过滤，3D 路径没有 —— 属官方上游缺陷。

因此 100/100 全量官方判分**对所有变体组合都不可达**（每个组合恰有 2 条样本
会使其预测触发不终止）。本脚本对每个 (run, variant) 合并：
  * 96 口径的 96 个 per_chunk（既有产物）
  * 该组合下「可完成」样本的补跑 per_chunk（ts3d_4b / ts3d_4c 产物）
排除项逐组合记录 excluded_sample_indices（官方代码不终止，不可判）。
"""
from __future__ import annotations

import json
import os
import sys

WS = "/root/private_data/workspace/robo"
sys.path.insert(0, os.path.join(WS, "submission/bench_public"))
import score_3d_chunked as S3  # noqa: E402

SRC96 = os.path.join(WS, "outputs/bench_public_3d96")
SUPP = {  # (run, variant) -> [(补跑产物根, {样本索引})]
    ("bench_public", "official"): [(os.path.join(WS, "outputs/bench_public_ts3d4c"), {82, 98})],
    ("bench_public_noneager", "official"): [(os.path.join(WS, "outputs/bench_public_ts3d4b"), {82, 98})],
    ("bench_public", "upstream"): [(os.path.join(WS, "outputs/bench_public_ts3d4c"), {56, 78, 98})],
    ("bench_public_noneager", "upstream"): [(os.path.join(WS, "outputs/bench_public_ts3d4b"), {56, 78, 82})],
}
EXCLUDED = {  # 官方代码不终止（根因见文件头）
    ("bench_public", "official"): [56, 78],
    ("bench_public_noneager", "official"): [56, 78],
    ("bench_public", "upstream"): [82],
    ("bench_public_noneager", "upstream"): [98],
}
DST = os.path.join(WS, "outputs/bench_public_ts3d_best")
ALL_IDX = [56, 78, 82, 98]


def main() -> int:
    report = {}
    for run in ("bench_public", "bench_public_noneager"):
        for variant in ("official", "upstream"):
            p96 = os.path.join(SRC96, run, "tracespatial", variant, "scores_3dchunk_3D.json")
            d96 = json.load(open(p96, encoding="utf-8"))
            kept96 = [i for i in range(100) if i not in ALL_IDX]
            pc = list(d96["per_chunk"])
            for c, i in zip(pc, kept96):
                c["sample_index"] = i
            n96 = len(pc)

            supp_idx = []
            for root, idxs in SUPP.get((run, variant), []):
                p = os.path.join(root, run, "tracespatial", variant, "scores_3dchunk_3D.json")
                d = json.load(open(p, encoding="utf-8"))
                # 补跑分片顺序与 KEEP 过滤顺序一致（按 [56,78,82,98] 子序列）
                order = [i for i in ALL_IDX if i in idxs]
                for c, i in zip(d["per_chunk"], order):
                    c["sample_index"] = i
                    pc.append(c)
                supp_idx.extend(order)

            combined = sorted(pc, key=lambda c: c["sample_index"])
            merged = S3.merge_chunks(combined, 1)
            merged["per_chunk"] = combined
            excl = EXCLUDED[(run, variant)]
            merged["excluded_sample_indices"] = excl
            merged["n_excluded_samples"] = len(excl)
            merged["exclusion_root_cause"] = (
                "official 3D path: recursive discrete_frechet_distance memoization "
                "breaks on NaN/inf interpolated trajectories (degenerate constant "
                "model predictions on deep-scene samples) -> never terminates; "
                "verified via py-spy stack + reproduction (reports/20)")
            a = merged.get("audit", {})
            out = os.path.join(DST, run, "tracespatial", variant)
            os.makedirs(out, exist_ok=True)
            op = os.path.join(out, "scores_3dchunk_3D.json")
            json.dump(merged, open(op, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            key = "%s/%s" % (run, variant)
            report[key] = {
                "n_chunks_96": n96, "supplementary_samples": supp_idx,
                "excluded": excl,
                "n_answers": a.get("n_answers_total"),
                "n_evaluated": a.get("n_evaluated_total"),
                "start": merged.get("start_success_rate"),
                "end": merged.get("end_success_rate"),
                "overall": merged.get("overall_success_rate"),
                "no_collision": merged.get("no_collision_rate"),
                "metrics_3d": merged.get("metrics_3d"),
            }
            print("%-40s 96=%d supp=%s excl=%s | answers=%s eval=%s start=%s end=%s overall=%s" % (
                key, n96, supp_idx, excl, a.get("n_answers_total"),
                a.get("n_evaluated_total"), merged.get("start_success_rate"),
                merged.get("end_success_rate"), merged.get("overall_success_rate")))
    rp = os.path.join(DST, "best_effort_report.json")
    json.dump(report, open(rp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("report ->", rp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
