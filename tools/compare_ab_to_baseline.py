#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compare_ab_to_baseline.py - 把一次 A/B 实验的结果与官方压测基线逐档对齐比较。

用于 E1（非 eager）/ E2（max-num-batched-tokens）等单变量实验：
控制组 = `submission/bench/round2.jsonl`（记录轮），实验组 = 任意 jsonl。

    python3 tools/compare_ab_to_baseline.py --ab outputs/phase5_ab_noneager/<tag>/ab_*.jsonl
    python3 tools/compare_ab_to_baseline.py --ab <file.jsonl> --baseline submission/bench/round2.jsonl

输出 Markdown 表（可直接贴进报告）。
"""
import argparse
import glob
import json
import os
from pathlib import Path
import sys

METRICS = ["completed", "duration", "output_throughput", "total_throughput",
           "mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms",
           "mean_e2e_latency_ms"]



_BASE = Path(__file__).resolve().parent.parent


def _resolve(rel):
    """解析相对路径到实际存在的位置。

    优先字面路径 <base>/rel（节点工作目录布局：submission/...），
    再回落去掉 ``submission/`` 前缀（仓库提交视图布局：提交包已上提到根）。
    保证节点侧行为不变，克隆后的仓库也能跑。
    """
    rel = str(rel)
    p = _BASE / rel
    if p.exists():
        return p
    if rel.startswith("submission/"):
        alt = _BASE / rel[len("submission/"):]
        if alt.exists():
            return alt
    return p
def load(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", nargs="+", required=True,
                    help="实验组 jsonl（可给 glob）")
    ap.add_argument("--baseline", default=str(_resolve("submission/bench/round2.jsonl")))
    ap.add_argument("--label", default="实验组")
    args = ap.parse_args()

    base = {}
    for r in load(args.baseline):
        base[(r.get("random_input_len"), r.get("max_concurrency"))] = r

    files = []
    for a in args.ab:
        files.extend(sorted(glob.glob(a)) or [a])

    rows = []
    for p in files:
        for r in load(p):
            tier = (r.get("random_input_len"), r.get("max_concurrency"))
            b = base.get(tier)
            rows.append((os.path.basename(p), tier, b, r))

    if not rows:
        print("no rows")
        return 2

    print("| 来源 | 档位(in/conc) | completed | failed | duration s | 输出吞吐 tok/s | vs 基线 | 平均TTFT ms | vs 基线 | 平均TPOT ms | vs 基线 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, tier, b, r in rows:
        def dev(key):
            if not b:
                return "-"
            bv, cv = b.get(key), r.get(key)
            if not isinstance(bv, (int, float)) or not bv:
                return "-"
            return "%+.1f%%" % ((cv - bv) / bv * 100.0)
        print("| %s | %s/%s | %s | %s | %.2f | %.2f | %s | %.2f | %s | %.2f | %s |" % (
            name, tier[0], tier[1], r.get("completed"), r.get("failed", 0),
            r.get("duration", 0), r.get("output_throughput", 0),
            dev("output_throughput"), r.get("mean_ttft_ms", 0), dev("mean_ttft_ms"),
            r.get("mean_tpot_ms", 0), dev("mean_tpot_ms")))
    print()
    print("基线：%s（记录轮）" % args.baseline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
