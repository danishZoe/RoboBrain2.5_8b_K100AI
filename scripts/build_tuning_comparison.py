#!/usr/bin/env python3
"""生成「首提基线 / graph 基线 / 最优版」的同口径对比表。

口径说明（必须一起读）：
  * 所有数值都是 vLLM serving + 官方 bench_serving 客户端（image 数据集，1×1080p 图/请求，
    输出 1024 token，seed 1，请求数 max(8, 2×并发)）实测。
  * 「节点」列标注数据采集日期：同一节点内的两列可直接比；跨节点的绝对值不可直接比
    （2026-10-31 实测同配置比 10-30 慢约 30%），跨节点只用于看同一配置的相对关系。

用法：
    python3 scripts/build_tuning_comparison.py --out reports/tuning_comparison.md
"""
import argparse
import json
import os
from pathlib import Path

WS = Path(os.environ.get("ROBO_WS", "/root/private_data/workspace/robo"))

SOURCES = [
    # (标签, 节点, 配置说明, [(文件, 说明)], 是否官方基线)
    ("首提基线 eager", "10-27", "首提提交配置：--enforce-eager + stock 库 + 默认 attention",
     ["submission/bench/round1.jsonl", "submission/bench/round2.jsonl"], "官方口径基线"),
    ("graph 基线", "10-30", "非 eager（graph）+ stock 库（= 首提基线之上只开 graph）",
     ["tmp/graph_bench/baseline_oldlib/round1.jsonl", "tmp/graph_bench/baseline_oldlib/round2.jsonl"], "同轮 A/B 对照"),
    ("调优库 E5", "10-30", "非 eager + GEMM 调优库（未开 aiter FA）",
     ["tmp/graph_bench/tuned_085/round1.jsonl", "tmp/graph_bench/tuned_085/round2.jsonl"], "同轮 A/B 对照"),
    ("调优库+aiterFA E7", "10-30", "非 eager + GEMM 调优库 + ROCM_AITER_FA（= 最终方案）",
     ["tmp/gate_aiterfa.jsonl"], "门槛档位"),
    ("graph 基线", "10-31", "非 eager + stock 库（当前节点复测；4096/c1、4096/c8 首轮 A/B，"
     "16384/c1、16384/c8、4096/c64 为候选压测后同节点链式补测）",
     ["outputs/phase5_ab_noneager/noneager_confirm_20261031_222226/ab_4096_c1.jsonl",
      "outputs/phase5_ab_noneager/noneager_confirm_20261031_222226/ab_4096_c8.jsonl",
      "outputs/fullbench_baseline_20261101_043223/round*.jsonl"], "同轮 A/B 对照"),
    ("首提基线 eager", "10-31", "首提提交配置（--enforce-eager + stock 库）在当前节点的复现",
     ["logs/clean_repro_full/clean_repro_full_20261031/bench/round1.jsonl",
      "logs/clean_repro_full/clean_repro_full_20261031/bench/round2.jsonl"], "同节点赛事口径对照"),
    ("最优版", "10-31", "非 eager + GEMM 调优库 + ROCM_AITER_FA（提交配置）",
     ["outputs/fullbench_candidate_20261031_231152/round*.jsonl"], "提交配置"),
]

ORDER = [(4096, 1), (4096, 8), (4096, 64), (16384, 1), (16384, 8), (16384, 64)]


def load(paths):
    """按 (输入长度, 并发) 聚合。

    同一文件里可能出现多个「同档位」的批次（不同轮次/不同批次压测追加写）。
    为了不把不同批次的数值混在一起，这里对**每个文件**只保留每个档位的
    第一次出现（文件内顺序即批次顺序），跨文件再聚合。
    """
    rows = {}
    for rel in paths:
        if "*" in rel:
            files = sorted(WS.glob(rel))
        else:
            p = WS / rel
            files = [p] if p.exists() else []
        for f in files:
            seen = set()
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (r.get("random_input_len"), r.get("max_concurrency"))
                if key in seen:
                    continue
                seen.add(key)
                rows.setdefault(key, []).append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=WS / "reports/tuning_comparison.md")
    args = ap.parse_args()

    lines = ["# 调优对比表：首提基线 / graph 基线 / 最优版", ""]
    lines.append("> 口径：单卡 K100AI（K500SM_AI / gfx928 / 63.98 GiB）/ vLLM 0.15.1+das.opt1.alpha.dtk2604 /")
    lines.append("> BF16 / TP=1 / `--gpu-memory-utilization 0.85 --max-model-len 32768` /")
    lines.append("> `--max-num-batched-tokens 4096 --max-num-seqs 64 --no-enable-prefix-caching` /")
    lines.append("> `--limit-mm-per-prompt '{\"image\":8}'`；官方 bench_serving 客户端（image 数据集，")
    lines.append("> 1×1080p 图/请求，输出 1024 token，seed 1，请求数 max(8, 2×并发)）。")
    lines.append(">")
    lines.append("> **节点差异警告**：2026-10-31 复测发现同一配置在当前节点比 10-30 慢约 30%")
    lines.append("> （同配置 4096/并发1：10-30 为 33.0~33.4 tok/s，10-31 为 22.4 tok/s；")
    lines.append("> graph 捕获列表与服务参数完全一致）。因此**跨节点的绝对值不可直接比较**，")
    lines.append("> 只有同一节点内的两列才构成同口径 A/B。")
    lines.append("")

    blocks = []
    for label, node, desc, paths, role in SOURCES:
        rows = load(paths)
        if not rows:
            continue
        blocks.append((label, node, desc, role, rows))

    # 1) 逐配置逐档表格
    lines.append("## 1. 各配置的逐档实测")
    lines.append("")
    lines.append("| 配置 | 节点 | 角色 | 输入 | 并发 | 轮数 | 输出吞吐 tok/s | TTFT ms | TPOT ms | 失败 |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |")
    for label, node, desc, role, rows in blocks:
        for key in ORDER:
            rs = rows.get(key)
            if not rs:
                continue
            tp = [r.get("output_throughput", 0) for r in rs]
            ttft = [r.get("mean_ttft_ms", 0) for r in rs]
            tpot = [r.get("mean_tpot_ms", 0) for r in rs]
            failed = sum(r.get("failed", 0) for r in rs)
            lines.append(
                f"| {label} | {node} | {role} | {key[0]} | {key[1]} | {len(rs)} | "
                f"{'/'.join(f'{v:.2f}' for v in tp)} | {'/'.join(f'{v:.0f}' for v in ttft)} | "
                f"{'/'.join(f'{v:.2f}' for v in tpot)} | {failed} |"
            )
    lines.append("")

    # 2) 同节点 A/B 提升
    lines.append("## 2. 同节点 A/B 提升（唯一可直接相减的两列）")
    lines.append("")

    def mean(rs, field):
        return sum(r.get(field, 0) for r in rs) / len(rs) if rs else 0.0

    # 10-30：graph 基线 → 调优库 E5 → +aiter FA E7
    base30 = load(SOURCES[1][3])
    tuned30 = load(SOURCES[2][3])
    e7 = load(SOURCES[3][3])
    if base30 and tuned30:
        lines.append("### 2.1 10-30：graph 基线 → GEMM 调优库（E5）")
        lines.append("")
        lines.append("| 输入 | 并发 | graph 基线 tok/s | 调优库 tok/s | 吞吐提升 | TTFT 变化 | TPOT 变化 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for key in ORDER:
            b, t = base30.get(key), tuned30.get(key)
            if not b or not t:
                continue
            bt, tt = mean(b, "output_throughput"), mean(t, "output_throughput")
            lines.append(
                f"| {key[0]} | {key[1]} | {bt:.2f} | {tt:.2f} | {(tt/bt-1)*100:+.2f}% | "
                f"{(mean(t,'mean_ttft_ms')/mean(b,'mean_ttft_ms')-1)*100:+.1f}% | "
                f"{(mean(t,'mean_tpot_ms')/mean(b,'mean_tpot_ms')-1)*100:+.1f}% |"
            )
        lines.append("")

    if base30 and e7:
        lines.append("### 2.1b 10-30：graph 基线 → 调优库 + aiter FA（E7，即最终方案）")
        lines.append("")
        lines.append("| 输入 | 并发 | graph 基线 tok/s | 最优版 tok/s | 吞吐提升 |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for key in ORDER:
            b, t = base30.get(key), e7.get(key)
            if not b or not t:
                continue
            bt, tt = mean(b, "output_throughput"), mean(t, "output_throughput")
            lines.append(f"| {key[0]} | {key[1]} | {bt:.2f} | {tt:.2f} | {(tt/bt-1)*100:+.2f}% |")
        lines.append("")

    # 10-31：graph 基线 → 最优版
    base31 = load(SOURCES[4][3])
    cand31 = load(SOURCES[6][3])
    if base31 and cand31:
        lines.append("### 2.2 10-31：graph 基线 → 最优版（提交配置）")
        lines.append("")
        lines.append("| 输入 | 并发 | graph 基线 tok/s | 最优版 tok/s | 吞吐提升 | 轮次 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
        for key in ORDER:
            b, t = base31.get(key), cand31.get(key)
            if not b or not t:
                continue
            bt, tt = mean(b, "output_throughput"), mean(t, "output_throughput")
            lines.append(
                f"| {key[0]} | {key[1]} | {bt:.2f} | {tt:.2f} | {(tt/bt-1)*100:+.2f}% | "
                f"基线 {len(b)} / 候选 {len(t)} |"
            )
        lines.append("")

    # 3) 同节点：首提基线（eager）→ 最优版（赛事口径的正式对比）
    first31 = load(SOURCES[5][3])
    if first31 and cand31:
        lines.append("### 2.3 同节点：首提基线（eager）→ 最优版（赛事口径正式对比）")
        lines.append("")
        lines.append("> 两者都在 **2026-10-31 同一节点、同一负载、同一客户端**下测得，")
        lines.append("> 只差配置（eager+stock 库 vs graph+调优库+aiter FA），是本轮唯一的完整同口径对比。")
        lines.append("")
        lines.append("| 输入 | 并发 | 首提基线 tok/s | 最优版 tok/s | 吞吐提升 | TTFT 变化 | TPOT 变化 | 基线轮/候选轮 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for key in ORDER:
            b, t = first31.get(key), cand31.get(key)
            if not b or not t:
                continue
            bt = mean(b, "output_throughput")
            tt = mean(t, "output_throughput")
            btt = mean(b, "mean_ttft_ms")
            ctt = mean(t, "mean_ttft_ms")
            btp = mean(b, "mean_tpot_ms")
            ctp = mean(t, "mean_tpot_ms")
            lines.append(
                f"| {key[0]} | {key[1]} | {bt:.2f} | {tt:.2f} | {(tt/bt-1)*100:+.2f}% | "
                f"{(ctt/btt-1)*100:+.1f}% | {(ctp/btp-1)*100:+.1f}% | {len(b)}/{len(t)} |")
        lines.append("")

    # 4) 跨节点存档：10-27 首提基线原始表（仅存档，不作为收益结论）
    first = load(SOURCES[0][3])
    if first:
        lines.append("### 2.4 跨节点存档：首提基线（2026-10-27 节点原始记录）")
        lines.append("")
        lines.append("> ⚠️ **该节点整体比 10-31 快约 30%**（同配置复测已证实），")
        lines.append("> 因此下表只能作为**官方口径基准的存档**，不能与 10-31 的最优版直接相减。")
        lines.append("> 收益结论请看 §2.1 / §2.2 / §2.3 的同节点对比。")
        lines.append("")
        lines.append("| 输入 | 并发 | 首提基线（10-27）tok/s | TTFT ms | TPOT ms |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for key in ORDER:
            b = first.get(key)
            if not b:
                continue
            lines.append(f"| {key[0]} | {key[1]} | {mean(b,'output_throughput'):.2f} | "
                         f"{mean(b,'mean_ttft_ms'):.0f} | {mean(b,'mean_tpot_ms'):.2f} |")
        lines.append("")

    out = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
