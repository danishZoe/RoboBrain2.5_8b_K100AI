#!/usr/bin/env python3
"""把完整压测结果回填到报告与提交包 README 的占位符位置。

回填位置：
  reports/25_tuning_delivery_20261031.md
      <!-- FULL_BENCH_TABLE -->   ← 最优版六档表（官方 13 列）
      <!-- SAME_NODE_AB -->       ← 同节点 A/B（graph 基线 vs 最优版）
  submission/README.md
      <!-- BENCH_SUMMARY:BEGIN --> ... <!-- BENCH_SUMMARY:END -->  ← 换成最优版表
      并在其后追加「与首提基线对比」小节

用法：
    python3 scripts/fill_tuning_tables.py --run-dir outputs/fullbench_candidate_xxx
"""
import argparse
import json
import os
import re
from pathlib import Path

WS = Path(os.environ.get("ROBO_WS", "/root/private_data/workspace/robo"))
ORDER = [(4096, 1), (4096, 8), (4096, 64), (16384, 1), (16384, 8), (16384, 64)]

# graph 基线（stock 库 + graph，默认 attention）在本节点的两个来源：
#   * noneager_confirm  → 4096/c1、4096/c8（首轮 A/B）
#   * fullbench_baseline → 16384/c1、16384/c8、4096/c64（候选压测后链式补测）
BASELINE_1031 = [
    "outputs/phase5_ab_noneager/noneager_confirm_20261031_222226",
    "outputs/fullbench_baseline_20261101_043223",
]
FIRST_SUBMISSION = "submission/bench"
FIRST_SUBMISSION_1031 = "logs/clean_repro_full/clean_repro_full_20261031/bench"

COLS = [
    "random_input_len", "random_output_len", "max_concurrency", "completed", "duration",
    "request_throughput", "output_throughput", "total_throughput",
    "mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms", "mean_e2e_latency_ms",
]
HEADER = ("| random_input_len | random_output_len | max_concurrency | completed | duration | "
          "request_throughput | output_throughput | total_throughput | mean_ttft_ms | "
          "p99_ttft_ms | mean_tpot_ms | p99_tpot_ms | mean_e2e_latency_ms |")


def load_dir(d: Path):
    rows = {}
    if not d.exists():
        return rows
    for f in sorted(d.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (r.get("random_input_len"), r.get("max_concurrency"))
            rows.setdefault(key, []).append(r)
    return rows


def fmt_val(v):
    """官方 13 列表的数值格式：整数列原样，浮点列保留 2 位小数，不用科学计数法。"""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def missing_cells_note(rows):
    """若某些档位缺少官方 13 列中的字段（节点不可达时可能只有部分指标留存），
    生成一条明确的缺口说明，避免读者把 '—' 误读成 0。"""
    gaps = {}
    for key in ORDER:
        rs = rows.get(key)
        if not rs:
            continue
        r = rs[-1]
        miss = [c for c in COLS if r.get(c) is None]
        if miss:
            gaps[key] = miss
    if not gaps:
        return []
    lines = [
        "> **缺口说明**：`—` 表示该字段在本次数据来源中未被留存，**不是 0**。"
        "以下档位存在未留存字段：",
        "",
    ]
    name = {(4096, 1): "4096/并发1", (4096, 8): "4096/并发8", (4096, 64): "4096/并发64",
            (16384, 1): "16384/并发1", (16384, 8): "16384/并发8", (16384, 64): "16384/并发64"}
    for key, miss in gaps.items():
        lines.append(f"> - {name.get(key, key)}：{', '.join(miss)}")
    lines.append(">")
    lines.append("> 原因与影响见 §9；吞吐、TTFT、TPOT、时长、完成数、失败数等核心指标齐全。")
    return lines


def fmt_official(rows):
    lines = [HEADER, "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key in ORDER:
        rs = rows.get(key)
        if not rs:
            lines.append(f"| {key[0]} | 1024 | {key[1]} | - | - | - | - | - | - | - | - | - | - |")
            continue
        r = rs[-1]  # 记录轮（最后一轮）
        vals = [fmt_val(r.get(c)) for c in COLS]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def fmt_rounds(rows):
    lines = ["| 输入 | 并发 | 轮数 | 输出吞吐 tok/s（各轮） | 轮间极差 | 失败 |",
             "| --- | ---: | ---: | --- | ---: | ---: |"]
    for key in ORDER:
        rs = rows.get(key)
        if not rs:
            continue
        tp = [r.get("output_throughput", 0) for r in rs]
        spread = (max(tp) - min(tp)) / min(tp) * 100 if len(tp) > 1 and min(tp) else 0.0
        failed = sum(r.get("failed", 0) for r in rs)
        lines.append(f"| {key[0]} | {key[1]} | {len(rs)} | {', '.join(f'{v:.2f}' for v in tp)} | "
                     f"{spread:.2f}% | {failed} |")
    return "\n".join(lines)


def fmt_ab(base_rows, cand_rows):
    lines = ["| 输入 | 并发 | graph 基线 tok/s | 最优版 tok/s | 吞吐提升 | TTFT 变化 | TPOT 变化 | 基线轮/候选轮 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for key in ORDER:
        b, c = base_rows.get(key), cand_rows.get(key)
        if not b or not c:
            continue
        bt = sum(r.get("output_throughput", 0) for r in b) / len(b)
        ct = sum(r.get("output_throughput", 0) for r in c) / len(c)
        btt = sum(r.get("mean_ttft_ms", 0) for r in b) / len(b)
        ctt = sum(r.get("mean_ttft_ms", 0) for r in c) / len(c)
        btp = sum(r.get("mean_tpot_ms", 0) for r in b) / len(b)
        ctp = sum(r.get("mean_tpot_ms", 0) for r in c) / len(c)
        lines.append(
            f"| {key[0]} | {key[1]} | {bt:.2f} | {ct:.2f} | {(ct/bt-1)*100:+.2f}% | "
            f"{(ctt/btt-1)*100:+.1f}% | {(ctp/btp-1)*100:+.1f}% | {len(b)}/{len(c)} |")
    return "\n".join(lines)


def fmt_vs_first(first_rows, cand_rows, title="首提基线"):
    lines = [f"| 输入 | 并发 | {title} tok/s | 最优版 tok/s | 提升 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for key in ORDER:
        b, c = first_rows.get(key), cand_rows.get(key)
        if not b or not c:
            continue
        bt = sum(r.get("output_throughput", 0) for r in b) / len(b)
        ct = sum(r.get("output_throughput", 0) for r in c) / len(c)
        lines.append(f"| {key[0]} | {key[1]} | {bt:.2f} | {ct:.2f} | {(ct/bt-1)*100:+.2f}% |")
    return "\n".join(lines)


def replace_marker(text: str, name: str, body: str) -> str:
    """在成对标记之间替换正文（标记本身保留，脚本可重复执行）。

    只匹配 BEGIN..END 之间的内容，**不会吃掉标记之外的正文**。
    """
    begin = f"<!-- {name}:BEGIN -->"
    end = f"<!-- {name}:END -->"
    if begin not in text or end not in text:
        raise SystemExit(f"ERROR: marker pair not found: {name}")
    pat = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.S)
    block = f"{begin}\n{body}\n{end}"
    return pat.sub(lambda _m: block, text, count=1)


def replace_block(text: str, body: str) -> str:
    pat = re.compile(r"<!-- BENCH_SUMMARY:BEGIN -->.*?<!-- BENCH_SUMMARY:END -->", re.S)
    if not pat.search(text):
        raise SystemExit("ERROR: BENCH_SUMMARY block not found")
    return pat.sub(f"<!-- BENCH_SUMMARY:BEGIN -->\n{body}\n<!-- BENCH_SUMMARY:END -->", text, count=1)


def replace_cmp(text: str, body: str) -> str:
    pat = re.compile(r"<!-- TUNING_COMPARISON:BEGIN -->.*?<!-- TUNING_COMPARISON:END -->", re.S)
    block = f"<!-- TUNING_COMPARISON:BEGIN -->\n{body}\n<!-- TUNING_COMPARISON:END -->"
    if pat.search(text):
        return pat.sub(lambda _m: block, text, count=1)
    anchor = "<!-- BENCH_SUMMARY:END -->"
    if anchor not in text:
        raise SystemExit("ERROR: BENCH_SUMMARY:END not found")
    return text.replace(anchor, anchor + "\n\n" + block, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=WS / "reports/25_tuning_delivery_20261031.md")
    ap.add_argument("--readme", type=Path, default=WS / "submission/README.md")
    args = ap.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        cands = sorted((WS / "outputs").glob("fullbench_candidate_*"))
        cands = [c for c in cands if any(c.glob("*.jsonl"))]
        if not cands:
            raise SystemExit("ERROR: no fullbench_candidate_* run dir with results")
        run_dir = cands[-1]
    print(f"[fill] run dir: {run_dir}")

    cand = load_dir(run_dir)
    if not cand:
        raise SystemExit(f"ERROR: no result rows in {run_dir}")
    base = {}
    for d in BASELINE_1031:
        for k, v in load_dir(WS / d).items():
            base.setdefault(k, []).extend(v)
    first = load_dir(WS / FIRST_SUBMISSION)
    first31 = load_dir(WS / FIRST_SUBMISSION_1031)

    official = fmt_official(cand)
    rounds = fmt_rounds(cand)
    ab = fmt_ab(base, cand)
    vs_first = fmt_vs_first(first31, cand, "首提基线 eager（同节点，10-31）")

    full_block = "\n".join([
        "**最优版（提交配置）六档实测结果**（官方 13 列格式，取记录轮）：",
        "",
        official,
        "",
    ] + missing_cells_note(cand) + [
        "各档轮次与波动：",
        "",
        rounds,
        "",
        "> 原始数据：`outputs/{RUN_DIR}/`（每档 `round*.jsonl` 与同名 `.log`）；",
        "> 驱动脚本 `scripts/run_full_bench.sh`，汇总 `scripts/summarize_full_bench.py`。",
        "> 同一配置在 10-30 节点上的绝对值更高（该节点整体快约 30%），",
        "> 跨节点绝对值不可直接比较，见 §6 节点差异警告。",
    ]).replace("{RUN_DIR}", run_dir.name)

    # --- reports/25 ---
    txt = args.report.read_text(encoding="utf-8")
    sections_before = txt.count("\n## ")
    txt = replace_marker(txt, "FULL_BENCH_TABLE", full_block)
    txt = replace_marker(txt, "SAME_NODE_AB", ab)
    sections_after = txt.count("\n## ")
    if sections_after < sections_before:
        raise SystemExit(
            f"ERROR: 章节数减少（{sections_before} -> {sections_after}），已中止写入。"
            "请检查 replace_marker 的正则是否越界。")
    args.report.write_text(txt, encoding="utf-8")
    print(f"[fill] updated {args.report} (sections {sections_before} -> {sections_after})")

    # --- submission/README.md ---
    txt = args.readme.read_text(encoding="utf-8")
    txt = replace_block(txt, official)
    cmp_block = "\n".join([
        "### 7.1 与基线的同口径对比",
        "",
        "**同节点 A/B（graph 基线 vs 最优版，2026-10-31 同一节点、同一负载）**",
        "",
        ab,
        "",
        "**同节点对比首提基线（eager vs 最优版，2026-10-31 同一节点、同一负载；赛事口径正式对比）**",
        "",
        vs_first,
        "",
        "> 首提基线在 **2026-10-27 节点** 的原始六档记录保留在 `bench/round1.jsonl` /",
        "> `bench/round2.jsonl`（官方口径基准，见第 8 节的节点差异说明）；",
        "> 最优版的原始数据在 `bench/tuned/`。逐档对比表见 `../reports/tuning_comparison.md`，",
        "> 完整交付报告见 `../reports/25_tuning_delivery_20261031.md`。",
    ])
    txt = replace_cmp(txt, cmp_block)
    args.readme.write_text(txt, encoding="utf-8")
    print(f"[fill] updated {args.readme}")

    print("\n=== 同节点 A/B ===")
    print(ab)
    print("\n=== 相对首提基线 ===")
    print(vs_first)


if __name__ == "__main__":
    main()
