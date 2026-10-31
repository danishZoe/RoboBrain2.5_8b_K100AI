#!/usr/bin/env python3
"""汇总完整压测结果：按官方 13 列口径输出 Markdown 表。

用法:
    python3 scripts/summarize_full_bench.py --run-dir outputs/fullbench_candidate_xxx --config candidate
    python3 scripts/summarize_full_bench.py --run-dir ... --ref-dir outputs/xxx （对照目录，按同档位比较）

输出:
    <run-dir>/summary.md
"""
import argparse
import json
import re
from pathlib import Path

COLUMNS = [
    ("random_input_len", "random_input_len"),
    ("random_output_len", "random_output_len"),
    ("max_concurrency", "max_concurrency"),
    ("completed", "completed"),
    ("duration", "duration"),
    ("request_throughput", "request_throughput"),
    ("output_throughput", "output_throughput"),
    ("total_throughput", "total_throughput"),
    ("mean_ttft_ms", "mean_ttft_ms"),
    ("p99_ttft_ms", "p99_ttft_ms"),
    ("mean_tpot_ms", "mean_tpot_ms"),
    ("p99_tpot_ms", "p99_tpot_ms"),
    ("mean_e2e_latency_ms", "mean_e2e_latency_ms"),
]


def load_rows(run_dir: Path):
    rows = []
    for path in sorted(run_dir.glob("*.jsonl")):
        if not re.match(r"round\d+_", path.name):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append((path.name, json.loads(line)))
            except json.JSONDecodeError:
                continue
    return rows


def key_of(row):
    return (row.get("random_input_len"), row.get("max_concurrency"))


def fmt_table(rows):
    lines = [
        "| random_input_len | random_output_len | max_concurrency | completed | failed | duration | "
        "request_throughput | output_throughput | total_throughput | mean_ttft_ms | p99_ttft_ms | "
        "mean_tpot_ms | p99_tpot_ms | mean_e2e_latency_ms |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _src, r in rows:
        vals = []
        for field, _label in COLUMNS:
            v = r.get(field)
            vals.append("—" if v is None else (f"{v:g}" if isinstance(v, (int, float)) else str(v)))
        failed = r.get("failed", 0)
        vals.insert(4, "—" if failed is None else str(failed))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--config", default="")
    ap.add_argument("--ref-dir", type=Path, default=None)
    args = ap.parse_args()

    rows = load_rows(args.run_dir)
    out = []
    out.append(f"# 完整压测汇总（{args.config or args.run_dir.name}）")
    out.append("")
    out.append(f"- 运行目录：`{args.run_dir}`")
    out.append(f"- 结果条数：{len(rows)}")
    out.append("")
    if not rows:
        out.append("**无结果行。**")
    else:
        out.extend(fmt_table(rows))
        out.append("")

        # 同档位多轮波动
        by_key = {}
        for _src, r in rows:
            by_key.setdefault(key_of(r), []).append(r)
        out.append("## 轮次与波动")
        out.append("")
        out.append("| 输入 | 并发 | 轮数 | 输出吞吐 tok/s | 轮间极差 |")
        out.append("| --- | --- | ---: | --- | --- |")
        for k in sorted(by_key, key=lambda x: (x[0] or 0, x[1] or 0)):
            vals = [r.get("output_throughput", 0) for r in by_key[k]]
            spread = (max(vals) - min(vals)) / min(vals) * 100 if len(vals) > 1 and min(vals) else 0.0
            txt = ", ".join(f"{v:.2f}" for v in vals)
            out.append(f"| {k[0]} | {k[1]} | {len(vals)} | {txt} | {spread:.2f}% |")
        out.append("")

        if args.ref_dir:
            ref_rows = load_rows(args.ref_dir)
            ref_by_key = {}
            for _src, r in ref_rows:
                ref_by_key.setdefault(key_of(r), []).append(r)
            out.append(f"## 与对照目录 `{args.ref_dir.name}` 的同档位对比")
            out.append("")
            out.append("| 输入 | 并发 | 本次 tok/s | 对照 tok/s | 偏差 | 本次 TTFT ms | 对照 TTFT ms | TTFT 偏差 |")
            out.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for k in sorted(by_key, key=lambda x: (x[0] or 0, x[1] or 0)):
                cand = by_key[k]
                base = ref_by_key.get(k)
                if not base:
                    out.append(f"| {k[0]} | {k[1]} | {cand[0].get('output_throughput', 0):.2f} | — | — | — | — | — |")
                    continue
                c = sum(r.get("output_throughput", 0) for r in cand) / len(cand)
                b = sum(r.get("output_throughput", 0) for r in base) / len(base)
                ctt = sum(r.get("mean_ttft_ms", 0) for r in cand) / len(cand)
                btt = sum(r.get("mean_ttft_ms", 0) for r in base) / len(base)
                dev = (c / b - 1) * 100 if b else 0.0
                tdev = (ctt / btt - 1) * 100 if btt else 0.0
                out.append(
                    f"| {k[0]} | {k[1]} | {c:.2f} | {b:.2f} | {dev:+.2f}% | {ctt:.1f} | {btt:.1f} | {tdev:+.2f}% |"
                )
            out.append("")

    (args.run_dir / "summary.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
