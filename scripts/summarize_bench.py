#!/usr/bin/env python3
"""Summarise the six-tier service load benchmark.

Reads every ``outputs/phase5_bench/*/bench_results.json`` chunk, prints a
Markdown table in the official column layout and writes it to a file so the
report and the submission README can quote the same numbers.

Usage (remote):
  python3 scripts/summarize_bench.py --out reports/bench_summary.md
"""

import argparse
import glob
import json
import os

WORK = os.environ.get("WORK_DIR", "/root/private_data/workspace/robo")

COLS = [
    ("random_input_len", "输入 token 档"),
    ("random_output_len", "输出 token 档"),
    ("max_concurrency", "请求并发"),
    ("completed", "完成"),
    ("failed", "失败"),
    ("duration", "时长 s"),
    ("request_throughput", "请求吞吐 req/s"),
    ("output_throughput", "输出吞吐 tok/s"),
    ("total_throughput", "总吞吐 tok/s"),
    ("mean_ttft_ms", "TTFT 均值 ms"),
    ("p50_ttft_ms", "TTFT P50 ms"),
    ("p95_ttft_ms", "TTFT P95 ms"),
    ("mean_tpot_ms", "TPOT 均值 ms"),
    ("mean_e2e_latency_ms", "端到端均值 ms"),
    ("p95_e2e_latency_ms", "端到端 P95 ms"),
    ("avg_prompt_tokens", "实际输入 token（含视觉）"),
    ("avg_output_tokens", "实际输出 token"),
]


def load_rows(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for rec in data:
            tier = rec.get("tier", {})
            rec = dict(rec)
            rec["_source"] = os.path.basename(os.path.dirname(path))
            rec["_input_len"] = tier.get("input_tokens")
            rec["_output_len"] = tier.get("output_tokens")
            rec["_conc"] = tier.get("concurrency")
            rec["_requests"] = tier.get("requests")
            rows.append(rec)
    # keep only the recorded (second) round when both are present
    best = {}
    for rec in rows:
        key = (rec["_input_len"], rec["_output_len"], rec["_conc"],
               rec.get("round"))
        best[key] = rec
    return list(best.values())


def sort_key(rec):
    return (rec.get("_input_len") or 0, rec.get("_conc") or 0,
            rec.get("round") or 0)


def render(rec):
    tier = rec.get("tier", {})
    vals = {
        "random_input_len": tier.get("input_tokens"),
        "random_output_len": tier.get("output_tokens"),
        "max_concurrency": tier.get("concurrency"),
        "completed": rec.get("completed"),
        "failed": rec.get("failed"),
        "duration": rec.get("wall_seconds"),
        "request_throughput": rec.get("request_throughput_rps"),
        "output_throughput": rec.get("output_token_throughput_tps"),
        "total_throughput": rec.get("total_token_throughput_tps"),
        "mean_ttft_ms": _ms(rec.get("ttft_seconds_mean")),
        "p50_ttft_ms": _ms(rec.get("ttft_seconds_p50")),
        "p95_ttft_ms": _ms(rec.get("ttft_seconds_p95")),
        "mean_tpot_ms": _ms(rec.get("tpot_seconds_mean")),
        "mean_e2e_latency_ms": _ms(rec.get("e2e_seconds_mean")),
        "p95_e2e_latency_ms": _ms(rec.get("e2e_seconds_p95")),
        "avg_prompt_tokens": rec.get("avg_prompt_tokens"),
        "avg_output_tokens": rec.get("avg_output_tokens"),
    }
    return vals


def _ms(seconds):
    if seconds is None:
        return None
    return round(seconds * 1000.0, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default=os.path.join(
        WORK, "outputs", "phase5_bench", "*", "bench_results.json"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = sorted(load_rows(args.pattern), key=sort_key)
    if not rows:
        print("no bench results found under %s" % args.pattern)
        return 1

    lines = []
    header = ["轮次", "档位(in/out/conc/请求数)"] + [c[1] for c in COLS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + " --- |" * len(header))
    for rec in rows:
        vals = render(rec)
        cells = [
            "R%d" % rec.get("round", 0),
            "%s/%s/%s/%s" % (rec.get("_input_len"), rec.get("_output_len"),
                             rec.get("_conc"), rec.get("_requests")),
        ]
        for key, _ in COLS:
            v = vals.get(key)
            cells.append("-" if v is None else
                         ("%.2f" % v if isinstance(v, float) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")

    # scaling summary on the recorded (second) rounds
    recorded = [r for r in rows if (r.get("round") or 0) != 99
                and (r.get("round") or 0) != 1]
    if not recorded:
        recorded = rows
    lines.append("")
    lines.append("**并发扩展（取每档记录轮）**")
    lines.append("")
    lines.append("| 输入 token 档 | 并发 | 输出吞吐 tok/s | 相对并发 1 |")
    lines.append("| --- | --- | --- | --- |")
    base = {}
    for rec in sorted(recorded, key=sort_key):
        tps = rec.get("output_token_throughput_tps")
        if tps is None:
            continue
        ilen, conc = rec.get("_input_len"), rec.get("_conc")
        if conc == 1:
            base[ilen] = tps
        mult = ("%.2f×" % (tps / base[ilen])) if base.get(ilen) else "-"
        lines.append("| %s | %s | %.2f | %s |" % (ilen, conc, tps, mult))

    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print("\nwritten -> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
