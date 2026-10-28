#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Segment the vLLM server log into per-tier busy windows and recompute the
generation throughput from the engine's own counters.

Why this exists
---------------
The official benchmark client reports ``output_throughput`` from *planned*
output lengths (``output_lens``), i.e. ``random_output_len`` per request,
regardless of how many tokens the engine really produced. The client also
exposes ``total_output_tokens_retokenized`` (re-encoding of the decoded text),
and the server log exposes the engine's own generation counter.

This script rebuilds the truth from the server side:

    total_generated_tokens = sum(interval_avg_generation_throughput * interval)

so the official table can be annotated with a *measured* caveat instead of an
assumed one, and so the per-tier busy windows can be lined up with the
client-side runs (which are executed tier by tier, round 1 then round 2).

Usage:
    python3 tools/analyze_server_throughput.py --log submission/server.log
    python3 tools/analyze_server_throughput.py --log submission/server.log --json out.json
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta

# vLLM periodic stats line, e.g.
# INFO 10-27 13:52:52 [loggers.py:257] Engine 000: Avg prompt throughput: 1715.8
#   tokens/s, Avg generation throughput: 16.9 tokens/s, Running: 1 reqs,
#   Waiting: 0 reqs, GPU KV cache usage: 1.8%, Prefix cache hit rate: 0.0%, ...
PAT = re.compile(
    r"(?P<ts>\d\d-\d\d\s+\d\d:\d\d:\d\d).*?"
    r"Avg prompt throughput:\s+(?P<ptp>[0-9.]+)\s+tokens/s,\s*"
    r"Avg generation throughput:\s+(?P<gtp>[0-9.]+)\s+tokens/s,\s*"
    r"Running:\s+(?P<run>\d+)\s+reqs,\s*"
    r"Waiting:\s+(?P<wait>\d+)\s+reqs,\s*"
    r"GPU KV cache usage:\s+(?P<kv>[0-9.]+)%"
)


def parse(path):
    recs = []
    year = datetime.now().year
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = PAT.search(line)
            if not m:
                continue
            ts = datetime.strptime("%d-%s" % (year, m.group("ts")), "%Y-%m-%d %H:%M:%S")
            recs.append(
                dict(
                    ts=ts,
                    ptp=float(m.group("ptp")),
                    gtp=float(m.group("gtp")),
                    run=int(m.group("run")),
                    wait=int(m.group("wait")),
                    kv=float(m.group("kv")),
                )
            )
    recs.sort(key=lambda r: r["ts"])
    return recs


def pct(vals, p):
    if not vals:
        return 0
    s = sorted(vals)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def segments(recs, idle_gap=4, min_seconds=20):
    """Split into busy windows: a window is a maximal run of records with
    run+wait > 0; short idle gaps inside a window are merged."""
    segs = []
    cur = []
    idle = 0
    for r in recs:
        busy = (r["run"] + r["wait"]) > 0
        if busy:
            cur.append(r)
            idle = 0
        else:
            if cur:
                idle += 1
                if idle > idle_gap:
                    segs.append(cur)
                    cur = []
                    idle = 0
            # keep the idle record as the tail of the window (it still carries
            # the last interval's generation throughput)
    if cur:
        segs.append(cur)

    out = []
    for s in segs:
        if len(s) < 2:
            continue
        start, end = s[0]["ts"], s[-1]["ts"]
        dur = (end - start).total_seconds()
        if dur < min_seconds:
            continue
        gen = sum(r["gtp"] for r in s) * 0.0  # placeholder, real calc below
        # every record covers [ts, ts+interval); use the median interval
        deltas = [
            (s[i + 1]["ts"] - s[i]["ts"]).total_seconds() for i in range(len(s) - 1)
        ]
        deltas.sort()
        interval = deltas[len(deltas) // 2] if deltas else 10.0
        gen = sum(r["gtp"] for r in s) * interval
        out.append(
            dict(
                start=start.strftime("%m-%d %H:%M:%S"),
                end=end.strftime("%m-%d %H:%M:%S"),
                duration_s=round(dur, 2),
                interval_s=interval,
                server_generated_tokens=round(gen, 1),
                server_gen_tok_s=round(gen / dur, 3) if dur else 0.0,
                peak_running=max(r["run"] for r in s),
                peak_waiting=max(r["wait"] for r in s),
                max_kv_pct=round(max(r["kv"] for r in s), 2),
                peak_prompt_tok_s=round(max(r["ptp"] for r in s), 1),
                samples=len(s),
                run_p50=round(pct([r["run"] for r in s], 50), 1),
                run_p90=round(pct([r["run"] for r in s], 90), 1),
                wait_p50=round(pct([r["wait"] for r in s], 50), 1),
                wait_p90=round(pct([r["wait"] for r in s], 90), 1),
                kv_p50=round(pct([r["kv"] for r in s], 50), 2),
                kv_p90=round(pct([r["kv"] for r in s], 90), 2),
            )
        )
    return out


def load_client_runs(bench_dir):
    """Load the official client runs (round1.jsonl / round2.jsonl) in
    (tier, round) order, keeping only the fields needed for the cross-check."""
    runs = []
    for rnd in (1, 2):
        p = os.path.join(bench_dir, "round%d.jsonl" % rnd)
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                runs.append(
                    dict(
                        tier=(r.get("random_input_len"), r.get("max_concurrency")),
                        round=rnd,
                        duration=r.get("duration", 0),
                        planned=r.get("total_output_tokens", 0),
                        retokenized=r.get("total_output_tokens_retokenized", 0),
                        out_tps=r.get("output_throughput", 0),
                        completed=r.get("completed", 0),
                    )
                )
    return runs


def cross_check(segs, runs):
    """Match server-side busy windows with client-side runs by duration and
    report the token accounting side by side."""
    used = [False] * len(runs)
    rows = []
    for i, s in enumerate(segs, 1):
        best, bestd = None, None
        for j, r in enumerate(runs):
            if used[j] or not r["duration"]:
                continue
            rel = abs(s["duration_s"] - r["duration"]) / r["duration"]
            if rel > 0.25:
                continue
            if bestd is None or rel < bestd:
                best, bestd = j, rel
        if best is None:
            rows.append((i, s, None))
            continue
        used[best] = True
        rows.append((i, s, runs[best]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="submission/server.log")
    ap.add_argument("--interval", type=float, default=None,
                    help="override log interval in seconds (default: median of observed)")
    ap.add_argument("--idle-gap", type=int, default=4,
                    help="merge busy windows separated by <= N idle log records")
    ap.add_argument("--cross", default=None,
                    help="bench dir with round1.jsonl/round2.jsonl for the token cross-check")
    ap.add_argument("--json", default=None)
    ap.add_argument("--out", default=None, help="also write a markdown table here")
    args = ap.parse_args()

    if not os.path.exists(args.log):
        print("log not found: %s" % args.log, file=sys.stderr)
        return 2
    recs = parse(args.log)
    segs = segments(recs, idle_gap=args.idle_gap)
    if args.interval:
        for s in segs:
            s["server_generated_tokens"] = round(s["server_gen_tok_s"] * s["duration_s"], 1)

    print("records=%d  busy_segments=%d" % (len(recs), len(segs)))
    print()
    hdr = ("#", "window", "duration_s", "running", "waiting", "KV%", "srv_gen_tokens",
           "srv_gen_tok/s")
    print("| " + " | ".join(hdr) + " |")
    print("|" + " --- |" * len(hdr))
    for i, s in enumerate(segs, 1):
        print("| %d | %s → %s | %.1f | %d | %d | %.1f | %.0f | %.2f |" % (
            i, s["start"], s["end"], s["duration_s"], s["peak_running"],
            s["peak_waiting"], s["max_kv_pct"], s["server_generated_tokens"],
            s["server_gen_tok_s"]))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(segs, f, ensure_ascii=False, indent=2)
        print("\nwrote %s" % args.json)

    if args.cross:
        runs = load_client_runs(args.cross)
        rows = cross_check(segs, runs)
        print()
        print("=== 服务端引擎计数器 vs 官方客户端计量（逐档交叉核对）===")
        hdr = ("#", "档位(in/conc)", "轮次", "引擎生成token", "客户端计划token", "客户端重分词token",
               "引擎/计划", "引擎吞吐tok/s", "客户端官方吞吐tok/s", "偏差%")
        print("| " + " | ".join(hdr) + " |")
        print("|" + " --- |" * len(hdr))
        for i, s, r in rows:
            if r is None:
                print("| %d | - | - | %.0f | - | - | - | %.2f | - | - |" % (
                    i, s["server_generated_tokens"], s["server_gen_tok_s"]))
                continue
            ratio = (s["server_generated_tokens"] / r["planned"] * 100.0
                     if r["planned"] else 0.0)
            dev = ((s["server_gen_tok_s"] - r["out_tps"]) / r["out_tps"] * 100.0
                   if r["out_tps"] else 0.0)
            print("| %d | %s/%s | R%s | %.0f | %d | %d | %.1f%% | %.2f | %.2f | %+.1f%% |" % (
                i, r["tier"][0], r["tier"][1], r["round"],
                s["server_generated_tokens"], r["planned"], r["retokenized"], ratio,
                s["server_gen_tok_s"], r["out_tps"], dev))

        print()
        print("=== 各档服务端常驻并发 / 排队 / KV 占用分布 ===")
        hdr2 = ("#", "档位(in/conc)", "轮次", "run p50", "run p90", "run max",
                "wait p50", "wait p90", "KV% p50", "KV% p90", "KV% max")
        print("| " + " | ".join(hdr2) + " |")
        print("|" + " --- |" * len(hdr2))
        for i, s, r in rows:
            if r is None:
                continue
            print("| %d | %s/%s | R%s | %.0f | %.0f | %d | %.0f | %.0f | %.1f | %.1f | %.1f |" % (
                i, r["tier"][0], r["tier"][1], r["round"],
                s["run_p50"], s["run_p90"], s["peak_running"],
                s["wait_p50"], s["wait_p90"],
                s["kv_p50"], s["kv_p90"], s["max_kv_pct"]))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("## 服务端计数器独立核算（vLLM 引擎侧生成 token）\n\n")
            f.write("| # | 时间窗 | 时长 s | 常驻并发 | 排队 | 峰值KV% | 引擎生成 token | 引擎生成吞吐 tok/s |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
            for i, s in enumerate(segs, 1):
                f.write("| %d | %s → %s | %.1f | %d | %d | %.1f | %.0f | %.2f |\n" % (
                    i, s["start"], s["end"], s["duration_s"], s["peak_running"],
                    s["peak_waiting"], s["max_kv_pct"], s["server_generated_tokens"],
                    s["server_gen_tok_s"]))
            if args.cross:
                runs = load_client_runs(args.cross)
                rows = cross_check(segs, runs)
                f.write("\n## 引擎计数器 vs 官方客户端计量（逐档交叉核对）\n\n")
                f.write("| # | 档位(in/conc) | 轮次 | 引擎生成token | 客户端计划token | 客户端重分词token | 引擎/计划 | 引擎吞吐tok/s | 官方表吞吐tok/s | 偏差 |\n")
                f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
                for i, s, r in rows:
                    if r is None:
                        f.write("| %d | - | - | %.0f | - | - | - | %.2f | - | - |\n" % (
                            i, s["server_generated_tokens"], s["server_gen_tok_s"]))
                        continue
                    ratio = (s["server_generated_tokens"] / r["planned"] * 100.0
                             if r["planned"] else 0.0)
                    dev = ((s["server_gen_tok_s"] - r["out_tps"]) / r["out_tps"] * 100.0
                           if r["out_tps"] else 0.0)
                    f.write("| %d | %s/%s | R%s | %.0f | %d | %d | %.1f%% | %.2f | %.2f | %+.1f%% |\n" % (
                        i, r["tier"][0], r["tier"][1], r["round"],
                        s["server_generated_tokens"], r["planned"], r["retokenized"],
                        ratio, s["server_gen_tok_s"], r["out_tps"], dev))
                f.write("\n## 各档常驻并发 / 排队 / KV 占用分布\n\n")
                f.write("| # | 档位(in/conc) | 轮次 | run p50 | run p90 | run max | wait p50 | wait p90 | KV% p50 | KV% p90 | KV% max |\n")
                f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
                for i, s, r in rows:
                    if r is None:
                        continue
                    f.write("| %d | %s/%s | R%s | %.0f | %.0f | %d | %.0f | %.0f | %.1f | %.1f | %.1f |\n" % (
                        i, r["tier"][0], r["tier"][1], r["round"],
                        s["run_p50"], s["run_p90"], s["peak_running"],
                        s["wait_p50"], s["wait_p90"],
                        s["kv_p50"], s["kv_p90"], s["max_kv_pct"]))
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
