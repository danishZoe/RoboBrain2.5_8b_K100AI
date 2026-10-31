#!/usr/bin/env python3
"""Aggregate every phase-4 experiment (plus the phase-3 baseline) into one
comparable table, and check that each configuration still produces the expected
task output format.

Usage:
  python3 scripts/compare_experiments.py \
      --baseline outputs/phase3_baseline/baseline_raw.json \
      --dirs outputs/phase4 --out outputs/phase4/comparison.md
"""

import argparse
import glob
import json
import os
import re
import statistics

BASELINE_ANSWERS = {
    "official_general": "The image shows a white ceramic mug filled with a dark liquid",
    "official_grounding": r"\[\s*0\s*,\s*309\s*,\s*620\s*,\s*989\s*\]",
    "official_affordance": r"\(\s*901\s*,\s*600\s*\)",
    "official_pointing": r"\(\s*582\s*,\s*725\s*\)",
    "official_navigation": r"\(\s*4[0-9]{2}\s*,\s*3[0-9]{2}\s*\)",
    "official_trajectory": r"\(\s*294\s*,\s*525\s*,\s*0\.577\s*\)",
}


def fmt(v, nd=4):
    return "-" if v is None else f"{v:.{nd}f}"


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def row_for(name, summary):
    ss = summary.get("steady_state", summary)
    ttft = (ss.get("ttft_seconds") or {}).get("mean")
    tot = (ss.get("total_seconds") or {}).get("mean")
    dec = (ss.get("decode_tok_per_s") or {}).get("mean")
    e2e = (ss.get("e2e_tok_per_s") or {}).get("mean")
    vram = (ss.get("peak_vram_reserved_gib") or {}).get("mean")
    ptok = (ss.get("prompt_tokens") or {}).get("mean")
    return {
        "config": name,
        "ttft": ttft, "total": tot, "decode": dec, "e2e": e2e, "vram": vram, "ptok": ptok,
    }


def answers_of(summary):
    a = summary.get("answers")
    if isinstance(a, dict):
        return a
    return {}


def answers_from_records(blob):
    """benchmark.py stores answers per record; collect the first steady-state one."""
    out = {}
    for rec in blob.get("records", []):
        if rec.get("warmup"):
            continue
        sid = rec.get("sample_id")
        if sid and sid not in out:
            out[sid] = rec.get("answer", "")
    return out


def prompt_tokens_from_records(blob):
    vals = [r["prompt_tokens"] for r in blob.get("records", [])
            if not r.get("warmup") and r.get("prompt_tokens") is not None]
    return round(statistics.mean(vals), 1) if vals else None


def check_answers(answers):
    """Each task must still match the baseline output format (not merely be non-empty)."""
    out = {}
    for sid, pattern in BASELINE_ANSWERS.items():
        got = answers.get(sid, "")
        out[sid] = bool(got) and bool(re.search(pattern, got))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="outputs/phase3_baseline/baseline_raw.json")
    ap.add_argument("--dirs", nargs="+", default=["outputs/phase4"])
    ap.add_argument("--out", default="outputs/phase4/comparison.md")
    args = ap.parse_args()

    entries = []

    b = load(args.baseline)
    bsum = dict(b["summary"])
    bsum.setdefault("answers", answers_from_records(b))
    if not bsum["steady_state"].get("prompt_tokens"):
        pt = prompt_tokens_from_records(b)
        if pt is not None:
            bsum["steady_state"]["prompt_tokens"] = {"n": 1, "mean": pt, "min": pt,
                                                     "max": pt, "median": pt}
    entries.append(("baseline (transformers, default)", row_for("baseline", bsum),
                    check_answers(answers_of(bsum) or {})))

    baseline_answers = answers_of(bsum) or {}

    for d in args.dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.json"))):
            name = os.path.splitext(os.path.basename(p))[0]
            if name in ("comparison",):
                continue
            try:
                blob = load(p)
            except Exception:
                continue
            summary = blob.get("summary", blob)
            if "steady_state" not in summary:
                continue  # not a tune.py / bench_http.py artefact
            ans = answers_of(summary) or answers_from_records(blob)
            entries.append((name, row_for(name, summary), check_answers(ans)))

    lines = ["# 第四阶段实验对比（单变量受控，同口径）", "",
             "基线 = 第三阶段 transformers 默认配置。TTFT/时延/吞吐为稳态均值（n=12，"
             "已剔除首次 warm-up）。", "",
             "| 配置 | 视觉 token 均值 | TTFT s | 端到端 s | 解码 tok/s | 端到端 tok/s | 峰值显存 GiB | 输出格式校验 |",
             "|---|---|---|---|---|---|---|---|"]
    base = entries[0][1]
    for name, r, checks in entries:
        ok = sum(1 for v in checks.values() if v)
        mark = f"{ok}/{len(checks)}" if checks else "-"
        if name.startswith("baseline"):
            mark = f"**{ok}/{len(checks)}**"
        lines.append(
            f"| `{name}` | {fmt(r['ptok'], 1)} | {fmt(r['ttft'])} | {fmt(r['total'])} | "
            f"{fmt(r['decode'], 2)} | {fmt(r['e2e'], 2)} | {fmt(r['vram'], 2)} | {mark} |")

    lines += ["", "## 相对基线的变化", ""]
    for name, r, _ in entries:
        if name.startswith("baseline"):
            continue
        def delta(k):
            if r[k] is None or base[k] is None or base[k] == 0:
                return "-"
            return f"{(r[k] - base[k]) / base[k] * 100:+.1f}%"
        lines.append(f"- `{name}`: TTFT {delta('ttft')}, 端到端 {delta('total')}, "
                     f"解码吞吐 {delta('decode')}, 峰值显存 {delta('vram')}, "
                     f"视觉 token {delta('ptok')}")

    lines += ["", "## 各配置的实际输出", ""]
    for name, r, checks in entries:
        lines.append(f"### `{name}`")
        blob_answers = {}
        for d in args.dirs:
            p = os.path.join(d, name + ".json")
            if os.path.isfile(p):
                blob = load(p)
                blob_answers = answers_of(blob.get("summary", {})) or answers_from_records(blob)
        if not blob_answers:
            blob_answers = baseline_answers
        for sid in BASELINE_ANSWERS:
            val = blob_answers.get(sid, "")
            flag = "OK " if checks.get(sid) else "!! "
            lines.append(f"- {flag}`{sid}`: `{val[:150]}`")
        lines.append("")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nartifact:", args.out)


if __name__ == "__main__":
    main()
