#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare a served-backend result file against the transformers baseline.

Purpose: produce the cross-backend agreement evidence that the competition
calls "解析后的动作 / 坐标与 GPU 参考输出一致率" — here measured between the
accepted transformers baseline and a serving backend (vLLM / SGLang) on the
SAME fixed samples, since no official GPU reference outputs are available yet.

Usage:
  python3 scripts/compare_backends.py \
      --baseline outputs/phase3_baseline/baseline_raw.json \
      --candidate outputs/probe_vllm/probe_result.json \
      --out outputs/probe_vllm/agreement.md

Coordinates are compared on the official 0..1000 normalised scale.
"""
from __future__ import annotations

import argparse
import json
import os
import re

RE_TRAJ = re.compile(r"(\d+)\s*,\s*(\d+)\s*,\s*([+-]?\d+\.\d+)")
RE_POINT = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
RE_BOX = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]")


def extract(task: str, text: str):
    """Return comparable structures per task type."""
    if task == "trajectory":
        pts = [(int(a), int(b), float(c)) for a, b, c in RE_TRAJ.findall(text)]
        return pts
    if task == "grounding":
        m = RE_BOX.search(text)
        return tuple(int(v) for v in m.groups()) if m else None
    if task == "pointing":
        pts = [(int(a), int(b)) for a, b in RE_POINT.findall(text)]
        return pts
    return text.strip()


def coord_delta(a, b):
    """Max absolute coordinate delta on the 0..1000 scale (depth excluded)."""
    if a is None or b is None:
        return None
    if isinstance(a, tuple) and isinstance(b, tuple):
        if len(a) != len(b):
            return None
        deltas = [abs(x - y) for x, y in zip(a, b) if not isinstance(x, float)]
        return max(deltas) if deltas else 0
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return None
        flat_a = [v for p in a for v in (p if isinstance(p, tuple) else (p,))
                  if not isinstance(v, float)]
        flat_b = [v for p in b for v in (p if isinstance(p, tuple) else (p,))
                  if not isinstance(v, float)]
        if len(flat_a) != len(flat_b):
            return None
        return max(abs(x - y) for x, y in zip(flat_a, flat_b))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    with open(args.baseline, encoding="utf-8") as f:
        base = json.load(f)
    with open(args.candidate, encoding="utf-8") as f:
        cand = json.load(f)

    # baseline: first non-warmup record per sample_id
    base_map = {}
    for r in base.get("records", []):
        if r.get("warmup"):
            continue
        base_map.setdefault(r["sample_id"], r)

    cand_records = cand.get("records", [])
    rows = []
    agree = 0
    comparable = 0
    for c in cand_records:
        sid = c.get("sample_id")
        b = base_map.get(sid)
        if b is None or "error" in c:
            rows.append((sid, c.get("task"), None, None, None, None, None))
            continue
        bt, ct = b.get("task"), c.get("task")
        bv = extract(bt, b["answer"])
        cv = extract(ct, c["text"])
        if bt == "general":
            same = bv == cv
            delta = 0 if same else None
        else:
            delta = coord_delta(bv, cv)
            same = (delta is not None and delta == 0)
        comparable += 1
        agree += 1 if same else 0
        rows.append((sid, bt, bv, cv, delta, same,
                     (c.get("seconds"), c.get("tok_per_s"))))

    lines = []
    lines.append("# 跨后端一致性比对（transformers 基线 vs serving 后端）\n")
    lines.append(f"- 基线：`{args.baseline}`")
    lines.append(f"- 对比对象：`{args.candidate}`")
    lines.append(f"- 后端：{cand.get('backend', '?')}  接口：{cand.get('url', '?')}\n")
    lines.append("| 样例 | 任务 | 基线输出 | serving 输出 | 最大坐标偏差(/1000) | 完全一致 |")
    lines.append("|---|---|---|---|---|---|")
    for sid, task, bv, cv, delta, same, perf in rows:
        lines.append(
            f"| {sid} | {task} | `{str(bv)[:60]}` | `{str(cv)[:60]}` | "
            f"{'-' if delta is None else delta} | {'✅' if same else '❌'} |")
    rate = (agree / comparable * 100) if comparable else 0.0
    lines.append("")
    lines.append(f"**逐字符/逐坐标完全一致率：{agree}/{comparable} = {rate:.1f}%**")
    lines.append("")
    lines.append("> 说明：这是「同一台卡、同一批样例、不同推理后端」之间的一致性，")
    lines.append("> 不是官方要求的「国产卡 vs GPU 参考输出」一致率（后者需要官方 GPU 参考文件）。")
    lines.append("> 但它能证明 serving 路径与已验证的 transformers 路径在数值上没有分歧。")

    text = "\n".join(lines) + "\n"
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.out}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
