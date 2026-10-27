#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 verify.py 自带的 agreement 判分器，对已有的跨后端一致性证据重算一致率。

纯内部指标（同一台卡、同一批样例、不同推理后端），不涉及官方数据，
不能替代官方「国产卡 vs GPU 参考输出」一致率（≥98%）。
"""
import json
import sys

sys.path.insert(0, "/root/private_data/workspace/robo/submission")
import verify as V  # noqa: E402  复用官方同款解析/判分函数

W = "/root/private_data/workspace/robo"
base = json.load(open(f"{W}/outputs/phase3_baseline/baseline_raw.json", encoding="utf-8"))
probe = json.load(open(f"{W}/outputs/phase5_probe_canonical/probe_result.json", encoding="utf-8"))


def as_list(d):
    if isinstance(d, list):
        return d
    if isinstance(d, dict):
        for k in ("results", "records"):
            if isinstance(d.get(k), list):
                return d[k]
    return []


B = {}
for r in as_list(base):
    if not isinstance(r, dict):
        continue
    sid = r.get("id") or r.get("sample_id")
    ans = r.get("answer")
    if isinstance(r.get("baseline"), dict):
        ans = r["baseline"].get("text")
    if sid and ans:
        B.setdefault(sid, ans)

P, TASK = {}, {}
for r in as_list(probe):
    if not isinstance(r, dict):
        continue
    sid = r.get("sample_id") or r.get("id")
    txt = r.get("text") or r.get("serving") or r.get("answer")
    if sid and txt:
        P.setdefault(sid, txt)
        TASK.setdefault(sid, r.get("task") or "")

print("基线样例:", len(B), " serving 样例:", len(P))
print()

rows = []
for sid in sorted(P):
    if sid not in B:
        continue
    task = TASK.get(sid) or "general"
    for t in ("general", "grounding", "pointing", "trajectory"):
        if t in sid.lower():
            task = t
            break
    ok0 = V.judge_sample("agreement", task, P[sid], B[sid], None, 0)
    ok10 = V.judge_sample("agreement", task, P[sid], B[sid], None, 10)
    pb = V.parse_answer(B[sid], task)
    ps = V.parse_answer(P[sid], task)
    dev = ""
    try:
        if isinstance(pb, dict) and isinstance(ps, dict):
            if "bbox" in pb and "bbox" in ps:
                dev = max(abs(a - b) for a, b in zip(pb["bbox"], ps["bbox"]))
            elif "points" in pb and "points" in ps:
                vals = []
                for a, b in zip(pb["points"], ps["points"]):
                    if isinstance(a, dict) and isinstance(b, dict):
                        vals.append(max(abs(a["x"] - b["x"]), abs(a["y"] - b["y"])))
                    else:
                        vals.append(max(abs(x - y) for x, y in zip(a, b)))
                dev = max(vals) if vals else ""
    except Exception:
        dev = ""
    rows.append((sid, task, bool(ok0), bool(ok10), dev))

print(f"{'样例':<22}{'任务':<12}{'完全一致':<12}{'±10px容差内':<14}{'坐标偏差px'}")
for sid, task, ok0, ok10, dev in rows:
    print(f"{sid:<22}{task:<12}{'YES' if ok0 else 'NO':<12}"
          f"{'YES' if ok10 else 'NO':<14}{dev}")

n = len(rows)
if n:
    s0 = sum(1 for r in rows if r[2])
    s10 = sum(1 for r in rows if r[3])
    print()
    print(f"样本数                       = {n}")
    print(f"完全一致率(容差 0)           = {s0}/{n} = {s0 / n * 100:.1f}%")
    print(f"agreement 一致率(容差 ±10px) = {s10}/{n} = {s10 / n * 100:.1f}%")
    coord = [r for r in rows if r[1] != "general"]
    if coord:
        c10 = sum(1 for r in coord if r[3])
        print(f"仅坐标类样例(不含自由文本)   = {c10}/{len(coord)} = "
              f"{c10 / len(coord) * 100:.1f}%   (官方 98% 门槛的参照对象)")
