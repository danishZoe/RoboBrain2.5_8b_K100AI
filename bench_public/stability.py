#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""稳定性检查：每个基准固定 ≥10 条样例，两次重复推理（验收要求 §6.5）。

* 样例**预先固定**并覆盖任务分组；
* 同一服务、同一配置、同一并发，两次独立推理；
* 逐条比较：原始输出是否逐字符一致 → 解析结果是否一致 → 得分是否一致；
* 输出变化比例与**得分影响**（变化样本上的官方得分差异）。

用法::

    python3 submission/bench_public/stability.py \
        --url http://127.0.0.1:30000/v1/chat/completions \
        --out-dir outputs/bench_public/stability
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C          # noqa: E402
import datasets as D        # noqa: E402

import run_refspatial as RR      # noqa: E402
import run_tracespatial as RT    # noqa: E402
import run_cv_bench as RC        # noqa: E402


def _pick(items: List[Dict[str, Any]], n: int, key: str) -> List[Dict[str, Any]]:
    """按分组均衡取样，保证覆盖任务分组。"""
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        buckets.setdefault(str(it.get(key)), []).append(it)
    per = max(1, n // max(len(buckets), 1))
    out: List[Dict[str, Any]] = []
    for k in sorted(buckets):
        out.extend(buckets[k][:per])
    # 不足则按顺序补齐
    if len(out) < n:
        chosen = {id(x) for x in out}
        for it in items:
            if len(out) >= n:
                break
            if id(it) not in chosen:
                out.append(it)
    return out[:n]


# --------------------------------------------------------------------------- #
def stability_refspatial(url: str, model: str, out_dir: str, n: int,
                         concurrency: int) -> Dict[str, Any]:
    samples: List[Dict[str, Any]] = []
    for split in ("Location", "Placement"):
        s = D.load_refspatial(split)
        for it in s:
            it = dict(it)
            it["group"] = split
            samples.append(it)
    picked = _pick(samples, n, "group")
    jobs = []
    for s in picked:
        s = dict(s)
        s["prompt"] = C.prompt_upstream(
            C.strip_one_trailing_period(s["fields"]["prompt"]), "pointing")
        jobs.append(s)

    runs = []
    for i in (1, 2):
        recs = _run_once(jobs, url, model, out_dir, f"refspatial_run{i}",
                          concurrency, "pointing")
        runs.append({r["sample_id"]: r for r in recs})
    return _compare("RefSpatial-Bench", jobs, runs)


def stability_tracespatial(url: str, model: str, out_dir: str, n: int,
                           concurrency: int) -> Dict[str, Any]:
    s = D.load_tracespatial("3D")
    for it in s:
        it["group"] = "3D"
    picked = _pick(s, n, "group")
    jobs = []
    for it in picked:
        it = dict(it)
        it["prompt"] = C.prompt_tracespatial_official(
            "3D", it["target_object"], it["destination"], it["raw_prompt"])
        jobs.append(it)
    runs = []
    for i in (1, 2):
        recs = _run_once(jobs, url, model, out_dir, f"tracespatial_run{i}",
                         concurrency, "trajectory")
        runs.append({r["sample_id"]: r for r in recs})
    return _compare("TraceSpatial-Bench", jobs, runs)


def stability_cvbench(url: str, model: str, out_dir: str, n: int,
                      concurrency: int) -> Dict[str, Any]:
    all_jobs = []
    for ann, data in D.iter_cvbench():
        all_jobs.append({
            "sample_id": f"CVBench/{ann['question_id']}",
            "split": ann["type_ori"],
            "group": ann["source"],
            "image_paths": [("bytes", data)],
            "prompt": C.prompt_cv_bench_flag_eval_mm(ann["question"],
                                                     ann["options"]),
            "fields": ann,
            "task": None,
        })
        if len(all_jobs) >= 4000:
            break
    picked = _pick(all_jobs, n, "group")
    runs = []
    for i in (1, 2):
        recs = _run_once(picked, url, model, out_dir, f"cvbench_run{i}",
                         concurrency, None)
        runs.append({r["sample_id"]: r for r in recs})
    return _compare("CV-Bench", picked, runs)


def _run_once(jobs, url, model, out_dir, tag, concurrency, task):
    import engine as E
    path = os.path.join(out_dir, tag, "predictions.jsonl")
    pf = (lambda t: C.parse_points_1000(t, task)) if task else \
         (lambda t: t.strip() or None)
    return E.run_jobs(jobs, path, url, model,
                      max_tokens=C.MAX_NEW_TOKENS, concurrency=concurrency,
                      resume=False, parse_fn=pf, label=f"[stab {tag}]")


def _compare(dataset: str, jobs, runs) -> Dict[str, Any]:
    a, b = runs
    rows = []
    for j in jobs:
        sid = j["sample_id"]
        ra = a.get(sid, {})
        rb = b.get(sid, {})
        ta = (ra.get("raw_output") or "").strip()
        tb = (rb.get("raw_output") or "").strip()
        rows.append({
            "sample_id": sid,
            "group": j.get("group") or j.get("split"),
            "identical_raw": ta == tb,
            "identical_parsed": ra.get("parsed") == rb.get("parsed"),
            "run1": ta,
            "run2": tb,
            "run1_parsed": ra.get("parsed"),
            "run2_parsed": rb.get("parsed"),
            "run1_state": ra.get("terminal_state"),
            "run2_state": rb.get("terminal_state"),
        })
    n = len(rows)
    same = sum(1 for r in rows if r["identical_raw"])
    return {
        "dataset": dataset,
        "n": n,
        "identical_raw": same,
        "changed_raw": n - same,
        "change_ratio": round((n - same) / n, 4) if n else None,
        "all_states_ok": all(r["run1_state"] == "ok" and r["run2_state"] == "ok"
                              for r in rows),
        "samples": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=C.DEFAULT_URL)
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--out-dir",
                    default=os.path.join(C.WORK_DIR, "outputs/bench_public/stability"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    C.guard_no_official_bench()

    out: Dict[str, Any] = {"protocol": {
        "n_per_benchmark": args.n,
        "concurrency": args.concurrency,
        "decode": C.DECODE,
        "max_tokens": C.MAX_NEW_TOKENS,
        "note": "两次为独立推理；vLLM 批内归约顺序可能带来极小数值差异，"
                "本检查量化该差异",
    }}
    out["refspatial"] = stability_refspatial(args.url, args.model, args.out_dir,
                                              args.n, args.concurrency)
    out["tracespatial"] = stability_tracespatial(args.url, args.model,
                                                  args.out_dir, args.n,
                                                  args.concurrency)
    out["cvbench"] = stability_cvbench(args.url, args.model, args.out_dir,
                                        args.n, args.concurrency)
    sp = os.path.join(args.out_dir, "stability.json")
    C.dump_json(sp, out)
    print("[stability] -> %s" % sp)
    for k in ("refspatial", "tracespatial", "cvbench"):
        d = out[k]
        print("  %-18s n=%s 原始输出一致 %s/%s  变化率=%s  状态全OK=%s" %
              (k, d["n"], d["identical_raw"], d["n"], d["change_ratio"],
               d["all_states_ok"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
