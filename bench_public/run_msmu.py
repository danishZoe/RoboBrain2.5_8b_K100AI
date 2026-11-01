#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MSMU-Bench 评测（test 987 题：5 类定量 + 3 类定性）。

上游官方件
----------
* 数据：``cpystan/MSMU``（HF，test 分片 arrow，图像内嵌）。
* 推理 prompt：官方 ``SD-VLM/evaluation/inference.py`` 用
  ``conversations[0].value``（``<image>\\n{question}``）原题，无额外模板。
* 判分流程（官方 ``SD-VLM/evaluation/evaluate_final.py`` + ``llm_generate.py``）：
  1) **judge LLM** 按官方 ``judge_prompt``（含固定单位换算表）把 GT 与模型回答
     抽成 ``answer_in_meters`` / ``response_in_meters``（定性题为 0/1 mark）；
  2) 定量：``success = [ratio<1.25, ratio<1.25², ratio<1.25³]``，
     ``error_rate = |resp-gt|/(gt+1e-4)``；定性：judge mark 直接计入。

⚠️ 判分适配披露
----------------
官方 judge 用 llama3-8B；本复现用**本任务同一 vLLM 服务**
（RoboBrain2.5-8B-NV，greedy）充当 judge，``judge_prompt`` 逐字沿用。
抽取为机械数值换算，greedy 下判定稳定；judge 原始输出逐题落盘可审计。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C          # noqa: E402
import engine as E          # noqa: E402

BENCH_DIR = "/root/private_data/workspace/robo/third_party/benchmarks"
ARROW = os.path.join(BENCH_DIR, "MSMU/test-data.arrow")
IMG_DIR = "/root/private_data/workspace/robo/tmp/msmu_images"

# 官方 SD-VLM/evaluation/inference.py 的 judge_prompt（逐字）
JUDGE_PROMPT = '''You are a helpful assistant designed to output in the format of given examples.You should help me to evaluate the response given the question and the correct answer. You need to convert the distance of the correct answer and response to meters.\
The conversion is: 1 inch = 0.0254 meters. 1 foot = 0.3048 meters. 1 centimeter (cm) = 0.01 meters.\
You should output two floats in meters, one for the answer, and one for the response. \
the format is {"answer_in_meters": [a float or list of floats], "response_in_meters": [a float or list of floats]}\
\n Answer: the table in the picture is 2.28 meters long, 0.75 meters wide and 0.87 meters tall.\
\n Responsde: The size of the table is 100 centimeters.\
\n {"answer_in_meters": [2.28,0.75,0.87], "response_in_meters": [1]}  \
\n Answer: The width of the chair is 1.02 meters. \
\n Responsde: The chair is 2.17 meters wide.\
\n {"answer_in_meters": [1.02], "response_in_meters": [2.17]} '''

# 定性题 judge：沿用同一抽取框架，输出 your_mark
JUDGE_PROMPT_QUAL = '''You are a helpful assistant designed to output in the format of given examples. You should help me to evaluate whether the response gives the same conclusion as the correct answer. \
Output {"your_mark": 1} if the response's conclusion matches the correct answer, otherwise {"your_mark": 0}. \
\n Answer: the office chair is more closer. \
\n Response: The office chair is closer to the viewer. \
\n {"your_mark": 1} \
\n Answer: The height of the black chair is 78.14 cm and the wooden bookshelf is 118.68 cm, so the wooden bookshelf is taller. \
\n Response: The wooden bookshelf is taller than the black chair. \
\n {"your_mark": 1}'''

QUANT_TYPES = ("scale_estimation", "absolute_distance", "count",
               "grounding", "refer_obj_estimation")
QUAL_TYPES = ("relative_position", "scale_compare", "existence")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=C.DEFAULT_URL)
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--run-id", default="noneager-msmu")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--judge-max-tokens", type=int, default=128)
    args = ap.parse_args()

    import pyarrow.ipc as ipc
    with open(ARROW, "rb") as fh:
        try:
            reader = ipc.open_stream(fh)
        except Exception:
            fh.seek(0)
            reader = ipc.open_file(fh)
        table = reader.read_all()
    rows = table.to_pylist()
    if args.limit > 0:
        rows = rows[:args.limit]
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    jobs: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        sid = "MSMU/%d" % i
        conv = r["conversations"]
        q = conv["value"][0].replace("<image>", "").strip()
        gt = conv["value"][1].strip()
        img_path = os.path.join(IMG_DIR, "%04d.jpg" % i)
        if not os.path.exists(img_path):
            with open(img_path, "wb") as fh:
                fh.write(r["image"]["bytes"])
        jobs.append({
            "sample_id": sid, "split": r["type"],
            "image_paths": [img_path], "prompt": q,
            "fields": {"question": q, "gt_answer": gt, "msmu_type": r["type"]},
        })

    out_jsonl = os.path.join(args.out_dir, "predictions.jsonl")
    recs = E.run_jobs(jobs, out_jsonl, args.url, args.model,
                      max_tokens=args.max_tokens, concurrency=args.concurrency,
                      run_meta={"run_id": args.run_id, "bench": "msmu"})
    meta_by_sid = {j["sample_id"]: j["fields"] for j in jobs}

    # ---- judge 阶段（官方 judge_prompt 逐字；judge = 本服务，greedy）----
    judge_jobs: List[Dict[str, Any]] = []
    for r in recs:
        f = meta_by_sid.get(r["sample_id"], {})
        t = f.get("msmu_type", "")
        if r.get("terminal_state") != "ok":
            continue
        if t in QUANT_TYPES:
            prompt = "%s\n\nAnswer: %s\nResponse: %s" % (
                JUDGE_PROMPT, f.get("gt_answer", ""), (r.get("raw_output") or "").strip())
        elif t in QUAL_TYPES:
            prompt = "%s\n\nAnswer: %s\nResponse: %s" % (
                JUDGE_PROMPT_QUAL, f.get("gt_answer", ""), (r.get("raw_output") or "").strip())
        else:
            continue
        judge_jobs.append({
            "sample_id": r["sample_id"] + "/judge", "split": t,
            "image_paths": [], "prompt": prompt, "fields": {"sid": r["sample_id"]},
        })
    judge_jsonl = os.path.join(args.out_dir, "judge.jsonl")
    jrecs = E.run_jobs(judge_jobs, judge_jsonl, args.url, args.model,
                       max_tokens=args.judge_max_tokens, concurrency=args.concurrency,
                       run_meta={"run_id": args.run_id + "-judge", "bench": "msmu-judge"})
    jmeta = {j["sample_id"]: (j["fields"] or {}).get("sid") for j in judge_jobs}
    judge_by_sid = {jmeta.get(r["sample_id"]): r for r in jrecs}

    # ---- 汇总（官方 evaluate_final.py 的公式逐字复刻）----
    def _parse_meters(text: str):
        import re, ast
        m = re.search(r"\{[^{}]*answer_in_meters[^{}]*\}", text or "")
        if not m:
            return None
        try:
            d = ast.literal_eval(m.group(0))
            a, p = d["answer_in_meters"], d["response_in_meters"]
            a = [float(x) for x in (a if isinstance(a, list) else [a])]
            p = [float(x) for x in (p if isinstance(p, list) else [p])]
            return a, p
        except Exception:
            return None

    type_stats: Dict[str, Counter] = {}
    per_sample = []
    n_succ_125 = n_succ_156 = n_succ_195 = n_qual = 0
    n_quant = n_qual_total = 0
    n_judge_fail = 0
    for r in recs:
        f = meta_by_sid.get(r["sample_id"], {})
        t = f.get("msmu_type", "?")
        st = type_stats.setdefault(t, Counter())
        st["total"] += 1
        succ = None
        if r.get("terminal_state") == "ok":
            j = judge_by_sid.get(r["sample_id"])
            jraw = (j.get("raw_output") or "") if j else ""
            if t in QUANT_TYPES:
                n_quant += 1
                parsed = _parse_meters(jraw)
                if parsed and parsed[0] and parsed[1]:
                    a, p = parsed
                    n = min(len(a), len(p))
                    ratios = []
                    for i in range(n):
                        if a[i] == 0 or p[i] == 0:
                            # 已披露健壮性补丁：任一侧为 0 时官方实现会除零崩溃；
                            # 两侧都 ~0 记 ratio=1（成功），否则记 inf（失败）。
                            ratios.append(1.0 if abs(a[i] - p[i]) < 1e-6 else float("inf"))
                        else:
                            ratios.append(max(p[i] / a[i], a[i] / p[i]))
                    ratio = sum(ratios) / len(ratios)
                    succ = [ratio < 1.25, ratio < 1.25 ** 2, ratio < 1.25 ** 3]
                else:
                    n_judge_fail += 1
                    succ = [False, False, False]
                st["succ125"] += int(succ[0])
                n_succ_125 += int(succ[0]); n_succ_156 += int(succ[1]); n_succ_195 += int(succ[2])
            else:
                n_qual_total += 1
                import re as _re
                m = _re.search(r'"your_mark":\s*(\d)', jraw)
                mark = int(m.group(1)) if m else 0
                if not m:
                    n_judge_fail += 1
                st["succ125"] += mark
                n_qual += mark
        per_sample.append({"sample_id": r["sample_id"], "type": t,
                           "success": succ, "terminal_state": r.get("terminal_state")})

    total = len(recs)
    summary = {
        "bench": "MSMU-Bench (test)",
        "protocol": "official prompts; judge_prompt verbatim; judge=local service (greedy); success ratio thresholds 1.25/1.5625/1.953125",
        "n": total,
        "quant": {"n": n_quant, "succ_rate_1.25": round(n_succ_125 / max(n_quant, 1), 4),
                  "succ_rate_1.25^2": round(n_succ_156 / max(n_quant, 1), 4),
                  "succ_rate_1.25^3": round(n_succ_195 / max(n_quant, 1), 4)},
        "qual": {"n": n_qual_total, "acc": round(n_qual / max(n_qual_total, 1), 4)},
        "judge_fail": n_judge_fail,
        "per_type": {k: {"n": v["total"], "succ": v.get("succ125", 0),
                         "rate": round(v.get("succ125", 0) / v["total"], 4)}
                     for k, v in sorted(type_stats.items())},
        "terminal_states": dict(Counter(r.get("terminal_state") for r in recs)),
        "per_sample": per_sample,
        "run_id": args.run_id,
    }
    C.dump_json(os.path.join(args.out_dir, "summary.json"), summary)
    print("[msmu] n=%d quant=%s qual=%s judge_fail=%d" %
          (total, summary["quant"], summary["qual"], n_judge_fail))
    return 0


if __name__ == "__main__":
    sys.exit(main())
