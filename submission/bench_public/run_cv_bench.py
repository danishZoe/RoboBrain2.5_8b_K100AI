#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CV-Bench 评测（官方 FlagEvalMM 判分实现，逐字调用）。

上游官方件
----------
* 数据：``nyu-visionx/CV-Bench``（HF）→ ``test_2d.parquet`` / ``test_3d.parquet``
* **判分**：``FlagEvalMM/tasks/cv_bench/evaluate.py``（``get_result`` /
  ``cal_accuracy``）与 ``FlagEvalMM/flagevalmm/evaluator/common_types.py``
  （``evaluate_multiple_choice``）。本脚本**逐字调用官方函数**，不改写判分逻辑、
  阈值与聚合公式。

官方聚合规则（CV-Bench 数据卡 + ``tasks/cv_bench/evaluate.py``）::

    accuracy_2d_ade  = Acc(ADE20K)
    accuracy_2d_coco = Acc(COCO)
    accuracy_3d_omni = Acc(Omni3D)
    accuracy_2d = (accuracy_2d_ade + accuracy_2d_coco) / 2
    accuracy_3d = accuracy_3d_omni
    accuracy   = (accuracy_2d + accuracy_3d) / 2      # 百分制

提示词
------
严格复刻官方链（见 ``common.prompt_cv_bench_flag_eval_mm`` 的逐行出处）：
``VqaBaseDataset.build_prompt`` → ``PromptTemplate.default_prompt_func`` →
``model_zoo/vlm/robobrain/model_adapter.py`` 的 ``<think></think><answer>``。

答案提取规则（冻结，验收要求 §5.1）
-----------------------------------
1. 官方 RoboBrain 适配器 ``_process_single_result``：若输出含 ``</think>``，用官方
   ``parse_think_answer_string`` 取 ``<answer>...</answer>``；否则整段作为答案。
2. **已披露的健壮性补丁**：官方 ``parse_think_answer_string`` 在没有
   ``<answer>`` 标签时返回 ``None``，下游 ``evaluate_multiple_choice`` 会对
   ``None`` 取 ``len()`` 抛 TypeError。本脚本在这种情况下退回整段原始输出，
   其余情况与官方完全一致。
3. 最终字母判定一律交给官方 ``evaluate_multiple_choice``（``maybe_clean_answer``
   → ``process_multiple_choice``），**不人工挑选、不改写选项抽取规则**。
4. 原始回复全量保留在逐题记录里。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C          # noqa: E402
import datasets as D        # noqa: E402
import engine as E          # noqa: E402

FLAG_EVAL_MM = os.path.join(C.WORK_DIR, "third_party/upstream_eval/FlagEvalMM")
OFFICIAL_EVAL = os.path.join(FLAG_EVAL_MM, "tasks/cv_bench/evaluate.py")
ADAPTER = os.path.join(FLAG_EVAL_MM, "model_zoo/vlm/robobrain/model_adapter.py")


# --------------------------------------------------------------------------- #
# 官方答案提取
# --------------------------------------------------------------------------- #
def load_official_parse_think():
    """加载官方 ``parse_think_answer_string``（逐字，不改写）。"""
    import importlib.util
    code = (
        "import re\n"
        "def parse_think_answer_string(text_string):\n"
        "    think_content = None\n"
        "    answer_content = None\n"
        "    think_match = re.search(r'<think>(.*?)</think>', text_string, re.DOTALL)\n"
        "    if think_match:\n"
        "        think_content = think_match.group(1).strip()\n"
        "    answer_match = re.search(r'<answer>(.*?)</answer>', text_string, re.DOTALL)\n"
        "    if answer_match:\n"
        "        answer_content = answer_match.group(1).strip()\n"
        "    return think_content, answer_content\n"
    )
    ns: Dict[str, Any] = {}
    exec(compile(code, "<official_parse_think_answer_string>", "exec"), ns)
    return ns["parse_think_answer_string"]


def extract_answer(raw: str, parse_think) -> Dict[str, Any]:
    """官方适配器的答案抽取 + 已披露的 None 兜底。"""
    reason = ""
    answer = raw
    if "</think>" in (raw or ""):
        reason, answer = parse_think(raw)
    fallback = False
    if answer is None:
        answer = raw
        fallback = True
    return {"answer": answer, "reason": reason, "used_fallback": fallback}


# --------------------------------------------------------------------------- #
# 推理
# --------------------------------------------------------------------------- #
def build_jobs(limit: int, img_dir: str | None,
               task_filter: str = "", prompt_style: str = "flagevalmm") -> Dict[str, Any]:
    """构造 annotations（官方评分用）与 jobs（推理用）。"""
    annotations: Dict[str, Any] = {}
    jobs: List[Dict[str, Any]] = []
    mapfile = (os.path.join(os.path.dirname(img_dir), "cvbench_image_map.json")
               if img_dir else None)
    imgmap: Dict[str, str] = {}
    if mapfile and os.path.exists(mapfile):
        for row in json.load(open(mapfile, encoding="utf-8")):
            imgmap[row["question_id"]] = row["file"]
    keep = {t.strip() for t in (task_filter or "").split(",") if t.strip()}

    def make_prompt(ann: Dict[str, Any]) -> str:
        if prompt_style == "dataset":
            return C.prompt_cv_bench_dataset(ann.get("prompt_official") or "")
        return C.prompt_cv_bench_flag_eval_mm(ann["question"], ann["options"])

    for ann, data in D.iter_cvbench():
        if limit and len(jobs) >= limit:
            break
        if keep and (ann.get("task") or "") not in keep:
            continue
        qid = ann["question_id"]
        annotations[qid] = {
            "question_id": qid,
            "answer": ann["answer"],
            "source": ann["source"],
            "type_ori": ann["type_ori"],
            "task": ann["task"],
            "question": ann["question"],
            "options": ann["options"],
            "filename": ann["filename"],
        }
        path = imgmap.get(qid)
        if path and os.path.exists(path):
            refs = [path]
        else:
            refs = [("bytes", data)]
        jobs.append({
            "sample_id": f"CVBench/{qid}",
            "split": ann["type_ori"],
            "image_paths": refs,
            "prompt": make_prompt(ann),
            "fields": annotations[qid],
            "task": None,
        })
    return {"annotations": annotations, "jobs": jobs}


# --------------------------------------------------------------------------- #
# 判分：官方 tasks/cv_bench/evaluate.py::get_result
# --------------------------------------------------------------------------- #
def score_official(out_dir: str, annotations_path: str,
                   predictions_path: str) -> Dict[str, Any]:
    code = (
        "import sys, json, os\n"
        "sys.path.insert(0, %r)\n"
        "sys.path.insert(0, %r)\n"
        "import official_flagevalmm as OF\n"
        "mod = OF.load_cv_bench_evaluate()\n"
        "annotations = json.load(open(%r, encoding='utf-8'))\n"
        "predictions = json.load(open(%r, encoding='utf-8'))\n"
        "res = mod.get_result(annotations, predictions)\n"
        "res['scored_predictions'] = predictions\n"
        "json.dump(res, open('cv_bench_official_result.json', 'w', encoding='utf-8'),\n"
        "          ensure_ascii=False, indent=2, default=str)\n"
        "print(json.dumps({k: v for k, v in res.items() if k != 'scored_predictions'},\n"
        "                 ensure_ascii=False, indent=2, default=str))\n"
        % (FLAG_EVAL_MM, os.path.dirname(os.path.abspath(__file__)),
           annotations_path, predictions_path)
    )
    tmp = os.path.join(out_dir, "_score_cv_bench.py")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(code)
    env = dict(os.environ)
    env["PYTHONPATH"] = (env.get("PYTHONPATH", "") + os.pathsep + FLAG_EVAL_MM +
                          os.pathsep + os.path.join(C.WORK_DIR, "envs/pylibs_core"))
    env["PYTHONWARNINGS"] = "ignore"
    r = subprocess.run([sys.executable, tmp], cwd=out_dir, env=env,
                       capture_output=True, text=True, timeout=7200)
    out: Dict[str, Any] = {"cmd": "python3 %s" % tmp, "cwd": out_dir,
                            "returncode": r.returncode,
                            "stdout": r.stdout.strip(),
                            "stderr": r.stderr.strip()[-1500:]}
    sp = os.path.join(out_dir, "cv_bench_official_result.json")
    if os.path.exists(sp):
        res = json.load(open(sp, encoding="utf-8"))
        out["official_result"] = {k: v for k, v in res.items()
                                  if k != "scored_predictions"}
        out["scored_predictions"] = res.get("scored_predictions")
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=C.DEFAULT_URL)
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--out-dir",
                    default=os.path.join(C.WORK_DIR, "outputs/bench_public/cvbench"))
    ap.add_argument("--img-dir",
                    default=os.path.join(C.WORK_DIR,
                                         "outputs/bench_public/manifest/cvbench_images"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=C.MAX_NEW_TOKENS)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--run-id", default="cvbench-baseline")
    ap.add_argument("--no-guard", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--image-protocol", default="canonical_png",
                    choices=list(C.IMAGE_PROTOCOLS),
                    help="图像输入协议：预缩放(qwen_vl_utils) × 编码(PNG/JPEG q95)。"
                         "官方 FlagEvalMM 路径 = raw_jpeg95")
    ap.add_argument("--task-filter", default="",
                    help="只跑指定 task（如 Count）；逗号分隔，空=全部")
    ap.add_argument("--prompt-style", default="flagevalmm",
                    choices=["flagevalmm", "dataset"],
                    help="提示词口径：flagevalmm=官方 FlagEvalMM build_prompt 逐字复刻"
                         "（默认）；dataset=CV-Bench 数据集自带的 prompt 字段"
                         "（Cambrian-1 原始口径）")
    args = ap.parse_args()

    C.set_image_protocol(args.image_protocol)
    print("[cvbench] 图像输入协议 = %s" % args.image_protocol)

    if not args.no_guard:
        try:
            C.guard_no_official_bench()
        except RuntimeError as exc:
            print("[cvbench] %s" % exc)
            return 3
    if not C.service_alive(args.url):
        print("[cvbench] 服务不可用：%s" % args.url)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    built = build_jobs(args.limit, args.img_dir, args.task_filter, args.prompt_style)
    annotations = built["annotations"]
    jobs = built["jobs"]
    print("[cvbench] 题目数 %d，分组 %s" %
          (len(jobs), _group_hist(annotations)), flush=True)

    parse_think = load_official_parse_think()

    def parse_fn(raw: str):
        ex = extract_answer(raw, parse_think)
        return ex if ex["answer"] else None

    pred_path = os.path.join(args.out_dir, "predictions", "cvbench.jsonl")
    recs = E.run_jobs(
        jobs, pred_path, args.url, args.model,
        max_tokens=args.max_tokens, concurrency=args.concurrency,
        resume=not args.no_resume, parse_fn=parse_fn, label="[cvbench]",
    )
    by_id = {r["sample_id"].split("/", 1)[1]: r for r in recs}

    # 构造官方 predictions（question_id + answer）
    predictions = []
    per_sample = []
    n_fallback = 0
    for qid, ann in annotations.items():
        r = by_id.get(qid, {})
        raw = r.get("raw_output", "")
        ex = extract_answer(raw, parse_think)
        if ex["used_fallback"]:
            n_fallback += 1
        predictions.append({"question_id": qid, "answer": ex["answer"]})
        per_sample.append({
            "question_id": qid,
            "source": ann["source"],
            "type_ori": ann["type_ori"],
            "task": ann["task"],
            "gt_answer": ann["answer"],
            "extracted_answer": ex["answer"],
            "used_fallback": ex["used_fallback"],
            "raw_output": raw,
            "terminal_state": r.get("terminal_state"),
            "prompt": r.get("prompt"),
        })
    C.dump_json(os.path.join(args.out_dir, "annotations.json"), annotations)
    C.dump_json(os.path.join(args.out_dir, "predictions_official.json"), predictions)
    C.dump_json(os.path.join(args.out_dir, "per_sample.json"), per_sample)

    sc = score_official(args.out_dir,
                        os.path.join(args.out_dir, "annotations.json"),
                        os.path.join(args.out_dir, "predictions_official.json"))
    summary = {
        "run_id": args.run_id,
        "dataset": "CV-Bench",
        "dataset_source": D.DATASET_SOURCES["CV-Bench"],
        "protocol": {
            "decode": C.DECODE,
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "images_per_sample": C.IMAGES_PER_SAMPLE,
            "prompt_chain": "VqaBaseDataset.build_prompt -> PromptTemplate -> "
                            "RoboBrain adapter (<think></think><answer>)",
            "answer_extraction": "官方 parse_think_answer_string + "
                                  "evaluate_multiple_choice（None 兜底已披露）",
            "image_protocol": args.image_protocol,
            "image_protocol_note": "canonical=先用上游 qwen_vl_utils 做 28 倍数重采样；"
                                   "raw=不预缩放；jpeg95=官方 FlagEvalMM encode_image 的"
                                   " JPEG q95 编码；png=无损 PNG；"
                                   "up2/up4=上采样（LANCZOS，总像素≤3,000,000）",
            "prompt_style": args.prompt_style,
            "prompt_style_note": "flagevalmm=FlagEvalMM build_prompt 逐字复刻；"
                                 "dataset=CV-Bench 数据集自带 prompt 字段"
                                 "（'(A) x' 标记 + 'Select from the following choices.'）",
            "task_filter": args.task_filter or "(全部)",
        },
        "scorer": "FlagEvalMM/tasks/cv_bench/evaluate.py::get_result (official)",
        "n_planned": len(jobs),
        "n_returned": sum(1 for r in recs if r.get("raw_output")),
        "n_fallback_extraction": n_fallback,
        "terminal_states": _count_states(recs),
        "scores": sc,
        "official_reference": {
            "source": "RoboBrain2.5 技术报告 Table 2（NV 训练变体）",
            "cv_bench": 94.58,
            "note": "论文分数是对照信息；协议未完全确认一致前不作通过阈值",
        },
    }
    C.dump_json(os.path.join(args.out_dir, "summary.json"), summary)
    print("\n[cvbench] 官方结果：")
    print(sc.get("stdout", "")[:2000])
    print("\n[cvbench] 汇总 -> %s" % os.path.join(args.out_dir, "summary.json"))
    return 0


def _group_hist(annotations: Dict[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for ann in annotations.values():
        out[ann["source"]] = out.get(ann["source"], 0) + 1
    return out


def _count_states(recs: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in recs:
        out[r["terminal_state"]] = out.get(r["terminal_state"], 0) + 1
    return out


if __name__ == "__main__":
    sys.exit(main())
