#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TraceSpatial-Bench 评测（官方 RoboTracer 判分脚本，逐字运行）。

上游官方件
----------
* 数据：``JingkunAn/TraceSpatial-Bench``（HF）→ ``trajectory_dataset.json``（100 条）
        + ``raw_data/<id>/{image,gt_depth,depth,mask}.png|jpg``
* 推理入口：``RoboTracer/Evaluation/test_benchmark.py``
* **判分**：``RoboTracer/Evaluation/summarize_acc.py``（本脚本**逐字调用其 CLI**，
  不改写判分逻辑、阈值、插值与聚合规则）

模型名约定
----------
官方 ``summarize_acc.py`` 的解析分派只认 ``"RoboTracer"`` / ``"Qwen"`` /
``"Gemini"`` 三个关键词，其它名字会直接 ``raise ValueError``。我们的模型是
RoboBrain2.5-8B-NV，但它的输出格式与官方 ``RoboTracer`` 分支完全一致
（元组 ``[(x,y)]`` / ``[(x,y,d)]``，坐标 0~1000），因此把 **model_id 取成
``RoboTracer_RoboBrain2.5-8B-NV``**，让官方判分脚本走它为元组输出准备的
``text2pts_normalize(..., is_normalized_1000=True)`` 分支 —— **判分代码零改动**。

提示词两个因子
--------------
* ``official``：RoboTracer 官方 ``get_prompt`` 的 "RoboTracer" 分支（2D/3D 各自原文）。
  上游自带的参考预测 ``Model_Prediction_TraceSpatial_{2D,3D}.jsonl`` 用的就是它。
* ``upstream``：RoboBrain2.5 上游 ``inference.py`` 的 trajectory 模板。

坐标尺度
--------
官方 2D/3D 判分都用 ``is_normalized_1000=True``（x,y ÷1000 → [0,1]），与模型
原生 0~1000 输出**天然吻合，无需换算**。深度 d 单位为米，官方按米制判 3D 距离。

3D 判分与 open3d
-----------------
官方 ``compute_accuracy_3d`` 的**碰撞检查**依赖 ``open3d``（TSDF 体素化 +
DBSCAN 去噪）。若 open3d 不可用，该项会被官方脚本自己捕获成
``[Warn][3D] Collision check failed`` → ``collision_flag=False``，
此时 Success 率会**偏乐观**。本脚本会检测并显式报告这一降级。
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

ROBOTRACER = os.path.join(C.WORK_DIR, "third_party/upstream_eval/RoboTracer/Evaluation")
MODEL_ID_BASE = "RoboTracer_RoboBrain2.5-8B-NV"
TASKS = ("2D", "3D")
# open3d 的依赖链带 numpy 2.x，与系统 numpy 1.25（torch 编译基线）冲突，
# 因此 open3d 单独放在 envs/pylibs_o3d，**只**给官方判分子进程使用。
O3D_PREFIX = os.path.join(C.WORK_DIR, "envs/pylibs_o3d")
BASE_PREFIX = os.path.join(C.WORK_DIR, "envs/pylibs_core")


def scoring_pythonpath() -> str:
    """判分子进程的前缀顺序。

    1. ``envs/pylibs_o3d``：open3d + numpy 2.x（open3d 0.20 按 numpy2 ABI 编译，
       与系统 numpy 1.25 不兼容，因此 numpy 也放在这里且必须排在最前）；
    2. ``envs/pylibs``：open3d 的其余依赖（dash / pydantic 等）；
    3. ``envs/pylibs_core``：通用依赖（不含 numpy，保证其它进程用系统 numpy 1.25）。

    ⚠️ ``ROBO_O3D_PREFIX`` 可覆盖第 1 项，用于切换 open3d 版本。
    open3d 0.19/0.20 的 ``ScalableTSDFVolume`` 改了体素语义（新增
    ``volume_unit_resolution`` / ``depth_sampling_stride``），在官方脚本给定的
    ``voxel_length=0.02, sdf_trunc=0.08`` 下 **抽不出任何点**，会让官方 3D 碰撞
    检查必然抛 ValueError（碰撞标志恒 False、No-Collision 恒 1.0）。详见
    ``reports/16_public_bench_results.md`` §5.6。
    """
    o3d_prefix = os.environ.get("ROBO_O3D_PREFIX") or O3D_PREFIX
    return os.pathsep.join([
        o3d_prefix,
        os.path.join(C.WORK_DIR, "envs/pylibs"),
        BASE_PREFIX,
        os.environ.get("PYTHONPATH", ""),
    ])


def build_prompt(variant: str, task_name: str, raw_prompt: str,
                 target_object: str, destination: str) -> str:
    if variant == "official":
        return C.prompt_tracespatial_official(task_name, target_object,
                                              destination, raw_prompt)
    if variant == "upstream":
        return C.prompt_upstream(C.strip_one_trailing_period(raw_prompt),
                                 "trajectory")
    raise ValueError(variant)


def model_id_for(variant: str) -> str:
    return f"{MODEL_ID_BASE}" if variant == "official" else f"{MODEL_ID_BASE}_{variant}"


def run_variant(variant: str, tasks: List[str], args) -> Dict[str, Any]:
    out_root = os.path.join(args.out_dir, variant)
    os.makedirs(out_root, exist_ok=True)
    mid = model_id_for(variant)
    info: Dict[str, Any] = {"variant": variant, "model_id": mid, "tasks": {}}

    for task_name in tasks:
        samples = D.load_tracespatial(task_name)
        if args.limit:
            samples = samples[:args.limit]
        jobs = []
        for s in samples:
            s = dict(s)
            s["prompt"] = build_prompt(variant, task_name, s["raw_prompt"],
                                       s["target_object"], s["destination"])
            jobs.append(s)

        pred_path = os.path.join(out_root, "predictions", f"{task_name}.jsonl")
        # 2D 任务的输出可能是 2 元组（official 变体）或 3 元组（upstream 变体），
        # 用 trajectory_2d 模式两种都接受，避免把合法输出误判成 parse_failed。
        parse_task = "trajectory_2d" if task_name == "2D" else "trajectory"
        recs = E.run_jobs(
            jobs, pred_path, args.url, args.model,
            max_tokens=args.max_tokens, concurrency=args.concurrency,
            resume=not args.no_resume,
            parse_fn=lambda t: C.parse_points_1000(t, parse_task),
            label=f"[tracespatial {variant}/{task_name}]",
        )
        by_id = {r["sample_id"]: r for r in recs}

        # 官方 eval_task 的 jsonl 字段（RoboTracer/test_benchmark.py:153-168）
        model_dir = os.path.join(out_root, "outputs")
        os.makedirs(model_dir, exist_ok=True)
        official_jsonl = os.path.join(model_dir, f"{mid}_{task_name}.jsonl")
        with open(official_jsonl, "w", encoding="utf-8") as fh:
            for s in jobs:
                r = by_id.get(s["sample_id"], {})
                rec = dict(s["fields"])
                rec["final_prompt"] = s["prompt"]
                rec["model_prediction"] = r.get("raw_output", "")
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        info["tasks"][task_name] = {
            "n": len(samples),
            "predictions": pred_path,
            "official_jsonl": official_jsonl,
            "terminal_states": _count_states(recs),
        }
    return info


def _count_states(recs: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in recs:
        out[r["terminal_state"]] = out.get(r["terminal_state"], 0) + 1
    return out


def score_official_cli(out_dir: str, variant: str, task_name: str) -> Dict[str, Any]:
    """逐字运行官方 summarize_acc.py CLI（串行版，供单点调用使用）。"""
    return score_official_cli_collect(
        score_official_cli_launch(out_dir, variant, task_name))


def score_official_cli_launch(out_dir: str, variant: str, task_name: str):
    """启动官方 summarize_acc.py CLI（不等待）。

    官方 3D 判分是**单进程单核**的（TSDF 体素化 + DBSCAN 去噪），
    单变体实测约 45 分钟。本机的 3D 判分（2 个变体 × 2 个任务 = 4 次）
    彼此**完全独立**——官方脚本只读 ``outputs/<model_id>_<task>.jsonl`` 并
    往 stdout 打印聚合指标，不写任何文件——因此可以**同时**启动 4 个官方 CLI
    进程并行判分。

    ⚠️ 被调用的仍然是**未经改动的官方 ``summarize_acc.py``**（逐字 CLI 调用），
    这里只改变**并发调度**，不改变判分逻辑、阈值、插值与聚合规则。
    """
    run_dir = os.path.join(out_dir, variant)
    link = os.path.join(run_dir, "TraceSpatial-Bench")
    if not os.path.exists(link):
        os.symlink(D.TRACESPATIAL_DIR, link)
    mid = model_id_for(variant)
    env = dict(os.environ)
    env["PYTHONPATH"] = (scoring_pythonpath() + os.pathsep + ROBOTRACER)
    env["PYTHONWARNINGS"] = "ignore"
    cmd = [sys.executable, os.path.join(ROBOTRACER, "summarize_acc.py"),
           "--model_name", mid, "--task_name", task_name]
    proc = subprocess.Popen(cmd, cwd=run_dir, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    return {"proc": proc, "cmd": " ".join(cmd), "cwd": run_dir}


def score_official_cli_collect(handle) -> Dict[str, Any]:
    """等待一次官方 CLI 判分结束并解析其 stdout。"""
    proc = handle["proc"]
    stdout, stderr = proc.communicate(timeout=7200)
    out = {"cmd": handle["cmd"], "cwd": handle["cwd"], "returncode": proc.returncode,
           "stdout": stdout, "stderr": stderr[-1500:]}
    out["metrics_2d"] = _grab(out["stdout"], "2D Path Metrics (Normalized [0,1])")
    out["start_in_mask"] = _grab(out["stdout"], "2D Start-in-Mask Rate")
    out["end_in_bbox"] = _grab(out["stdout"], "2D End-in-2D-BBox Rate")
    out["metrics_3d"] = _grab(out["stdout"], "3D Path Metrics")
    out["start_distance"] = _grab(out["stdout"], "Start Point Distance (mean)")
    out["end_distance"] = _grab(out["stdout"], "End Point Distance (mean)")
    out["no_collision_rate"] = _grab(out["stdout"], "No-Collision Rate")
    out["start_success_rate"] = _grab(out["stdout"], "Start Success Rate")
    out["end_success_rate"] = _grab(out["stdout"], "End Success Rate")
    out["overall_success_rate"] = _grab(out["stdout"], "Overall Success Rate")
    out["collision_warnings"] = out["stdout"].count("[Warn][3D] Collision check failed")
    out["skip_depthless"] = out["stdout"].count("[Skip] Prediction has no depth")
    out["skip_parse"] = out["stdout"].count("[Skip] Failed to parse prediction")
    out["open3d_available"] = "import open3d" in (out["stdout"] + out["stderr"]) or \
        ("Collision check failed" not in out["stdout"])
    return out


def _grab(text: str, key: str):
    """取以 ``key`` 开头的行，去掉前缀后剩余部分。

    ⚠️ 官方打印成 ``Key: value``，截取后必须去掉行首冒号，
    否则 ``": 0.7300"`` 不能直接当数值用。
    """
    for line in text.splitlines():
        if line.startswith(key):
            return line[len(key):].lstrip(":").strip()
    return None


def check_open3d() -> Dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONPATH"] = scoring_pythonpath()
    code = ("import importlib.util as u;"
            "print('open3d_available =', u.find_spec('open3d') is not None);"
            "import numpy; print('numpy =', numpy.__version__)")
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True, text=True, timeout=120)
    return {"stdout": r.stdout.strip(), "stderr": r.stderr.strip()[-300:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=C.DEFAULT_URL)
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--out-dir",
                    default=os.path.join(C.WORK_DIR, "outputs/bench_public/tracespatial"))
    ap.add_argument("--variants", default="official,upstream")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=C.MAX_NEW_TOKENS)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--run-id", default="tracespatial-baseline")
    ap.add_argument("--no-guard", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--stage", default="all", choices=("all", "infer", "score"),
                    help="all=推理+判分；infer=只推理并落盘；score=只对已落盘的"
                         "预测跑官方判分（可与 rescore 并行）")
    args = ap.parse_args()

    print("[tracespatial] 提示词一致性自检")
    if not C.selfcheck_prompt_parity():
        print("[tracespatial] 警告：上游提示词与 robo_infer 不一致，仍继续但请人工确认")

    # --stage score 不发任何请求，不需要服务在线，也不需要压测安全闸
    if args.stage != "score":
        if not args.no_guard:
            try:
                C.guard_no_official_bench()
            except RuntimeError as exc:
                print("[tracespatial] %s" % exc)
                return 3

        if not C.service_alive(args.url):
            print("[tracespatial] 服务不可用：%s" % args.url)
            return 1

    o3d = check_open3d()
    print("[tracespatial] open3d 状态: %s" % o3d.get("stdout"))

    os.makedirs(args.out_dir, exist_ok=True)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    summary_path = os.path.join(args.out_dir, "summary.json")

    if args.stage == "score":
        # 只判分：沿用已落盘的 summary（推理阶段写好的）
        if not os.path.exists(summary_path):
            print("[tracespatial] --stage score 需要先跑 --stage infer（缺 %s）"
                  % summary_path)
            return 1
        summary = json.load(open(summary_path, encoding="utf-8"))
        missing = [v for v in variants if v not in summary.get("variants", {})]
        if missing:
            print("[tracespatial] summary 缺少变体 %s，请先跑 --stage infer" % missing)
            return 1
        for v in variants:
            for t in tasks:
                oj = os.path.join(args.out_dir, v, "outputs",
                                  "%s_%s.jsonl" % (model_id_for(v), t))
                if not os.path.exists(oj):
                    print("[tracespatial] 缺官方输入 %s" % oj)
                    return 1
    else:
        summary = {
            "run_id": args.run_id,
            "dataset": "TraceSpatial-Bench",
            "dataset_source": D.DATASET_SOURCES["TraceSpatial-Bench"],
            "protocol": {
                "decode": C.DECODE,
                "max_tokens": args.max_tokens,
                "concurrency": args.concurrency,
                "images_per_sample": C.IMAGES_PER_SAMPLE,
                "coordinate_contract": "0~1000 (x,y) + depth in meters; no scaling needed",
                "model_input_depth_or_intrinsics": False,
                "note": "深度与内参仅判分需要，不作为模型输入",
            },
            "scorer": "RoboTracer/Evaluation/summarize_acc.py (official CLI, verbatim)",
            "open3d": o3d,
            "variants": {},
        }
        for variant in variants:
            print("\n########## variant=%s ##########" % variant, flush=True)
            summary["variants"][variant] = run_variant(variant, tasks, args)
        # 推理阶段先落一次 summary，便于 --stage score / rescore 并行复用
        C.dump_json(summary_path, summary)

    if args.stage in ("all", "score"):
        # ---- 官方判分：4 个 (变体 × 任务) 互相独立，并行启动 ----
        pending = []
        for variant in variants:
            for task_name in tasks:
                print("[tracespatial] 启动官方判分 %s/%s ..." % (variant, task_name),
                      flush=True)
                pending.append((variant, task_name,
                                score_official_cli_launch(args.out_dir, variant,
                                                          task_name)))
        for variant, task_name, h in pending:
            sc = score_official_cli_collect(h)
            summary["variants"][variant]["tasks"][task_name]["scores"] = sc
            print("[tracespatial] %s/%s" % (variant, task_name), flush=True)
            for k in ("metrics_2d", "start_in_mask", "end_in_bbox", "metrics_3d",
                      "start_success_rate", "end_success_rate",
                      "overall_success_rate", "no_collision_rate",
                      "collision_warnings", "skip_depthless"):
                if sc.get(k) is not None:
                    print("    %-22s %s" % (k, sc[k]), flush=True)
        C.dump_json(summary_path, summary)

    print("\n[tracespatial] 汇总 -> %s" % summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
