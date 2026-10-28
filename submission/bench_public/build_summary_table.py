#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成公开基准评测汇总表（验收要求 §8 的表格格式）。

输出 ``summary_table.md``，列为：
``基准/指标 | 计划题数 | 有效返回数 | 实际计分数 | 请求/解析失败数 | 参考值 | 本次值 | 差值/单位 | 允许下降 | 判定``

参考值来源：
* 论文：``material/robo_2601.14352v1.pdf`` §6.1/§6.2 Table 2/3（NV 变体）
* 同数据集官方参照：上游自带参考预测经官方 ``summarize_acc.py`` 现场打分

用法::

    python3 submission/bench_public/build_summary_table.py \
        --bench-root outputs/bench_public --out outputs/bench_public/summary_table.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

PAPER = {
    "CV-Bench": ("94.58", "%", "论文 Table 2（NV 变体）"),
    "RefSpatial": ("60.50", "%", "论文 Table 2（NV 变体）"),
    "TraceSpatial-3D Start": ("83", "%", "论文 Table 3（NV 变体）"),
    "TraceSpatial-3D End": ("63", "%", "论文 Table 3（NV 变体）"),
    "TraceSpatial-Success": ("44", "%", "论文 Table 3（NV 变体）"),
}
# 同数据集官方参照（上游自带参考预测，用官方脚本现场打分得到）
UPSTREAM_REF = {
    "RoboTracer_Intrinsics_Depth": {
        "3D Start": 0.75, "3D End": 0.57, "Success": 0.52,
        "2D DFD": 0.2228, "2D RMSE": 0.1608, "2D Start-in-Mask": 0.58,
        "2D End-in-BBox": 0.43,
    },
    "Gemini3Pro": {
        "3D Start": 0.10, "3D End": 0.22, "Success": 0.07,
        "2D DFD": 0.2385, "2D RMSE": 0.1868, "2D Start-in-Mask": 0.60,
        "2D End-in-BBox": 0.60,
    },
}


def _load(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return None


def _pct(v) -> str:
    try:
        return f"{float(v) * 100:.2f}%"
    except Exception:
        return "—"


def _num(v, nd=4) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return "—"


def rows_refspatial(root: str) -> List[List[str]]:
    d = _load(os.path.join(root, "refspatial", "summary.json"))
    if not d:
        return [["RefSpatial-Bench", "—", "—", "—", "—", "—", "—", "—",
                 "未定义", "NOT_RUN"]]
    out = []
    for variant, vi in d.get("variants", {}).items():
        for split, si in vi.get("splits", {}).items():
            sc = si.get("scores", {}).get("official_cli", {})
            acc = sc.get("official_accuracy")
            ev = sc.get("official_evaluated")
            tot = sc.get("official_total")
            states = si.get("terminal_states", {})
            fails = sum(v for k, v in states.items() if k != "ok")
            ref = "60.50%（论文，聚合口径未公开）"
            scope = "默认范围" if split in ("Location", "Placement") \
                else "重叠子集（单列，不计入默认总分）"
            out.append([
                f"RefSpatial / {split} / {variant}（{scope}）",
                str(si.get("n")), str(tot), str(ev), str(fails),
                ref, _pct(acc), f"{_pct(acc)}（官方 CLI）", "未定义",
                "PASS（公开基准）",
            ])
    return out


def rows_tracespatial(root: str) -> List[List[str]]:
    d = _load(os.path.join(root, "tracespatial", "summary.json"))
    if not d:
        return [["TraceSpatial-Bench", "—", "—", "—", "—", "—", "—", "—",
                 "未定义", "NOT_RUN"]]
    out = []
    for variant, vi in d.get("variants", {}).items():
        for task, ti in vi.get("tasks", {}).items():
            sc = ti.get("scores", {})
            states = ti.get("terminal_states", {})
            fails = sum(v for k, v in states.items() if k != "ok")
            metrics = [
                ("3D Start", sc.get("start_success_rate"), "83%（论文）",
                 "RoboTracer_Intrinsics_Depth=75%"),
                ("3D End", sc.get("end_success_rate"), "63%（论文）",
                 "RoboTracer_Intrinsics_Depth=57%"),
                ("Success", sc.get("overall_success_rate"), "44%（论文）",
                 "RoboTracer_Intrinsics_Depth=52%"),
                ("2D Start-in-Mask", sc.get("start_in_mask"), "—",
                 "RoboTracer_Intrinsics_Depth=58%"),
                ("2D End-in-BBox", sc.get("end_in_bbox"), "—",
                 "RoboTracer_Intrinsics_Depth=43%"),
            ]
            for name, val, ref, ref2 in metrics:
                if val is None:
                    continue
                out.append([
                    f"TraceSpatial / {task} / {variant} / {name}",
                    str(ti.get("n")), str(ti.get("n")), str(ti.get("n")),
                    str(fails), ref, _pct(val),
                    f"{_pct(val)}；同集官方参照 {ref2}", "未定义",
                    "PASS（公开基准）",
                ])
            for name, key in (("2D DFD", "metrics_2d"), ("3D DFD", "metrics_3d")):
                raw = sc.get(key)
                if not raw:
                    continue
                # 官方脚本打印的是 dict 的 repr；分片合并后已经是 dict
                if isinstance(raw, dict):
                    d = raw
                else:
                    try:
                        d = eval(raw) if raw.strip().startswith("{") else None
                    except Exception:
                        d = None
                if not d:
                    continue
                v = d.get("average_discrete_frechet_distance")
                if v is None:
                    continue
                out.append([
                    f"TraceSpatial / {task} / {variant} / {name}",
                    str(ti.get("n")), str(ti.get("n")), str(ti.get("n")),
                    str(fails), "越低越好；RoboTracer_Intrinsics_Depth 见报告",
                    _num(v), _num(v), "未定义", "PASS（公开基准）",
                ])
    return out


def rows_cvbench(root: str) -> List[List[str]]:
    d = _load(os.path.join(root, "cvbench", "summary.json"))
    if not d:
        return [["CV-Bench", "—", "—", "—", "—", "—", "—", "—", "未定义",
                 "NOT_RUN"]]
    res = d.get("scores", {}).get("official_result", {}) or {}
    states = d.get("terminal_states", {})
    fails = sum(v for k, v in states.items() if k != "ok")
    n = d.get("n_planned")
    out = []
    for key, label in (("accuracy_2d_ade", "Acc(ADE20K)"),
                       ("accuracy_2d_coco", "Acc(COCO)"),
                       ("accuracy_3d_omni", "Acc(Omni3D)"),
                       ("accuracy_2d", "Acc_2D"),
                       ("accuracy_3d", "Acc_3D"),
                       ("accuracy", "CV-Bench 综合分")):
        v = res.get(key)
        if v is None:
            continue
        out.append([
            f"CV-Bench / {label}", str(n), str(d.get("n_returned")), str(n),
            str(fails), "94.58%（论文 Table 2）", f"{_num(v, 2)}%",
            f"{_num(v, 2)}%（官方 FlagEvalMM get_result）", "未定义",
            "PASS（公开基准）",
        ])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root",
                    default=os.path.join(C.WORK_DIR, "outputs/bench_public"))
    ap.add_argument("--out",
                    default=os.path.join(C.WORK_DIR, "outputs/bench_public/summary_table.md"))
    args = ap.parse_args()

    rows = (rows_refspatial(args.bench_root) + rows_tracespatial(args.bench_root)
            + rows_cvbench(args.bench_root))
    head = ["基准/指标", "计划题数", "有效返回数", "实际计分数", "请求/解析失败数",
            "参考值", "本次值", "差值/单位", "允许下降", "判定"]
    lines = ["# 公开基准评测汇总表", "",
             "> **这三个评测集都不是赛事官方验收数据集。** 本表只证明公开基准上的"
             "测试有效性与任务质量，**不能替代赛事官方 98% 一致率门槛**。", "",
             "「允许下降」一栏全部为「未定义」：本次没有预先登记公开基准的质量"
             "回归阈值，按《精度测试验收要求》§7.3，精度回归结论为 "
             "`BLOCKED（阈值未定义）`，不得自拟通过线。", "",
             "| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")

    stab = _load(os.path.join(args.bench_root, "stability", "stability.json"))
    if stab:
        lines += ["", "## 稳定性检查（每基准固定样例 × 2 次重复推理）", "",
                  "| 基准 | 样例数 | 原始输出一致 | 变化数 | 变化率 | 终态全 OK |",
                  "|---|---|---|---|---|---|"]
        for k in ("refspatial", "tracespatial", "cvbench"):
            d = stab.get(k)
            if not d:
                continue
            lines.append("| %s | %s | %s | %s | %s | %s |" %
                         (d["dataset"], d["n"], d["identical_raw"], d["changed_raw"],
                          d["change_ratio"], d["all_states_ok"]))

    rs = _load(os.path.join(args.bench_root, "rescore_check.json"))
    if rs:
        lines += ["", "## 离线重判分复现", "",
                  "- 比较指标数：**%s**；不一致：**%s**" %
                  (rs.get("summary", {}).get("compared_metrics"),
                     rs.get("summary", {}).get("mismatched")),
                  "- 可复现：**%s**" % rs.get("summary", {}).get("reproducible")]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("[summary_table] -> %s" % args.out)
    print("\n".join(lines[:14]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
