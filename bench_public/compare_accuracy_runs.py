#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""精度回归对比：参考配置 vs 候选配置（同样本、同协议）。

用途：验收要求 §6.7「候选配置复测：同样本、同评分协议全量重跑」，
以及 §7.3「参考与候选配置可比；门槛、波动及指标方向明确」。

对比内容：
1. 三个基准的**逐指标**得分差（方向已标注：越高越好 / 越低越好）；
2. **逐题**输出是否逐字符一致，变化比例；
3. 变化样本上的**得分影响**。

用法::

    python3 submission/bench_public/compare_accuracy_runs.py \
        --reference outputs/bench_public \
        --candidate outputs/bench_public_noneager \
        --out outputs/bench_public/accuracy_regression.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def _load(p: str) -> Optional[Any]:
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def _preds(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        sid = r.get("sample_id")
        if sid:
            out[sid] = (r.get("raw_output") or "").strip()
    return out


def compare_refspatial(ref: str, cand: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rs = _load(os.path.join(ref, "refspatial", "summary.json"))
    cs = _load(os.path.join(cand, "refspatial", "summary.json"))
    if not rs or not cs:
        return rows
    for variant in rs.get("variants", {}):
        if variant not in cs.get("variants", {}):
            continue
        for split, si in rs["variants"][variant].get("splits", {}).items():
            ci = cs["variants"][variant].get("splits", {}).get(split)
            if not ci:
                continue
            r_acc = si.get("scores", {}).get("official_cli", {}).get("official_accuracy")
            c_acc = ci.get("scores", {}).get("official_cli", {}).get("official_accuracy")
            rows.append({
                "benchmark": "RefSpatial-Bench", "variant": variant, "split": split,
                "metric": "accuracy (mask 命中, 越高越好)",
                "reference": r_acc, "candidate": c_acc,
                "delta": (None if r_acc is None or c_acc is None
                          else round(c_acc - r_acc, 6)),
                "unit": "absolute",
            })
            rp = _preds(os.path.join(ref, "refspatial", variant, "predictions",
                                     f"{split}.jsonl"))
            cp = _preds(os.path.join(cand, "refspatial", variant, "predictions",
                                     f"{split}.jsonl"))
            common = set(rp) & set(cp)
            if common:
                diff = [k for k in common if rp[k] != cp[k]]
                rows.append({
                    "benchmark": "RefSpatial-Bench", "variant": variant,
                    "split": split,
                    "metric": "逐题原始输出变化率 (参考 vs 候选)",
                    "reference": 0.0, "candidate": round(len(diff) / len(common), 6),
                    "delta": round(len(diff) / len(common), 6),
                    "unit": "ratio", "n": len(common), "changed": len(diff),
                })
    return rows


def compare_tracespatial(ref: str, cand: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rs = _load(os.path.join(ref, "tracespatial", "summary.json"))
    cs = _load(os.path.join(cand, "tracespatial", "summary.json"))
    if not rs or not cs:
        return rows
    keys = [("start_success_rate", "3D Start Success (越高越好)"),
            ("end_success_rate", "3D End Success (越高越好)"),
            ("overall_success_rate", "3D Overall Success (越高越好)"),
            ("start_in_mask", "2D Start-in-Mask (越高越好)"),
            ("end_in_bbox", "2D End-in-BBox (越高越好)"),
            ("no_collision_rate", "No-Collision Rate (越高越好)"),
            ("start_distance", "3D Start 距离 (m, 越低越好)"),
            ("end_distance", "3D End 距离 (m, 越低越好)")]
    for variant in rs.get("variants", {}):
        if variant not in cs.get("variants", {}):
            continue
        for task, ti in rs["variants"][variant].get("tasks", {}).items():
            ci = cs["variants"][variant].get("tasks", {}).get(task)
            if not ci:
                continue
            for k, label in keys:
                r = ti.get("scores", {}).get(k)
                c = ci.get("scores", {}).get(k)
                if r is None or c is None:
                    continue
                rows.append({
                    "benchmark": "TraceSpatial-Bench", "variant": variant,
                    "split": task, "metric": label,
                    "reference": r, "candidate": c,
                    "delta": round(c - r, 6), "unit": "absolute",
                })
            # 路径指标（DFD / HD / RMSE / MAE）：越低越好
            for mk, mlabel in (("metrics_2d", "2D 路径指标"), ("metrics_3d", "3D 路径指标")):
                rd = ti.get("scores", {}).get(mk)
                cd = ci.get("scores", {}).get(mk)
                if not isinstance(rd, dict) or not isinstance(cd, dict):
                    continue
                for mk2 in ("average_discrete_frechet_distance",
                            "average_hausdorff_distance",
                            "average_root_mean_square_error",
                            "average_mean_absolute_error"):
                    if mk2 not in rd or mk2 not in cd:
                        continue
                    rows.append({
                        "benchmark": "TraceSpatial-Bench", "variant": variant,
                        "split": task,
                        "metric": "%s / %s (越低越好)" % (mlabel, mk2),
                        "reference": rd[mk2], "candidate": cd[mk2],
                        "delta": round(cd[mk2] - rd[mk2], 6), "unit": "absolute",
                    })
    return rows


def compare_cvbench(ref: str, cand: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rs = _load(os.path.join(ref, "cvbench", "summary.json"))
    cs = _load(os.path.join(cand, "cvbench", "summary.json"))
    if not rs or not cs:
        return rows
    rr = rs.get("scores", {}).get("official_result", {}) or {}
    cr = cs.get("scores", {}).get("official_result", {}) or {}
    labels = {"accuracy_2d_ade": "Acc(ADE20K)", "accuracy_2d_coco": "Acc(COCO)",
              "accuracy_3d_omni": "Acc(Omni3D)", "accuracy_2d": "Acc_2D",
              "accuracy_3d": "Acc_3D", "accuracy": "CV-Bench 综合分"}
    for k, label in labels.items():
        if k not in rr or k not in cr:
            continue
        rows.append({
            "benchmark": "CV-Bench", "variant": "-", "split": "all",
            "metric": f"{label} (越高越好, %)",
            "reference": rr[k], "candidate": cr[k],
            "delta": round(cr[k] - rr[k], 4), "unit": "percentage point",
        })
    rp = _preds(os.path.join(ref, "cvbench", "predictions", "cvbench.jsonl"))
    cp = _preds(os.path.join(cand, "cvbench", "predictions", "cvbench.jsonl"))
    common = set(rp) & set(cp)
    if common:
        diff = [k for k in common if rp[k] != cp[k]]
        rows.append({
            "benchmark": "CV-Bench", "variant": "-", "split": "all",
            "metric": "逐题原始输出变化率 (参考 vs 候选)",
            "reference": 0.0, "candidate": round(len(diff) / len(common), 6),
            "delta": round(len(diff) / len(common), 6),
            "unit": "ratio", "n": len(common), "changed": len(diff),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=os.path.join(C.WORK_DIR,
                                                        "outputs/bench_public"))
    ap.add_argument("--candidate", default=os.path.join(C.WORK_DIR,
                                                        "outputs/bench_public_noneager"))
    ap.add_argument("--out", default=os.path.join(
        C.WORK_DIR, "outputs/bench_public/accuracy_regression.md"))
    args = ap.parse_args()

    rows = (compare_refspatial(args.reference, args.candidate) +
            compare_tracespatial(args.reference, args.candidate) +
            compare_cvbench(args.reference, args.candidate))
    if not rows:
        print("[compare] 无可比较数据（候选目录是否已跑完？）")
        return 1

    head = ["基准", "变体/子集", "指标", "参考值", "候选值", "差值", "单位", "判定"]
    lines = ["# 精度回归对比（参考配置 vs 候选配置）", "",
             "> 同样本、同提示词、同解码参数、同判分脚本；**唯一变量**是服务配置"
             "（参考 = 首次提交基线 eager；候选 = E1 非 eager）。", "",
             "> 「允许下降」未预先登记，按验收要求 §7.3，精度回归结论为 "
             "**BLOCKED（阈值未定义）**；下表只报告差值与变化比例，不判定通过。", "",
             "| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        judge = "—" if r["metric"].startswith("逐题") else "未定义阈值"
        lines.append("| %s | %s/%s | %s | %s | %s | %s | %s | %s |" % (
            r["benchmark"], r["variant"], r["split"], r["metric"],
            r["reference"], r["candidate"], r["delta"], r["unit"], judge))

    out_rows = [r for r in rows if not r["metric"].startswith("逐题")]
    worst = min(out_rows, key=lambda r: r["delta"]) if out_rows else None
    lines += ["", "## 小结", ""]
    if worst:
        lines.append("- 最大退化项：**%s / %s / %s**，差值 **%s %s**" %
                      (worst["benchmark"], worst["variant"] + "/" + worst["split"],
                       worst["metric"], worst["delta"], worst["unit"]))
    chg = [r for r in rows if r["metric"].startswith("逐题")]
    for r in chg:
        lines.append("- %s / %s：逐题原始输出变化 **%s**（%s/%s 题不同）" %
                     (r["benchmark"], r["variant"] + "/" + r["split"],
                      r["candidate"], r.get("changed"), r.get("n")))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    C.dump_json(args.out.replace(".md", ".json"), rows)
    print("[compare] -> %s" % args.out)
    print("\n".join(lines[8:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
