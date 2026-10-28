#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inject_accuracy_table.py — 生成并注入「精度总表」到 submission/README.md §6.2。

要解决的问题
------------
`submission/README.md` §6.2 原本只留了一个 `<!-- PUBLIC_BENCH_TABLE -->` 占位符，
**没有任何脚本负责填充**（§7 的压测表有 `tools/inject_final_report.py` 负责回填，
§6.2 没有）。于是提交给评审的正文里「见下表」下面并没有表；
真正的汇总表躺在 `outputs/bench_public/summary_table.md`，
而 `outputs/` 按 `.gitignore` 不在提交包内 —— 评审在提交包里找不到这张表。

本脚本把 §6.2 的精度数据整合成**一屏可读的总表**，一次注入：

* **表 A 主指标一览**：三个公开基准的主口径主指标（论文参考 / 本次值 / 差值）
* **表 B 口径索引**：本节每张表分别是什么口径、能不能作为上报数字
* **表 C 稳定性与可复现性**：同配置重复推理 + 离线重判分
* **表 D 全量指标明细**：全部指标 × 全部次口径（官方判分器逐字产出）

同时把 §6.2.3 的两张上采样表也改成脚本生成（原来手写），
消除「报告里的数字」与「原始 JSON 里的数字」漂移的风险。

数据来源（全部为官方判分器产出的原始 JSON，脚本内不手写任何数字）
--------------------------------------------------------------
* ``outputs/bench_public/cvbench/summary.json``             官方 FlagEvalMM ``get_result``
* ``outputs/bench_public/refspatial/summary.json``           官方 RoboRefer ``summarize_acc.py``
* ``outputs/bench_public/tracespatial/summary.json``         官方 RoboTracer ``summarize_acc.py``
* ``outputs/bench_public/stability/stability.json``          同配置重复推理
* ``outputs/bench_public/rescore_check.json``                离线重判分复现
* ``outputs/bench_public/cvbench_upscale/upscale_compare.json``
* ``outputs/bench_public/cvbench_up2_full_result.json``

主口径（与 ``reports/16_public_bench_results.md`` 一致）
------------------------------------------------------
* CV-Bench     : ``official_result``（``canonical_png`` 全量 2638，与上游 ``qwen_vl_utils`` 逐字一致）
* RefSpatial   : ``upstream_scaled`` 变体（上游 pointing 提示词 + 推理侧 0-1000→像素换算）
* TraceSpatial : ``official`` 变体（RoboTracer 官方提示词）；3D 子集分母 96/100
                 （官方判分器排除 4 条深景病态样本），2D 子集分母 100/100

用法
----
    python3 tools/inject_accuracy_table.py
    python3 tools/inject_accuracy_table.py --bench-root outputs/bench_public

幂等：重复执行只替换标记块，不追加。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

sys.path.insert(0, os.path.join(ROOT, "submission", "bench_public"))
import build_summary_table as BST  # noqa: E402

# ---- 论文参考值（RoboBrain2.5 技术报告 Table 2 / Table 3，NV 变体）----------
PAPER_CV = 94.58
PAPER_REFSPATIAL = 60.50
PAPER_TS = {"3D Start": 83, "3D End": 63, "Success": 44}

# ---- 主口径选择器 -----------------------------------------------------------
CV_MAIN = "canonical_png"
REF_MAIN_VARIANT = "upstream_scaled"
TS_MAIN_VARIANT = "official"

BLOCKS = {
    "public": ("<!-- PUBLIC_BENCH_TABLE:BEGIN -->", "<!-- /PUBLIC_BENCH_TABLE -->",
               "<!-- PUBLIC_BENCH_TABLE -->"),
    "upscale": ("<!-- UPSCALE_TABLE:BEGIN -->", "<!-- /UPSCALE_TABLE -->",
                "<!-- UPSCALE_TABLE -->"),
    "upscale_full": ("<!-- UPSCALE_FULL_TABLE:BEGIN -->", "<!-- /UPSCALE_FULL_TABLE -->",
                     "<!-- UPSCALE_FULL_TABLE -->"),
}


# =============================================================================
# 取数
# =============================================================================
def _load(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def cv_main(root: str) -> Dict[str, Any]:
    d = _load(os.path.join(root, "cvbench", "summary.json")) or {}
    res = (d.get("scores") or {}).get("official_result") or {}
    return {
        "n_planned": d.get("n_planned"),
        "n_returned": d.get("n_returned"),
        "states": d.get("terminal_states") or {},
        "res": res,
    }


def ref_main(root: str) -> Dict[str, Any]:
    d = _load(os.path.join(root, "refspatial", "summary.json")) or {}
    v = (d.get("variants") or {}).get(REF_MAIN_VARIANT) or {}
    splits = v.get("splits") or {}
    out = {"default_scope": d.get("default_scope_splits") or [],
           "overlapping": d.get("overlapping_subset_splits") or [],
           "splits": {}}
    for name, si in splits.items():
        cli = ((si.get("scores") or {}).get("official_cli") or {})
        out["splits"][name] = {
            "n": si.get("n"),
            "acc": cli.get("official_accuracy"),
            "evaluated": cli.get("official_evaluated"),
            "total": cli.get("official_total"),
            "states": si.get("terminal_states") or {},
        }
    return out


def ts_main(root: str) -> Dict[str, Any]:
    d = _load(os.path.join(root, "tracespatial", "summary.json")) or {}
    v = (d.get("variants") or {}).get(TS_MAIN_VARIANT) or {}
    out = {"tasks": {}}
    for task, ti in (v.get("tasks") or {}).items():
        sc = ti.get("scores") or {}
        aud = sc.get("scoring_audit") or {}
        out["tasks"][task] = {
            "n": ti.get("n"),
            "scores": sc,
            "audit": aud,
            "scored": aud.get("n_evaluated_total", ti.get("n")),
            "excluded": sc.get("n_excluded_samples", 0),
            "states": ti.get("terminal_states") or {},
            "metric2d_samples": aud.get("n_metric2d_samples"),
        }
    return out


def stability(root: str) -> Dict[str, Any]:
    return _load(os.path.join(root, "stability", "stability.json")) or {}


def rescore(root: str) -> Dict[str, Any]:
    return _load(os.path.join(root, "rescore_check.json")) or {}


# =============================================================================
# 表 A：主指标一览（主口径）
# =============================================================================
def table_a(root: str) -> List[str]:
    L: List[str] = []
    cv, ref, ts = cv_main(root), ref_main(root), ts_main(root)

    L.append("| 基准 | 主指标 | 方向 | 分母 / 口径 | 论文参考 | 本次主口径 | 差值 |")
    L.append("|---|---|---|---|---|---|---|")

    # ---- CV-Bench ----
    res = cv["res"]
    n_cv = cv["n_planned"]
    cv_rows = [
        ("**综合分 CV_score**", "accuracy", True),
        ("Acc(ADE20K)", "accuracy_2d_ade", False),
        ("Acc(COCO)", "accuracy_2d_coco", False),
        ("Acc(Omni3D 3D)", "accuracy_3d_omni", False),
        ("Acc_2D", "accuracy_2d", False),
        ("Acc_3D", "accuracy_3d", False),
    ]
    for label, key, is_main in cv_rows:
        v = res.get(key)
        if v is None:
            continue
        val = BST._num(v, 2)
        if is_main:
            ref_s = "%s" % BST._num(PAPER_CV, 2)
            diff = "**%s pt**" % BST._delta(v, PAPER_CV)
            val = "**%s**" % val
        else:
            ref_s = "论文只给综合分 %s" % BST._num(PAPER_CV, 2)
            diff = "—"
        L.append("| CV-Bench | %s | ↑ | %s / `%s` | %s | %s | %s |"
                 % (label, n_cv, CV_MAIN, ref_s, val, diff))

    # ---- RefSpatial ----
    for sp in ref["default_scope"]:
        s = ref["splits"].get(sp) or {}
        if s.get("acc") is None:
            continue
        L.append("| RefSpatial-Bench | %s | ↑ | %s / `%s` | 论文未公开子集切分 | **%s** | — |"
                 % (sp, s.get("n"), REF_MAIN_VARIANT, BST._num(s["acc"] * 100, 2)))
    # 默认范围均值（论文 60.50 是聚合口径，只能与聚合值对照）
    scope = [ref["splits"][x]["acc"] for x in ref["default_scope"]
             if (ref["splits"].get(x) or {}).get("acc") is not None]
    if scope:
        mean = sum(scope) / len(scope)
        L.append("| RefSpatial-Bench | **默认范围均值**（%s） | ↑ | mean(%s) / `%s` | %s（聚合口径未公开） | **%s** | **%s pt** |"
                 % ("+".join(ref["default_scope"]),
                    ", ".join(ref["default_scope"]), REF_MAIN_VARIANT,
                    BST._num(PAPER_REFSPATIAL, 2),
                    BST._num(mean * 100, 2),
                    BST._delta(mean * 100, PAPER_REFSPATIAL)))
    for sp in ref["overlapping"]:
        s = ref["splits"].get(sp) or {}
        if s.get("acc") is None:
            continue
        L.append("| RefSpatial-Bench | %s（重叠子集，单列，不计入默认均值） | ↑ | %s / `%s` | — | %s | — |"
                 % (sp, s.get("n"), REF_MAIN_VARIANT, BST._num(s["acc"] * 100, 2)))

    # ---- TraceSpatial ----
    t3 = ts["tasks"].get("3D") or {}
    if t3:
        sc = t3["scores"]
        den3 = "%s/%s" % (t3["scored"], t3["n"])
        for label, key, paper_key in (("**3D Start Success**", "start_success_rate", "3D Start"),
                                      ("**3D End Success**", "end_success_rate", "3D End"),
                                      ("**Overall Success**", "overall_success_rate", "Success")):
            v = sc.get(key)
            if v is None:
                continue
            L.append("| TraceSpatial-Bench | %s | ↑ | %s / `%s` | %s | **%s** | **%s pt** |"
                     % (label, den3, TS_MAIN_VARIANT, PAPER_TS[paper_key],
                        BST._num(v * 100, 2), BST._delta(v * 100, PAPER_TS[paper_key])))
        nc = sc.get("no_collision_rate")
        if nc is not None:
            L.append("| TraceSpatial-Bench | No-Collision Rate | ↑ | %s / `%s` | — | %s | — |"
                     % (den3, TS_MAIN_VARIANT, BST._num(nc * 100, 2)))
        for label, key in (("3D Start 距离 (m)", "start_distance"),
                           ("3D End 距离 (m)", "end_distance")):
            v = sc.get(key)
            if v is None:
                continue
            L.append("| TraceSpatial-Bench | %s | ↓ | %s / `%s` | — | %s | — |"
                     % (label, den3, TS_MAIN_VARIANT, BST._num(v, 4)))

    t2 = ts["tasks"].get("2D") or {}
    if t2:
        sc = t2["scores"]
        den2 = "%s/%s" % (t2["scored"], t2["n"])
        for label, key in (("2D Start-in-Mask", "start_in_mask"),
                           ("2D End-in-BBox", "end_in_bbox")):
            v = sc.get(key)
            if v is None:
                continue
            L.append("| TraceSpatial-Bench | %s | ↑ | %s / `%s` | — | %s | — |"
                     % (label, den2, TS_MAIN_VARIANT, BST._num(v * 100, 2)))
        m2 = sc.get("metrics_2d") or {}
        dfd = m2.get("average_discrete_frechet_distance")
        if dfd is not None:
            den_dfd = "%s/%s" % (t2.get("metric2d_samples") or t2["scored"], t2["n"])
            L.append("| TraceSpatial-Bench | 2D DFD | ↓ | %s（计入路径均值） / `%s` | — | %s | — |"
                     % (den_dfd, TS_MAIN_VARIANT, BST._num(dfd, 4)))
    return L


# =============================================================================
# 表 B：口径索引
# =============================================================================
TABLE_B = [
    "| 小节 / 表 | 内容 | 口径性质 | 能否作为上报数字 |",
    "|---|---|---|---|",
    "| §6.2.0 表 A | 三个公开基准的**主口径主指标** | 主口径（与上游逐字一致） | ✅ 可以 |",
    "| §6.2.0 表 D | 全部指标 × 全部次口径 | 主口径 + 次口径并列 | 仅「主口径」行可以 |",
    "| §6.2.0 表 C | 同配置重复推理、离线重判分 | 稳定性 / 可复现性 | ✅ 可以（作为证据） |",
    "| §6.2.2 | `Count` 缺口成因（误差分布 + 目视核对） | 归因 | ❌ 仅归因 |",
    "| §6.2.3 | 上采样（`Count` 788 题 / 全量 2638 题） | **在评测集上按分数选出的协议** | ❌ 禁止（验收要求 §4.2） |",
    "| §6.2.4 | 提示词口径、运行间噪声底、评测耗时台账 | 归因 + 台账 | ❌ 仅归因 |",
]


# =============================================================================
# 表 C：稳定性与可复现性
# =============================================================================
def table_c(root: str) -> List[str]:
    st, rs = stability(root), rescore(root)
    L = ["| 检查 | 结果 |", "|---|---|"]
    for key in ("cvbench", "refspatial", "tracespatial"):
        d = st.get(key)
        if not d:
            continue
        L.append("| %s 固定样例 × 2 次重复推理 | 原始输出一致 %s/%s，变化数 %s，变化率 %s，终态全 OK %s |"
                 % (d.get("dataset"), d.get("identical_raw"), d.get("n"),
                    d.get("changed_raw"), d.get("change_ratio"),
                    "✅" if d.get("all_states_ok") else "❌"))
    s = (rs.get("summary") or {})
    if s:
        L.append("| 离线重判分复现（`rescore.py`） | 比较指标数 **%s**，不一致 **%s**，可复现 **%s** |"
                 % (s.get("compared_metrics"), s.get("mismatched"),
                    "✅" if s.get("reproducible") else "❌"))
    return L


# =============================================================================
# 表 D：全量指标明细（复用 build_summary_table 的行生成，保证与汇总表一致）
# =============================================================================
def table_d(root: str) -> List[str]:
    head = ["基准/指标", "计划题数", "有效返回数", "实际计分数", "请求/解析失败数",
            "参考值", "本次值", "差值/单位", "允许下降", "判定"]
    rows = (BST.rows_refspatial(root) + BST.rows_tracespatial(root)
            + BST.rows_cvbench(root))
    L = ["| " + " | ".join(head) + " |",
         "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        L.append("| " + " | ".join(r) + " |")
    return L


# =============================================================================
# §6.2.3 上采样表（Count 788 题）
# =============================================================================
def table_upscale(root: str) -> List[str]:
    d = _load(os.path.join(root, "cvbench_upscale", "upscale_compare.json"))
    if not d:
        return []
    base = None
    for it in d:
        if it.get("protocol") == CV_MAIN:
            base = it
    if base is None:
        return []
    b = base.get("official") or {}
    L = ["结果（`Count` 788 题，同一会话，唯一变量 = 缩放策略；判分 = 官方 FlagEvalMM `get_result`）：",
         "",
         "| 缩放策略 | Acc(ADE20K) | Acc(COCO) | Acc_2D | Δ vs canonical |",
         "|---|---|---|---|---|"]
    for it in d:
        p = it.get("protocol")
        o = it.get("official") or {}
        if p == CV_MAIN:
            L.append("| `%s`（同会话对照） | %s | %s | %s | — |"
                     % (p, BST._num(o.get("accuracy_2d_ade"), 2),
                        BST._num(o.get("accuracy_2d_coco"), 2),
                        BST._num(o.get("accuracy_2d"), 2)))
        else:
            d2 = o.get("accuracy_2d", 0) - b.get("accuracy_2d", 0)
            L.append("| **`%s`** | **%s** | **%s** | **%s** | **%s** |"
                     % (p, BST._num(o.get("accuracy_2d_ade"), 2),
                        BST._num(o.get("accuracy_2d_coco"), 2),
                        BST._num(o.get("accuracy_2d"), 2),
                        BST._delta(o.get("accuracy_2d", 0), b.get("accuracy_2d", 0))))
    return L


# =============================================================================
# §6.2.3 上采样全量表（2638 题）
# =============================================================================
def table_upscale_full(root: str) -> List[str]:
    up = _load(os.path.join(root, "cvbench_up2_full_result.json"))
    cv = cv_main(root)
    if not up:
        return []
    res = cv["res"]
    L = ["| 指标 | 主口径 `%s` | `up2_png`（诊断口径） | Δ |" % CV_MAIN,
         "|---|---|---|---|"]
    rows = [("Acc(ADE20K)", "accuracy_2d_ade", False),
            ("Acc(COCO)", "accuracy_2d_coco", False),
            ("Acc_2D", "accuracy_2d", False),
            ("Acc(Omni3D 3D)", "accuracy_3d_omni", False),
            ("**CV_score（官方综合分）**", "accuracy", True)]
    for label, key, is_main in rows:
        a = res.get(key)
        b = up.get(key)
        if a is None or b is None:
            continue
        delta = "**%s**" % BST._delta(b, a)
        if is_main:
            L.append("| %s | **%s** | **%s** | %s |"
                     % (label, BST._num(a, 2), BST._num(b, 2), delta))
        else:
            L.append("| %s | %s | %s | %s |"
                     % (label, BST._num(a, 2), BST._num(b, 2), delta))
    return L


# =============================================================================
# 组装与注入
# =============================================================================
def build_public_block(root: str) -> str:
    body = [
        "#### 6.2.0 精度总表（主口径 · 对外上报口径）",
        "",
        "> 本节所有数字由 `tools/inject_accuracy_table.py` 从**官方判分器产出的原始 JSON**",
        "> 现场生成（`outputs/bench_public/{cvbench,refspatial,tracespatial}/summary.json`），",
        "> **不在本文手写**；改口径请改脚本后重跑，不要直接编辑本块。",
        "",
        "**表 A：主指标一览（主口径，一屏看完）**",
        "",
    ]
    body += table_a(root)
    body += ["", "**表 B：本节各表的口径性质 —— 哪些数字可以上报**", ""]
    body += TABLE_B
    body += ["", "**表 C：稳定性与可复现性**", ""]
    body += table_c(root)
    body += ["", "**表 D：全量指标明细**（官方判分器逐字产出，含全部次口径；",
              "同一张表也随包提交为 `bench_public/summary_table.md`）", ""]
    body += table_d(root)
    body += [
        "",
        "> 表 D 的「实际计分数」是**官方判分器真正计分的分母**：",
        "> TraceSpatial 3D 子集为 **96/100**（官方脚本排除 4 条深景病态样本，见 §6.2.1）、",
        "> 2D 子集 2D DFD 计入路径均值的样本数为 **97/100**；",
        "> RefSpatial `Unseen` 77 条是**重叠子集**，不计入默认范围均值。",
        "",
    ]
    return "\n".join(body)


def replace_block(text: str, begin: str, end: str, legacy: str, body: str) -> str:
    if begin in text and end in text:
        pat = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.S)
        return pat.sub(lambda _m: begin + "\n\n" + body + "\n\n" + end, text, count=1)
    if legacy in text:
        return text.replace(legacy, begin + "\n\n" + body + "\n\n" + end, 1)
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root", default=os.path.join(ROOT, "outputs/bench_public"))
    ap.add_argument("--readme", default=os.path.join(ROOT, "submission/README.md"))
    args = ap.parse_args()
    root = args.bench_root

    if not os.path.isdir(root):
        print("[accuracy] bench root not found: %s" % root)
        return 2

    t = open(args.readme, encoding="utf-8").read()
    orig = t

    blocks = [
        ("public", build_public_block(root)),
        ("upscale", "\n".join(table_upscale(root))),
        ("upscale_full", "\n".join(table_upscale_full(root))),
    ]
    for key, body in blocks:
        if not body.strip():
            print("[accuracy] skip %s (no source data)" % key)
            continue
        begin, end, legacy = BLOCKS[key]
        before = t
        t = replace_block(t, begin, end, legacy, body)
        if t == before:
            print("[accuracy] WARN: marker not found for %s (%s)" % (key, legacy))
        else:
            print("[accuracy] %s injected (%d chars)" % (key, len(body)))

    if t != orig:
        with open(args.readme, "w", encoding="utf-8") as fh:
            fh.write(t)
        print("[accuracy] %s updated" % args.readme)
    else:
        print("[accuracy] %s unchanged" % args.readme)
    return 0


if __name__ == "__main__":
    sys.exit(main())
