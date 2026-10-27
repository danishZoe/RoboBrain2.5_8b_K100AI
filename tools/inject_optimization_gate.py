#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inject_optimization_gate.py - 在 reports/05_final.md 写入 §4.6
「验收标准 §8.2 优化门槛（≥10%）核算」。

§8.2 要求：同卡型、同框架、同负载下性能提升 ≥10%，且**优化后必须重跑官方精度门槛**，
精度未通过的性能结果不算优化验收成功。本节把这两条逐项对照实测结果列出，
数字全部由原始文件派生：

  基准   = submission/bench/round2.jsonl（首次提交性能实测，记录轮）
  优化项 = outputs/phase5_ab_noneager/*   （E1 非 eager）
           outputs/phase5_ab_bnb/*        （E2 max-num-batched-tokens）

Idempotent: 重复执行会替换标记块。

Usage: python3 tools/inject_optimization_gate.py [REPORT]
"""
import glob
import json
import os
import re
import sys

REPORT = sys.argv[1] if len(sys.argv) > 1 else "reports/05_final.md"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

BEGIN = "<!-- OPTGATE:BEGIN -->"
END = "<!-- /OPTGATE -->"


def load_jsonl(p):
    if not os.path.exists(p):
        return []
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def dev(new, old):
    if not old:
        return None
    return (new - old) / old * 100.0


def baseline(tier):
    for r in load_jsonl("submission/bench/round2.jsonl"):
        if r.get("random_input_len") == tier[0] and r.get("max_concurrency") == tier[1]:
            return r
    return None


def e1_runs(prefix_filter=None):
    dirs = [p for p in sorted(glob.glob("outputs/phase5_ab_noneager/*"))
            if os.path.basename(p).startswith("noneager_")]
    if prefix_filter is not None:
        dirs = [p for p in dirs if prefix_filter(p)]
    if not dirs:
        return None, {}
    d = dirs[-1]
    runs = {}
    for p in sorted(glob.glob(os.path.join(d, "ab_*.jsonl"))):
        for r in load_jsonl(p):
            runs[(r.get("random_input_len"), r.get("max_concurrency"))] = r
    return d, runs


def agree_rate(d):
    if not d:
        return None
    f = os.path.join(d, "agreement.md")
    if not os.path.exists(f):
        return None
    txt = open(f, encoding="utf-8", errors="replace").read()
    m = re.search(r"(\d+)\s*/\s*(\d+)\s*=\s*([\d.]+)%", txt)
    if m:
        return int(m.group(1)), int(m.group(2)), float(m.group(3))
    return None


def build():
    L = []
    L.append(BEGIN)
    L.append("")
    L.append("### 4.6 验收标准 §8.2 优化门槛（≥10%）核算")
    L.append("")
    L.append("§8.2 有两条硬要求：**同卡型、同框架、同负载下提升 ≥10%**；"
             "**优化后必须重跑官方精度门槛，精度未通过的性能结果不算优化验收成功**。")
    L.append("下表按 §8.2 的表格模板逐格填写，数字取自原始文件。")
    L.append("")
    L.append("| 档位/指标 | 首次提交基准 | 优化结果 | 公式与变化 | 精度结果 | 原始证据 |")
    L.append("|---|---|---|---|---|---|")

    d1, runs1 = e1_runs(lambda p: not os.path.basename(p).startswith("noneager_confirm_"))
    dc, runsc = e1_runs(lambda p: os.path.basename(p).startswith("noneager_confirm_"))
    ar = agree_rate(d1)
    arc = agree_rate(dc)

    covered = False
    max_dev = None
    max_tier = None
    for tier in sorted(runs1, key=lambda t: (t[0] or 0, t[1] or 0)):
        b = baseline(tier)
        if not b:
            continue
        covered = True
        r = runs1[tier]
        btps = b.get("output_throughput", 0)
        atps = r.get("output_throughput", 0)
        d = dev(atps, btps)
        if d is not None and (max_dev is None or d > max_dev):
            max_dev, max_tier = d, tier
        verdict = "≥10%" if (d is not None and d >= 10.0) else "<10%"
        acc = ("官方精度门槛 **BLOCKED**（缺官方 GPU 基线/判分器）；"
               "内部一致性：与 transformers 基线逐坐标 %s"
               % ("%d/%d = %.1f%%" % ar if ar else "未取得"))
        L.append("| %s/%s 输出吞吐 tok/s | %.2f | %.2f | `(%.2f / %.2f - 1) × 100%%` = **%+.1f%%**（%s） | %s | `outputs/phase5_ab_noneager/%s/` + `submission/bench/round2.jsonl` |"
                 % (tier[0], tier[1], btps, atps, atps, btps, d, verdict, acc,
                    os.path.basename(d1) if d1 else "-"))

    if not covered:
        L.append("| （无 E1 实测结果） | — | — | — | — | — |")

    L.append("")
    L.append("**结论（逐条对照 §8.2）**：")
    L.append("")
    if d1:
        L.append("1. **≥10% 门槛：只在并发 1 档成立，且不构成整件级提升。**")
        L.append("   最高提升出现在 %s 档（**%+.1f%%**），但官方口径的核心档（并发 8 / 64）没有同量级收益；"
                 "§8.2 明确「不因单档提升就宣称整件已获官方认可」，因此本报告**不宣称整件达到 ≥10%%**。"
                 % ("%s/%s" % max_tier if max_tier else "-", max_dev or 0))
        L.append("2. **精度复验：无法完成 → §8.2 的第二个条件不成立。**")
        L.append("   §8.2 要求「优化后重跑服务功能及官方精度门槛，精度未通过的性能结果不算优化验收成功」。"
                 "本任务的官方精度门槛因缺官方 GPU 参考输出与判分器而 **BLOCKED**"
                 "（见 `reports/08_gap_vs_official_baseline.md` 风险 #1），"
                 "无法证明非 eager 配置下输出数值变化（内部逐坐标一致率 %s）是无害的。"
                 % (("%d/%d = %.1f%%" % ar) if ar else "未取得"))
        L.append("3. **因此采纳决策：提交配置保持首提基线（`--enforce-eager`）。**")
        L.append("   零风险、可复现、与提交表完全一致；非 eager 的可启动性与收益/代价在 "
                 "`reports/11_tuning_experiments.md` §2 如实登记，不隐藏、也不冒充已采纳的优化。")
        if dc and runsc:
            tiers_txt = "、".join("%s/%s" % t for t in sorted(runcs))
            L.append("4. **确认实验**：非 eager 重跑四档（%s），与首测结论一致，"
                     "内部逐坐标一致率 %s。"
                     % (tiers_txt, ("%d/%d = %.1f%%" % arc) if arc else "未取得"))
    else:
        L.append("E1 尚未执行，本节待回填。")
    L.append("")
    L.append("> 一句话：**本次交付没有可宣称达标的性能优化**。同口径基准已完整建立、"
             "实验方法与判定标准已实验前登记、单变量 A/B 已执行并留证；"
             "但 §8.2 的两个条件（≥10% 且精度复验通过）在本节点上只满足前者的一部分，"
             "而后者因缺官方材料无法满足 —— 如实标注，不凑数。")
    L.append("")
    L.append(END)
    return "\n".join(L) + "\n"


def inject(text, body):
    if BEGIN in text and END in text:
        return re.sub(re.escape(BEGIN) + r".*?" + re.escape(END),
                      (BEGIN + "\n" + body + END).replace("\\", "\\\\"),
                      text, flags=re.S)
    if BEGIN in text:
        return text.replace(BEGIN, BEGIN + "\n" + body + END, 1)
    return text


def main():
    if not os.path.exists(REPORT):
        print("[optgate] missing %s" % REPORT)
        return 2
    t = open(REPORT, encoding="utf-8").read()
    body = build()
    if BEGIN not in t:
        # 插到 §5 之前的分隔线处
        anchor = "\n---\n\n## 5. 兼容性改动清单"
        if anchor in t:
            t = t.replace(anchor, "\n" + body + anchor, 1)
        else:
            t = t.rstrip("\n") + "\n\n" + body
    else:
        t = inject(t, body)
    open(REPORT, "w", encoding="utf-8").write(t)
    print("[optgate] %s updated" % REPORT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
