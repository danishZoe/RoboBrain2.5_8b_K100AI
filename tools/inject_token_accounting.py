#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inject_token_accounting.py - render the §3.8 section of reports/10 from the
measured output of scripts/verify_output_token_accounting.py.

The section is generated from the result JSON (not hand-written) so the numbers
in the report always match the run that produced them.

Idempotent: re-running replaces the marked block.

Usage: python3 tools/inject_token_accounting.py [RESULT_JSON] [REPORT]
"""
import json
import os
import re
import sys

RESULT = sys.argv[1] if len(sys.argv) > 1 else "outputs/phase5_token_accounting/result.json"
REPORT = sys.argv[2] if len(sys.argv) > 2 else "reports/10_phase5_bench.md"

BEGIN = "<!-- SECTION:3.8_TOKEN_ACCOUNTING:BEGIN -->"
END = "<!-- SECTION:3.8_TOKEN_ACCOUNTING:END -->"


def build(res):
    rows = res.get("results", [])
    lines = []
    lines.append(BEGIN)
    lines.append("")
    lines.append("### 3.8 对照实验：随机提示词 vs 自然提示词的输出 token 计量")
    lines.append("")
    lines.append("**目的**：把 §3.7 里「`total_output_tokens_retokenized` 偏小是文本压缩而非生成变短」")
    lines.append("这一判断从**推断**升级为**实测**。做法是同一服务、同一参数，只改提示词形态：")
    lines.append("")
    lines.append("- **A 组（随机 token）**：用随机 token id 解码成文本，等价于官方客户端 random 数据集的构造；")
    lines.append("- **B 组（自然散文）**：等长（%d token）的普通英文散文。"
                 % (next((r.get("prompt_text_tokens", 0) for r in rows
                          if r.get("family") == "natural_prose"), 0)))
    lines.append("")
    lines.append("两组都用 `ignore_eos=True`、`max_tokens=%d`，因此输出长度**受控一致**；" % res.get("max_tokens", 1024))
    lines.append("差异只可能来自「文本本身被压缩的程度」。")
    lines.append("")
    lines.append("| 组别 | 提示词文本 token | 引擎生成 token（usage） | 文本重新分词长度 | 比值 | 40-gram 重复率 | 文本字符数 | 耗时 s |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append("| %s | %d | %s | %d | %s | %.1f%% | %d | %.1f |" % (
            r["family"], r["prompt_text_tokens"], r["engine_completion_tokens"],
            r["retokenized_len"],
            ("%.3f" % r["ratio"]) if r.get("ratio") is not None else "-",
            100.0 * r.get("repeat_ratio_40gram", 0.0),
            r["text_len_chars"], r["wall_s"]))
    lines.append("")

    a = next((r for r in rows if r["family"] == "gibberish_random_tokens"), None)
    b = next((r for r in rows if r["family"] == "natural_prose"), None)
    if a and b:
        ra = a.get("ratio")
        rb = b.get("ratio")
        lines.append("**结论**：")
        lines.append("")
        lines.append("- A 组（随机提示词）：引擎生成 %s token，但同一段文本重新分词只要 %d token（比值 %s），"
                     % (a["engine_completion_tokens"], a["retokenized_len"],
                        ("%.3f" % ra) if ra is not None else "-"))
        lines.append("  40-gram 重复率 **%.1f%%** —— 模型面对随机提示词时输出了高度重复的内容，"
                     % (100.0 * a.get("repeat_ratio_40gram", 0.0)))
        lines.append("  重复文本被 BPE 压成更少的 token，因此 `total_output_tokens_retokenized` 偏小。")
        lines.append("- B 组（自然提示词）：引擎生成 %s token，重新分词 %d token（比值 %s），重复率 %.1f%% —— "
                     % (b["engine_completion_tokens"], b["retokenized_len"],
                        ("%.3f" % rb) if rb is not None else "-",
                        100.0 * b.get("repeat_ratio_40gram", 0.0)))
        lines.append("  **同样存在压缩**（比值 < 1），只是重复率为 0，压缩幅度明显更小。")
        if ra is not None and rb is not None:
            lines.append("- **A 组比 B 组多压缩 %.1f 个百分点**（A 组压缩 %.1f%%，B 组压缩 %.1f%%）。"
                         % ((rb - ra) * 100.0, (1 - ra) * 100.0, (1 - rb) * 100.0))
            lines.append("  这个差值**只能解释官方压测中重分词偏差的一小部分**"
                         "（官方 16384/并发 64 档偏差约 22.6%）："
                         "本次对照是**单请求、1024 token 输出**，而官方那一档是 128 个请求、"
                         "运行 2 小时、每请求的退化程度远高于本对照的平均水平。")
        lines.append("- **因此要说清楚**：本对照**证明的是方向**（随机 token 提示词 → 输出退化 → 重分词更短，"
                     "**与提前 EOS / 生成变短无关**），**不是官方那一档 22.6% 偏差的完整定量归因**。")
        lines.append("  决定性的证据仍然是 §3.7 的**服务端引擎计数器交叉核算**"
                     "（引擎生成量 = 计划量的 95.6%–99.7%，决定性一档 99.7%）；"
                     "官方 `output_throughput` 列（引擎生成速率）**可直接填报**。")
        lines.append("")
        lines.append("> 对「16384 输入档的重分词偏差（12%–23%）比 4096 档（≤7%）大」的解释"
                     "（上下文越长、随机 token 越密集，模型越容易滑入重复输出）"
                     "目前是**与数据一致的推断**，不是本对照直接证明的结论 —— 本对照未覆盖不同输入长度。")
    lines.append("")
    lines.append("**证据**：`outputs/phase5_token_accounting/result.json`（含完整生成文本的头尾片段）。")
    lines.append("")
    lines.append(END)
    return "\n".join(lines) + "\n"


def main():
    if not os.path.exists(RESULT):
        print("[inject] missing %s -> skip" % RESULT)
        return 2
    with open(RESULT, encoding="utf-8") as f:
        res = json.load(f)
    if not os.path.exists(REPORT):
        print("[inject] missing %s" % REPORT)
        return 2
    with open(REPORT, encoding="utf-8") as f:
        text = f.read()
    block = build(res)
    if BEGIN in text and END in text:
        text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END),
                      block.replace("\\", "\\\\"), text, flags=re.S)
        mode = "replaced"
    else:
        anchor = "\n---\n\n## 4. 复现命令"
        if anchor not in text:
            print("[inject] anchor not found in %s" % REPORT)
            return 2
        text = text.replace(anchor, "\n" + block + anchor, 1)
        mode = "inserted"
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(text)
    print("[inject] %s %s (%d B)" % (mode, REPORT, len(block)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
