#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_parse_coverage.py - 官方解析正则的覆盖边界单独检查

验收标准 §4.3 的原文要求：

  "不能仅靠上游绘图正则是否匹配来判模型正确性；其对整数深度、科学计数法等格式的
   覆盖需要另行检查。保留原文、解析值、解析失败原因与可视化，解析失败不能剔除出
   官方计分分母。"

本脚本做三件事：
  1) 断言本文件里的三条正则与 `scripts/robo_infer.py` 里的**逐字一致**
     （防止日后有人改了实现却没改口径）；
  2) 用真实录到的模型输出 + 构造的边界样例，跑一遍正则，把
     「原文 → 解析值 / 解析失败原因」逐条列出来；
  3) 输出 JSON + Markdown，供报告直接引用。

**不改官方口径**：这里发现的覆盖缺口（整数深度、科学计数法、浮点坐标等）
在官方评测里同样存在，所以本脚本只做**记录**，不改 `robo_infer.py` 的正则；
真正的计分以官方判分器为准。

用法:
    python3 scripts/test_parse_coverage.py --out-dir outputs/phase4_parse_coverage
"""
import argparse
import json
import os
import re
import sys

# ---- 与 scripts/robo_infer.py 逐字一致的三条官方正则 --------------------------
RE_TRAJ = r"(\d+),\s*(\d+),\s*([+-]?\d+\.\d+)"
RE_POINT = r"\(\s*(\d+)\s*,\s*(\d+)\s*\)"
RE_BBOX = r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"

# ---- 用例 ---------------------------------------------------------------------
CASES = [
    # (类别, 文本, 说明)
    ("trajectory", "[(294, 525, 0.577), (405, 350, 0.607), (525, 300, 0.720)]",
     "本节点真实输出（官方样例）"),
    ("trajectory", "[(294, 525, 0.577)]", "单点轨迹"),
    ("trajectory", "[(294,525,0.577)]", "无空格"),
    ("trajectory", "[(294, 525, -0.577)]", "负深度"),
    ("trajectory", "[(294, 525, .577)]", "深度省略前导 0（官方正则要求 \\d+ 前导数字）"),
    ("trajectory", "[(294, 525, 1)]", "整数深度（官方正则要求小数点）"),
    ("trajectory", "[(294, 525, 1e-3)]", "科学计数法深度"),
    ("trajectory", "[(294.5, 525, 0.577)]", "浮点坐标"),
    ("trajectory", "[(294, 525, 0.577), (405, 350, 0.607)]", "两点（无截断）"),
    ("pointing", "the answer is [(901, 600)]", "本节点真实输出（官方样例）"),
    ("pointing", "[(901,600)]", "无空格"),
    ("pointing", "[(901, 600), (120, 340)]", "两点"),
    ("pointing", "[(901.5, 600)]", "浮点坐标"),
    ("pointing", "901, 600", "无括号（上游绘图正则要求括号）"),
    ("grounding", "The box is [0, 309, 620, 989].", "本节点真实输出（官方样例）"),
    ("grounding", "box <bbox>0 298 622 998</bbox>", "vLLM 分析模式的尖括号形式"),
    ("grounding", "[0,309,620,989]", "无空格"),
    ("grounding", "[0.0, 309, 620, 989]", "浮点坐标"),
    ("grounding", "[0, 309, 620]", "只有三个数"),
    ("general", "A white mug on a wooden table next to a laptop.",
     "自由文本（general 不做结构化解析）"),
]


def parse(kind, text):
    """按官方正则解析，返回 (解析值, 失败原因)"""
    if kind == "trajectory":
        pts = re.findall(RE_TRAJ, text)
        if not pts:
            return None, "官方正则未匹配：轨迹要求 (x, y, d) 且 d 为带小数点的数（%s）" % RE_TRAJ
        return [[int(x), int(y), float(d)] for x, y, d in pts], None
    if kind == "pointing":
        pts = re.findall(RE_POINT, text)
        if not pts:
            return None, "官方正则未匹配：指点要求 (x, y)（%s）" % RE_POINT
        return [{"x": int(x), "y": int(y)} for x, y in pts], None
    if kind == "grounding":
        boxes = re.findall(RE_BBOX, text)
        if not boxes:
            return None, "官方正则未匹配：grounding 要求 [x1, y1, x2, y2]（%s）" % RE_BBOX
        return {"bbox": [int(v) for v in boxes[0]]}, None
    return text, None


def check_regex_parity():
    """断言本脚本的正则与 robo_infer.py 逐字一致"""
    here = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(here, "robo_infer.py")
    if not os.path.exists(src_path):
        return None, "robo_infer.py 不存在，跳过一致性断言"
    src = open(src_path, encoding="utf-8").read()
    out = {}
    for name, pat in (("trajectory", RE_TRAJ), ("pointing", RE_POINT), ("grounding", RE_BBOX)):
        out[name] = (repr(pat) in src) or ('"%s"' % pat in src) or ("'%s'" % pat in src)
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/phase4_parse_coverage")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    parity, perr = check_regex_parity()
    results = []
    for kind, text, note in CASES:
        val, err = parse(kind, text)
        results.append(dict(category=kind, text=text, note=note,
                            parsed=val, parse_error=err,
                            parseable=err is None))

    n_ok = sum(1 for r in results if r["parseable"])
    n_gap = len(results) - n_ok

    payload = dict(
        schema="robo.parse_coverage.v1",
        regexes=dict(trajectory=RE_TRAJ, pointing=RE_POINT, grounding=RE_BBOX),
        regex_parity_with_robo_infer=parity,
        total=len(results), parseable=n_ok, coverage_gaps=n_gap,
        results=results,
        note=("这里只做**覆盖边界记录**，不修改官方正则与判分口径；"
              "官方评测以官方判分器为准。"),
    )
    with open(os.path.join(args.out_dir, "parse_coverage.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = ["## 官方解析正则的覆盖边界（验收标准 §4.3）", ""]
    lines.append("正则与 `scripts/robo_infer.py` 逐字一致：**%s**" % (
        "是" if parity and all(parity.values()) else (str(parity) if parity else "未校验")))
    lines.append("")
    lines.append("| 类别 | 原文 | 解析值 | 解析失败原因 | 说明 |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        lines.append("| %s | `%s` | %s | %s | %s |" % (
            r["category"], r["text"].replace("|", "\\|"),
            ("`%s`" % json.dumps(r["parsed"], ensure_ascii=False)) if r["parseable"] else "—",
            r["parse_error"] or "—", r["note"]))
    lines.append("")
    lines.append("**结论**：%d/%d 条可被官方正则解析；%d 条暴露覆盖缺口。" % (
        n_ok, len(results), n_gap))
    lines.append("")
    lines.append("已知的官方正则覆盖缺口（**同样存在于官方评测中，本节点不改口径**）：")
    lines.append("")
    lines.append("- 轨迹深度写成**整数**（如 `[(1, 2, 3)]`）或**科学计数法**（`1e-3`）→ 不匹配；")
    lines.append("- 坐标写成**浮点**（如 `901.5`）→ 不匹配；")
    lines.append("- pointing 缺括号 → 不匹配；grounding 缺一个坐标 → 不匹配。")
    lines.append("")
    lines.append("处理原则：解析失败**保留原文与失败原因、计入官方计分分母、不得剔除**；"
                 "真实输出样例见 `outputs/tasks_official/task_results.json`。")
    lines.append("")
    with open(os.path.join(args.out_dir, "parse_coverage.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("[parse-coverage] %d/%d parseable, %d gaps -> %s" % (
        n_ok, len(results), n_gap, args.out_dir))
    for r in results:
        if not r["parseable"]:
            print("   GAP  %-11s %s" % (r["category"], r["text"][:60]))
    if parity is not None and not all(parity.values()):
        print("   WARNING: regex parity with robo_infer.py FAILED: %s" % parity)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
