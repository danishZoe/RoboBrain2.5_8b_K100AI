#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inject_clean_repro.py - render the §7.2 clean-reproduction result into
reports/05_final.md from logs/clean_repro/*.

The section is generated from the reproduction run's own artifacts
(performance jsonl + verify_result.json), so the numbers cannot drift from
the run they describe.

Idempotent: re-running replaces the marked block.

Usage: python3 tools/inject_clean_repro.py [REPORT] [REPRO_DIR]
"""
import json
import os
from pathlib import Path
import re
import sys

REPORT = sys.argv[1] if len(sys.argv) > 1 else "reports/05_final.md"
REPRO = sys.argv[2] if len(sys.argv) > 2 else "logs/clean_repro"
BEGIN = "<!-- CLEAN_REPRO:BEGIN -->"
END = "<!-- CLEAN_REPRO:END -->"



_BASE = Path(__file__).resolve().parent.parent


def _resolve(rel):
    """解析相对路径到实际存在的位置（基准 = 脚本自身所在根）。

    优先字面路径 <base>/rel（节点工作目录布局：submission/...），
    再回落去掉 ``submission/`` 前缀（仓库提交视图布局：提交包已上提到根）。
    """
    rel = str(rel)
    p = _BASE / rel
    if p.exists():
        return p
    if rel.startswith("submission/"):
        alt = _BASE / rel[len("submission/"):]
        if alt.exists():
            return alt
    return p
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


def build():
    repro = load_jsonl(os.path.join(REPRO, "repro_4096_c1.jsonl"))
    base = [r for r in load_jsonl(str(_resolve("submission/bench/round2.jsonl")))
            if r.get("random_input_len") == 4096 and r.get("max_concurrency") == 1]
    vp = os.path.join(REPRO, "verify_result.json")
    vstatus, vreason = "-", ""
    if os.path.exists(vp):
        with open(vp, encoding="utf-8") as f:
            d = json.load(f)
        vstatus = d.get("status", "-")
        vreason = (d.get("reason") or "").strip()
    has_smoke = os.path.exists(os.path.join(REPRO, "smoke.log"))
    has_env = os.path.exists(os.path.join(REPRO, "env.log"))

    lines = [BEGIN, ""]
    lines.append("### 4.5 干净环境复现（验收标准 §7.2）")
    lines.append("")
    if not repro or not base:
        lines.append("尚未执行（`logs/clean_repro/` 无复跑数据）。")
        lines.append("")
        lines.append(END)
        return "\n".join(lines) + "\n"
    b, c = base[0], repro[0]
    dev = (c["output_throughput"] - b["output_throughput"]) / b["output_throughput"] * 100.0
    ttft_dev = ((c["mean_ttft_ms"] - b["mean_ttft_ms"]) / b["mean_ttft_ms"] * 100.0)
    lines.append("复现方式：用提交包自带的 `run.sh` **全新拉起**服务，再依次执行")
    lines.append("`env_check.sh` → `smoke.sh` → `verify.py` → 官方压测客户端（4096/1024/并发 1，8 请求，同 seed），")
    lines.append("全程日志落 `logs/clean_repro/`。")
    lines.append("")
    lines.append("| 项 | 提交表（记录轮 R2） | 干净复跑 | 相对偏差 |")
    lines.append("|---|---|---|---|")
    lines.append("| 输出吞吐 tok/s | %.2f | %.2f | **%+.1f%%** |" % (
        b["output_throughput"], c["output_throughput"], dev))
    lines.append("| 完成 / 失败请求 | %d / %d | %d / %d | - |" % (
        b["completed"], b.get("failed", 0), c["completed"], c.get("failed", 0)))
    lines.append("| 平均 TTFT ms | %.2f | %.2f | %+.1f%% |" % (
        b["mean_ttft_ms"], c["mean_ttft_ms"], ttft_dev))
    lines.append("| 平均 TPOT ms | %.2f | %.2f | %+.1f%% |" % (
        b["mean_tpot_ms"], c["mean_tpot_ms"],
        (c["mean_tpot_ms"] - b["mean_tpot_ms"]) / b["mean_tpot_ms"] * 100.0))
    lines.append("")
    lines.append("| 复现项 | 结果 |")
    lines.append("|---|---|")
    lines.append("| 环境采集 `env_check.sh` | %s |" % ("已生成 `env.log`" if has_env else "缺失"))
    lines.append("| Smoke `smoke.sh` | %s |" % ("已生成 `smoke.log`" if has_smoke else "缺失"))
    lines.append("| 精度 `verify.py` | **%s**%s |" % (
        vstatus, ("（%s）" % vreason[:60]) if vreason else ""))
    lines.append("")
    if abs(dev) <= 20.0:
        lines.append("**判定：性能复现偏差 %+.1f%%，在官方 20%% 打回线以内；复现通过。**" % dev)
    else:
        lines.append("**判定：性能复现偏差 %+.1f%%，超过官方 20%% 打回线 —— 需在提交材料中写明原因。**" % dev)
    lines.append("")
    lines.append(END)
    return "\n".join(lines) + "\n"


def main():
    if not os.path.exists(REPORT):
        print("[inject] missing %s" % REPORT)
        return 2
    with open(REPORT, encoding="utf-8") as f:
        text = f.read()
    block = build()
    if BEGIN in text and END in text:
        text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END),
                      block.replace("\\", "\\\\"), text, flags=re.S)
        mode = "replaced"
    else:
        anchor = "\n---\n\n## 5. 兼容性改动清单"
        if anchor not in text:
            print("[inject] anchor not found in %s" % REPORT)
            return 2
        text = text.replace(anchor, "\n" + block + anchor, 1)
        mode = "inserted"
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(text)
    print("[inject] %s %s (clean repro)" % (mode, REPORT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
