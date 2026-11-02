#!/usr/bin/env python3
"""Inject bench/summary.md into README.md at the <!-- BENCH_SUMMARY --> marker.

Usage: python3 tools/inject_bench_summary.py [README_PATH] [SUMMARY_PATH]

Idempotent: if the marker is already replaced, the previously injected block
(between the marker comments) is replaced instead of appending a second copy.
"""
import os
import re
import sys
from pathlib import Path

BEGIN = "<!-- BENCH_SUMMARY -->"
BEGIN_OPEN = "<!-- BENCH_SUMMARY:BEGIN -->"
BEGIN_CLOSE = "<!-- BENCH_SUMMARY:END -->"

_BASE = Path(__file__).resolve().parent.parent


def _resolve(rel):
    """解析相对路径到实际存在的位置。

    优先字面路径 <base>/rel（节点工作目录布局：submission/...），
    再回落去掉 ``submission/`` 前缀（仓库提交视图布局：提交包已上提到根）。
    保证节点侧行为不变，克隆后的仓库也能跑。
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


readme = sys.argv[1] if len(sys.argv) > 1 else str(_resolve("submission/README.md"))
summary = sys.argv[2] if len(sys.argv) > 2 else str(_resolve("bench/summary.md"))

if not os.path.exists(readme):
    sys.exit("README not found: %s" % readme)
if not os.path.exists(summary):
    sys.exit("summary not found: %s" % summary)

text = open(readme, encoding="utf-8").read()
table = open(summary, encoding="utf-8").read().strip()

block = "%s\n\n%s\n\n%s" % (BEGIN_OPEN, table, BEGIN_CLOSE)

if BEGIN_CLOSE in text:
    # replace the previously injected block
    text = re.sub(re.escape(BEGIN_OPEN) + r".*?" + re.escape(BEGIN_CLOSE),
                  block.replace("\\", "\\\\"), text, flags=re.S)
    mode = "replaced"
elif BEGIN in text:
    text = text.replace(BEGIN, block)
    mode = "filled"
else:
    sys.exit("marker %s not found in %s" % (BEGIN, readme))

with open(readme, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)
print("[inject] %s %s <- %s (%d B)" % (mode, readme, summary, len(table)))
