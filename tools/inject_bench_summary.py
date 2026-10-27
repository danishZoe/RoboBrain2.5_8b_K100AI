#!/usr/bin/env python3
"""Inject bench/summary.md into README.md at the <!-- BENCH_SUMMARY --> marker.

Usage: python3 tools/inject_bench_summary.py [README_PATH] [SUMMARY_PATH]

Idempotent: if the marker is already replaced, the previously injected block
(between the marker comments) is replaced instead of appending a second copy.
"""
import os
import re
import sys

BEGIN = "<!-- BENCH_SUMMARY -->"
BEGIN_OPEN = "<!-- BENCH_SUMMARY:BEGIN -->"
BEGIN_CLOSE = "<!-- BENCH_SUMMARY:END -->"

readme = sys.argv[1] if len(sys.argv) > 1 else "README.md"
summary = sys.argv[2] if len(sys.argv) > 2 else "bench/summary.md"

if not os.path.exists(readme):
    sys.exit("README not found: %s" % readme)
if not os.path.exists(summary):
    sys.exit("summary not found: %s" % summary)

text = open(readme, encoding="utf-8").read()
table = open(summary, encoding="utf-8").read().strip()

block = "%s\n%s\n%s" % (BEGIN_OPEN, table, BEGIN_CLOSE)

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

open(readme, "w", encoding="utf-8").write(text)
print("[inject] %s %s <- %s (%d B)" % (mode, readme, summary, len(table)))
