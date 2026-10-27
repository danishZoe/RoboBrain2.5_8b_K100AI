#!/usr/bin/env python3
"""Inject the official bench tables into reports/10_phase5_bench.md.

Replaces the <!-- OFFICIAL_BENCH --> marker with:
  * the official 13-column table (bench/summary.md)
  * the detailed table (bench/summary_detailed.md)

Idempotent: if the marker was already replaced, the injected block is
replaced instead of appending a second copy.

Usage: python3 tools/inject_report10_bench.py [REPORT] [SUMMARY] [DETAILED]
"""
import os
import re
import sys

report = sys.argv[1] if len(sys.argv) > 1 else "reports/10_phase5_bench.md"
summary = sys.argv[2] if len(sys.argv) > 2 else "submission/bench/summary.md"
detailed = sys.argv[3] if len(sys.argv) > 3 else "submission/bench/summary_detailed.md"

BEGIN = "<!-- OFFICIAL_BENCH -->"
OPEN = "<!-- OFFICIAL_BENCH:BEGIN -->"
CLOSE = "<!-- OFFICIAL_BENCH:END -->"

for p in (report, summary, detailed):
    if not os.path.exists(p):
        sys.exit("missing: %s" % p)

text = open(report, encoding="utf-8").read()
s = open(summary, encoding="utf-8").read().strip()
d = open(detailed, encoding="utf-8").read().strip()

block = "%s\n\n**官方格式表（`bench/summary.md` 原文，13 列）**\n\n%s\n\n**详细表（`bench/summary_detailed.md`）**\n\n%s\n\n%s" % (
    OPEN, s, d, CLOSE)

if CLOSE in text:
    text = re.sub(re.escape(OPEN) + r".*?" + re.escape(CLOSE),
                  block.replace("\\", "\\\\"), text, flags=re.S)
    mode = "replaced"
elif BEGIN in text:
    text = text.replace(BEGIN, block)
    mode = "filled"
else:
    sys.exit("marker %s not found in %s" % (BEGIN, report))

open(report, "w", encoding="utf-8").write(text)
print("[inject] %s %s (summary %d B, detailed %d B)" % (mode, report, len(s), len(d)))
