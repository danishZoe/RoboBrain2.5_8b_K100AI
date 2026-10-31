#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""insert_section.py - insert a markdown section into a report at a stable anchor.

Used to keep the hand-written parts of the reports (analysis, caveats) in
separate files that can be re-applied idempotently after the automatic table
injection (``tools/inject_report10_bench.py``) rewrites a report.

    python3 tools/insert_section.py REPORT SECTION_FILE [--anchor "text"] \
            [--replace] [--marker "<!-- SECTION:NAME -->"]

Behaviour:
  * The section file may contain a marker comment ``<!-- SECTION:NAME -->``.
    If the report already contains that marker, the text between the marker
    and the end marker is replaced (idempotent re-apply).
  * Otherwise the section is inserted *before* the anchor line.
  * ``--replace`` replaces the anchor block instead of inserting before it.
"""
import argparse
import os
import re
import sys


def find_marker(text, marker):
    if not marker:
        return None
    m = re.search(re.escape(marker), text)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("section")
    ap.add_argument("--anchor", default=None)
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--marker", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.report):
        sys.exit("missing report: %s" % args.report)
    if not os.path.exists(args.section):
        sys.exit("missing section: %s" % args.section)

    report = open(args.report, encoding="utf-8").read()
    with open(args.section, encoding="utf-8") as f:
        section = f.read().rstrip() + "\n"

    marker = args.marker
    if marker is None:
        m = re.search(r"<!--\s*SECTION:([A-Za-z0-9_.-]+)\s*-->", section)
        marker = ("<!-- SECTION:%s:BEGIN -->" % m.group(1)) if m else None

    if marker and marker in report:
        # replace from marker to the matching ...:END --> or the next '---'
        begin = marker
        endm = begin.replace(":BEGIN -->", ":END -->")
        idx = report.index(begin)
        if endm in report:
            end = report.index(endm) + len(endm)
        else:
            end = report.find("\n---", idx)
            end = len(report) if end == -1 else end
        new = report[:idx] + section.rstrip() + "\n" + report[end:]
        open(args.report, "w", encoding="utf-8").write(new)
        print("[insert] replaced %s in %s" % (marker, args.report))
        return 0

    if not args.anchor:
        sys.exit("need --anchor (or an existing marker) for %s" % args.report)
    # allow multi-line anchors written as the two-character sequence \n
    anchor = args.anchor.replace("\\n", "\n")
    if anchor not in report:
        sys.exit("anchor not found in %s:\n%r" % (args.report, anchor))
    if args.replace:
        new = report.replace(anchor, section, 1)
        mode = "replaced"
    else:
        new = report.replace(anchor, section + "\n" + anchor, 1)
        mode = "inserted before"
    open(args.report, "w", encoding="utf-8").write(new)
    print("[insert] %s %s bytes in %s" % (mode, len(section), args.report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
