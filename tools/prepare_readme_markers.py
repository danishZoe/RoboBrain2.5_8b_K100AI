#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prepare_readme_markers.py — 给 submission/README.md 的 §6.2 精度表加可回填的标记块。

背景
----
`tools/inject_accuracy_table.py` 负责把精度总表**回填**进 README，但需要先有
可定位、可幂等替换的标记块：

* ``PUBLIC_BENCH_TABLE``  —— 原文件里是一个**没有脚本负责填充**的裸占位符，
  正文写着「见下表」但下面没有表。本脚本把它升级成 BEGIN/END 标记块。
* ``UPSCALE_TABLE``       —— §6.2.3 的 `Count` 788 题上采样表，原先是**手写**的，
  数字可能与原始 JSON 漂移。本脚本把手写表整体替换成标记块（内容由注入脚本重生成）。
* ``UPSCALE_FULL_TABLE``  —— §6.2.3 的全量 2638 题上采样表，同样是手写的，
  原文件里连标记都没有，本脚本补上标记块。

只做**结构**改动，不产生任何数字；数字一律由注入脚本从原始 JSON 生成。

幂等：重复执行不会重复包裹，也不会改变已生成的表格内容。

用法::

    python3 tools/prepare_readme_markers.py
    python3 tools/prepare_readme_markers.py --readme submission/README.md
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LEGACY_PUBLIC = "<!-- PUBLIC_BENCH_TABLE -->"
LEGACY_UPSCALE = "<!-- UPSCALE_TABLE -->"
PUBLIC_B = "<!-- PUBLIC_BENCH_TABLE:BEGIN -->"
PUBLIC_E = "<!-- /PUBLIC_BENCH_TABLE -->"

UPSCALE_B = "<!-- UPSCALE_TABLE:BEGIN -->"
UPSCALE_E = "<!-- /UPSCALE_TABLE -->"

FULL_B = "<!-- UPSCALE_FULL_TABLE:BEGIN -->"
FULL_E = "<!-- /UPSCALE_FULL_TABLE -->"

# 全量上采样表的表头（唯一，用于定位）
FULL_HEAD = "| 指标 | 主口径 `canonical_png` | `up2_png`（诊断口径） | Δ |"


def _is_table_row(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.endswith("|")


def _wrap(lines, start, stop, begin, end):
    """把 [start, stop) 行替换成标记块。"""
    return lines[:start] + [begin, end] + lines[stop:]


def upgrade_public(lines):
    """裸占位符 -> BEGIN/END 标记块。"""
    if PUBLIC_B in "\n".join(lines) and PUBLIC_E in "\n".join(lines):
        return lines, 0, "已存在标记块"
    for i, ln in enumerate(lines):
        if ln.strip() == LEGACY_PUBLIC:
            return lines[:i] + [PUBLIC_B, PUBLIC_E] + lines[i + 1:], 1, "已升级占位符"
    return lines, 0, "未找到占位符"


def upgrade_upscale(lines):
    """手写的 Count 788 上采样表 -> 标记块。"""
    txt = "\n".join(lines)
    if UPSCALE_B in txt and UPSCALE_E in txt:
        return lines, 0, "已存在标记块"
    for i, ln in enumerate(lines):
        if ln.strip() == LEGACY_UPSCALE:
            # 找到标记后紧随其后的手写表并整体删除
            j = i + 1
            # 跳过空行
            while j < len(lines) and not lines[j].strip():
                j += 1
            # 跳过「结果（...）」说明行
            if j < len(lines) and lines[j].startswith("结果（`Count`"):
                j += 1
                # 说明行与表头之间还有一个空行，必须再跳一次，
                # 否则表格行不被识别，手写表会留在原地造成重复。
                while j < len(lines) and not lines[j].strip():
                    j += 1
            # 吃掉整张手写表（表头 + 分隔行 + 全部数据行）
            while j < len(lines) and _is_table_row(lines[j]):
                j += 1
            # 回退掉尾部空行
            while j > i + 1 and not lines[j - 1].strip():
                j -= 1
            return lines[:i] + [UPSCALE_B, UPSCALE_E] + lines[j:], 1, "已替换手写表"
    return lines, 0, "未找到手写表"


def upgrade_upscale_full(lines):
    """全量 2638 上采样表（无标记）-> 标记块。"""
    txt = "\n".join(lines)
    if FULL_B in txt and FULL_E in txt:
        return lines, 0, "已存在标记块"
    for i, ln in enumerate(lines):
        if ln.strip() == FULL_HEAD:
            j = i + 1
            while j < len(lines) and _is_table_row(lines[j]):
                j += 1
            while j > i + 1 and not lines[j - 1].strip():
                j -= 1
            return lines[:i] + [FULL_B, FULL_E] + lines[j:], 1, "已替换手写表"
    return lines, 0, "未找到全量表"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", default=os.path.join(ROOT, "submission/README.md"))
    args = ap.parse_args()

    path = args.readme
    with open(path, encoding="utf-8", newline="") as fh:
        text = fh.read()
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(nl)

    total = 0
    for fn in (upgrade_public, upgrade_upscale, upgrade_upscale_full):
        lines, n, msg = fn(lines)
        total += n
        print("[markers] %-20s %s (%d)" % (fn.__name__, msg, n))

    if total == 0:
        print("[markers] %s 无需改动" % path)
        return 0

    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(nl.join(lines))
    print("[markers] %s updated (%d 处)" % (path, total))
    return 0


if __name__ == "__main__":
    sys.exit(main())
