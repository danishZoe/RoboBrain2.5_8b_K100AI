#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自检：技术报告 README.md 的 Markdown 表格是否能被正常渲染。

检查三项：
  1) 每张表的**表头 / 分隔行 / 数据行列数一致**（不一致 -> Markdown 表格解析失败，
     渲染成纯文本，这正是 vLLM 性能表曾经消失的原因）；
  2) **HTML 注释块之后紧跟非空行**（CommonMark type 2：注释后的内容会被当作 HTML
     原样输出 -> 表格/标题不按 Markdown 渲染）；
  3) 实际渲染出的 <table> 数量 == Markdown 表数量。

用法:
    python3 tools/check_report_render.py [README.md]
    # 若无 pandoc，跳过第 3 项（仍检查 1、2）
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def resolve_readme(arg):
    if arg:
        return arg
    for cand in (os.path.join(ROOT, "README.md"),
                 os.path.join(ROOT, "submission", "README.md")):
        if os.path.exists(cand):
            return cand
    sys.exit("ERROR: 找不到 README.md")


def cells(line):
    s = line.strip()
    return [c.strip() for c in s.split("|")[1:-1]] if s.startswith("|") else None


def is_sep(line):
    s = line.strip()
    return bool(s.startswith("|")) and bool(re.match(r"^\|[\s:|-]+\|$", s))


def main():
    path = resolve_readme(sys.argv[1] if len(sys.argv) > 1 else None)
    lines = open(path, encoding="utf-8").read().split("\n")
    problems = []

    # --- 1) 表格列数一致性 + 收集表位置 ---
    tables = []
    in_code = False
    i = 0
    while i < len(lines):
        l = lines[i]
        if l.strip().startswith("```"):
            in_code = not in_code
            i += 1
            continue
        if not in_code and cells(l) and i + 1 < len(lines) and is_sep(lines[i + 1]):
            head = cells(l)
            sep = cells(lines[i + 1])
            j = i + 2
            data = []
            while j < len(lines) and cells(lines[j]) and not is_sep(lines[j]):
                data.append((j + 1, len(cells(lines[j]))))
                j += 1
            tables.append((i + 1, head))
            if len(head) != len(sep):
                problems.append("行%d 表头 %d 列 vs 分隔行 %d 列" % (i + 1, len(head), len(sep)))
            for ln, c in data:
                if c != len(sep):
                    problems.append("行%d 数据行 %d 列（分隔行 %d 列）" % (ln, c, len(sep)))
            i = j
            continue
        i += 1

    # --- 2) HTML 注释块后紧跟非空行 ---
    in_code = False
    for i, l in enumerate(lines):
        s = l.strip()
        if s.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if re.match(r"^<!--.*-->$", s):
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if nxt:
                problems.append("行%d HTML 注释后紧跟非空行（该内容不会被 Markdown 渲染）：%.50s"
                                % (i + 1, nxt))

    # --- 3) 实际渲染出的 <table> 数 ---
    rendered = None
    pandoc = None
    for p in ("pandoc", "/usr/bin/pandoc", r"D:\Anaconda\Scripts\pandoc.exe"):
        try:
            subprocess.run([p, "--version"], capture_output=True, check=True)
            pandoc = p
            break
        except Exception:
            continue
    if pandoc:
        out = os.path.join(ROOT, "tmp", "_render_check.html")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        r = subprocess.run([pandoc, "-f", "markdown", "-t", "html5", "-o", out, path],
                           capture_output=True)
        if r.returncode == 0:
            html = open(out, encoding="utf-8").read()
            rendered = len(re.findall(r"<table[^>]*>", html))
            os.remove(out)

    print("[render] 文件      : %s" % path)
    print("[render] Markdown 表: %d 张" % len(tables))
    if rendered is not None:
        print("[render] 渲染 <table>: %d 个" % rendered)
        if rendered != len(tables):
            problems.append("渲染出的 <table> 数 %d != Markdown 表数 %d" % (rendered, len(tables)))
    else:
        print("[render] 渲染 <table>: 跳过（未找到 pandoc）")

    if problems:
        print("[render] FAIL %d 项：" % len(problems))
        for p in problems:
            print("   - %s" % p)
        sys.exit(1)
    print("[render] PASS：全部表格列数一致、注释块后有空行、渲染数量匹配")


if __name__ == "__main__":
    main()
