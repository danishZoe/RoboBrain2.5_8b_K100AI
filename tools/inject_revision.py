#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inject_revision.py - 把交付代码 revision 写进 reports/05_final.md 与
submission/README.md。

数据来源：工作目录下的 .revision（由 tools/record_revision.sh 生成），
不手写任何 revision 字符串。

Idempotent: 重复执行会替换标记块。

Usage: python3 tools/inject_revision.py
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

REV_FILE = ".revision"
BEGIN = "<!-- REVISION:BEGIN -->"
END = "<!-- /REVISION -->"


def read_revision():
    if not os.path.exists(REV_FILE):
        return None
    rec = {}
    for line in open(REV_FILE, encoding="utf-8"):
        line = line.strip()
        if "=" not in line or line.startswith("#"):
            continue
        k, v = line.split("=", 1)
        rec[k.strip()] = v.strip()
    return rec


def block(rec):
    return (BEGIN + "\n\n"
            "| 项 | 值 |\n|---|---|\n"
            "| 交付代码 revision（git commit） | `%s` |\n"
            "| 短 hash | `%s` |\n"
            "| 记录时间 | %s |\n"
            "| 纳入仓库的文件数 | %s |\n"
            "| 提交包逐文件校验值 | `submission/MANIFEST.sha256` |\n"
            "| 仓库内容范围 | `%s` |\n\n"
            "> 该 revision 由 `tools/record_revision.sh` 在交付包定稿后生成，"
            "仓库只含交付物（隔离依赖 `envs/pylibs/` 与运行产物 `outputs/`、`logs/` 不入仓库，"
            "可由提交包脚本重新生成）。PDF 要求代码与镜像分别提交到指定托管平台，"
            "**托管链接仍需用户上传后回填**；节点无外网，无法代替上传。\n"
            % (rec.get("revision", "-"), rec.get("revision_short", "-"),
               rec.get("recorded_at", "-"), rec.get("tracked_files", "-"),
               rec.get("repo_layout", "-")))


def inject(text, body):
    if BEGIN in text and END in text:
        return re.sub(re.escape(BEGIN) + r".*?" + re.escape(END),
                      (BEGIN + "\n" + body + END).replace("\\", "\\\\"),
                      text, flags=re.S)
    return text


def main():
    rec = read_revision()
    if not rec:
        print("[rev] .revision not found - run tools/record_revision.sh first")
        return 2
    body = block(rec)

    p = "reports/05_final.md"
    if os.path.exists(p):
        t = open(p, encoding="utf-8").read()
        if BEGIN in t:
            t = inject(t, body)
        else:
            # 插到「## 5. 兼容性改动清单」之后、§6 之前
            anchor = "\n---\n\n## 6. 目录与产物索引"
            ins = ("\n### 5.1 交付代码 revision\n\n" + body + "\n" + anchor)
            if anchor in t:
                t = t.replace(anchor, ins, 1)
            else:
                t = t.rstrip("\n") + "\n\n### 5.1 交付代码 revision\n\n" + body
        open(p, "w", encoding="utf-8").write(t)
        print("[rev] %s updated" % p)

    p = "submission/README.md"
    if os.path.exists(p):
        t = open(p, encoding="utf-8").read()
        rev = rec.get("revision", "-")
        short = rec.get("revision_short", "-")
        old = "- 提交仓库：【待填】"
        new = ("- 提交仓库：【待填：托管平台可访问链接】；"
               "本次交付代码 revision = `%s`（短 `%s`，逐文件校验值见 "
               "`MANIFEST.sha256`，详见 `.revision`）" % (rev, short))
        if old in t:
            t = t.replace(old, new, 1)
            open(p, "w", encoding="utf-8").write(t)
            print("[rev] %s updated" % p)
        else:
            # 已经是「带 revision 的长形式」：只替换 hash，其余文字（含托管平台
            # 待填占位符）原样保留，避免每次回填都把用户填的链接冲掉。
            pat = re.compile(r"(revision = `)[0-9a-f]{7,40}(`（短 `)[0-9a-f]{7,40}(`)")
            t2, n = pat.subn(
                lambda m: m.group(1) + rev + m.group(2) + short + m.group(3),
                t, count=1)
            if n:
                open(p, "w", encoding="utf-8").write(t2)
                print("[rev] %s revision hash updated -> %s" % (p, short))
            else:
                print("[rev] %s: 提交仓库占位符未找到，跳过" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
