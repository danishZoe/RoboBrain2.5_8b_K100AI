#!/usr/bin/env python3
"""补 .gitignore：排除运行时组装的调优库目录与 core 转储。"""
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else ".gitignore")
text = path.read_text(encoding="utf-8") if path.exists() else ""

BLOCK = """
# 运行时组装的 rocBLAS 调优库（仓库内只提交 6 个差异文件，运行时装配目录不进仓库）
/rocblas_tuned/

# 进程崩溃转储（节点上出现过 3 GB 级 core 文件，绝不能进仓库）
core
core.*
"""

if "/rocblas_tuned/" not in text:
    text = text.rstrip("\n") + "\n" + BLOCK
    path.write_text(text, encoding="utf-8")
    print(f"updated: {path}")
else:
    print("already present, skip")
