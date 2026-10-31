#!/usr/bin/env python3
"""补 .gitignore：排除厂商调优工具的 8 GB 工作目录（证据已固化到 reports/evidence/）。"""
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else ".gitignore")
text = path.read_text(encoding="utf-8") if path.exists() else ""

BLOCK = """
# 厂商 GEMM 调优工具的工作目录（4 个实例合计 8 GB+，含 Tensile 二进制与旧库副本）。
# 可复核的关键证据（选优日志、shape 采集、算子级对比 CSV）已固化到
# reports/evidence/gemm_tuning_20261030/；生效的候选库差异文件在
# submission/rocblas_tuned/library_gpu4/。
tools/auto_select_tools*/
"""

if "tools/auto_select_tools*/" not in text:
    text = text.rstrip("\n") + "\n" + BLOCK
    path.write_text(text, encoding="utf-8")
    print(f"updated: {path}")
else:
    print("already present, skip")
