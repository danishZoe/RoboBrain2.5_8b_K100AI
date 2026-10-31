#!/usr/bin/env python3
"""让提交包自检脚本能识别缩进的 export 语句（env.sh 里条件分支内的赋值）。

只改 grep 模式，不改判定逻辑：
  "^export $v="      → "^[[:space:]]*export $v="
"""
import re
import sys
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else "tools/check_submission.sh")
text = p.read_text(encoding="utf-8")
orig = text

text = text.replace(
    'if grep -q "^export $v=" env.sh; then',
    'if grep -qE "^[[:space:]]*export $v=" env.sh; then',
)
text = text.replace(
    'val=$(grep "^export $v=" env.sh | head -1 | cut -d= -f2- | sed \'s/[[:space:]]*#.*$//\' | tr -d \'"\' | cut -c1-72)',
    'val=$(grep -E "^[[:space:]]*export $v=" env.sh | head -1 | cut -d= -f2- | sed \'s/[[:space:]]*#.*$//\' | tr -d \'"\' | cut -c1-72)',
)
text = text.replace(
    'grep "^export $1=" env.sh | head -1 | cut -d= -f2- \\',
    'grep -E "^[[:space:]]*export $1=" env.sh | head -1 | cut -d= -f2- \\',
)

if text == orig:
    print("no change needed (already tolerant)")
    sys.exit(0)

p.write_text(text, encoding="utf-8")
print(f"patched: {p}")
for line in text.splitlines():
    if "export $v=" in line or "export $1=" in line:
        print("   ", line.strip()[:120])
