#!/usr/bin/env bash
# SGLang aiter_backend 需要的符号 vs aiter 实际提供的符号 —— 精确差集
W=/root/private_data/workspace/robo
D=/usr/local/lib/python3.10/dist-packages/aiter
S=/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/attention/aiter_backend.py

echo "=== 1. SGLang 从 aiter 导入的完整符号清单 ==="
sed -n '35,52p' "$S"

echo
echo "=== 2. aiter/__init__.py 第 40-78 行（看哪些 import 被注释掉） ==="
sed -n '40,78p' "$D/__init__.py" | nl -ba -v40

echo
echo "=== 3. aiter/ops/attention.py 顶层定义 ==="
grep -nE '^def |^class |^@|jit' "$D/ops/attention.py" | head -40

echo
echo "=== 4. aiter/ops/triton/unified_attention.py 顶层定义 ==="
grep -nE '^def |^class |^@|^import |^from ' "$D/ops/triton/unified_attention.py" | head -30

echo
echo "=== 5. aiter/ops/attention.py 头部 import（看是否触发 JIT 编译） ==="
sed -n '1,14p' "$D/ops/attention.py"

echo
echo "=== 6. 精确差集（纯静态分析，不 import，不触发 JIT） ==="
python3 - <<'PY'
import re, ast, os

S = "/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/attention/aiter_backend.py"
D = "/usr/local/lib/python3.10/dist-packages/aiter"

src = open(S, encoding="utf-8", errors="replace").read()
tree = ast.parse(src)

wanted = []          # from aiter import (...)
mod_imports = []     # from aiter.X import Y
for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("aiter"):
        for a in node.names:
            if node.module == "aiter":
                wanted.append(a.name)
            else:
                mod_imports.append((node.module, a.name))
print("SGLang 从 aiter 顶层导入的符号:", wanted)
print("SGLang 从 aiter 子模块导入:", mod_imports)

# aiter/ops/attention.py 里实际定义的名字
defs = set()
for fn in ("ops/attention.py", "ops/triton/unified_attention.py", "paged_attn.py", "mla.py"):
    p = os.path.join(D, fn)
    if not os.path.exists(p):
        continue
    t = ast.parse(open(p, encoding="utf-8", errors="replace").read())
    for node in t.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs.add(node.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    defs.add(tgt.id)
print()
print("aiter/ops/attention.py + ops/triton/unified_attention.py + paged_attn.py + mla.py 定义的顶层名字:")
print("  ", sorted(defs))

init_src = open(os.path.join(D, "__init__.py"), encoding="utf-8", errors="replace").read()
commented = [l.strip() for l in init_src.splitlines() if l.strip().startswith("# from .ops.attention")]
print()
print("aiter/__init__.py 中被注释掉的 attention 导入:", commented)

print()
missing = [w for w in wanted if w not in defs]
present = [w for w in wanted if w in defs]
print("★ 顶层符号: 定义存在于 aiter 但未从 __init__ 导出 ->", present)
print("★ 顶层符号: aiter 里根本没有 ->", missing)
for mod, name in mod_imports:
    p = mod.replace(".", "/")
    cand = [os.path.join(D, p + ".py"), os.path.join(D, p, "__init__.py")]
    ok = any(os.path.exists(c) for c in cand)
    print(f"  {mod}.{name}: 路径存在={ok}  ({', '.join(os.path.relpath(c, D) for c in cand if os.path.exists(c)) or '缺失'})")
PY
