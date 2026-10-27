#!/usr/bin/env bash
# 全量扫描 aiter，确认 SGLang 需要的 9 个符号在包内是否存在（只是没被 __init__ 导出）
W=/root/private_data/workspace/robo
D=/usr/local/lib/python3.10/dist-packages/aiter

python3 - <<'PY'
import ast, os

D = "/usr/local/lib/python3.10/dist-packages/aiter"
WANT = ["flash_attn_varlen_func", "get_mla_metadata_info_v1", "get_mla_metadata_v1",
        "get_ps_metadata_info_v1", "get_ps_metadata_v1", "mha_batch_prefill_func",
        "mla_prefill_ps_asm_fwd", "mla_reduce_v1", "paged_attention_ragged"]
SUB = {"aiter.mla": ["mla_decode_fwd", "mla_prefill_fwd"],
       "aiter.ops.triton.attention.unified_attention": ["unified_attention"]}

# 1) 收集所有 aiter 模块的顶层名字
top = {}
allsrc = {}
for root, dirs, files in os.walk(D):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for fn in files:
        if not fn.endswith(".py"):
            continue
        p = os.path.join(root, fn)
        rel = os.path.relpath(p, D).replace(os.sep, ".")
        try:
            src = open(p, encoding="utf-8", errors="replace").read()
            tree = ast.parse(src)
        except Exception:
            continue
        names = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        names.add(t.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    names.add(a.asname or a.name.split(".")[0])
        top[rel] = names
        allsrc[rel] = src

print("=== 9 个顶层符号在 aiter 包内的定义位置 ===")
missing = []
for w in WANT:
    hits = [m for m, ns in top.items() if w in ns]
    # 排除 __init__ 里的 star-import 伪命中
    hits = [m for m in hits if m != "__init__.py"]
    if hits:
        print(f"  {w:32s} -> {hits[:4]}")
    else:
        print(f"  {w:32s} -> ✗ 包内没有")
        missing.append(w)

print()
print("=== 3 个子模块导入路径 ===")
for mod, names in SUB.items():
    parts = mod.split(".")
    cand = [os.path.join(D, *parts) + ".py", os.path.join(D, *parts, "__init__.py")]
    real = [c for c in cand if os.path.exists(c)]
    print(f"  {mod}: 存在={bool(real)} {real}")
    if real:
        rel = os.path.relpath(real[0], D).replace(os.sep, ".")
        have = top.get(rel, set())
        for n in names:
            print(f"      {n}: {'有' if n in have else '✗ 缺'}")

print()
print("=== aiter/__init__.py 中被注释掉的 star-import（候选恢复点） ===")
init = allsrc.get("__init__.py", "")
for i, l in enumerate(init.splitlines(), 1):
    if l.strip().startswith("# from"):
        print(f"  {i:3d}: {l.strip()}")

print()
print("=== 若全部命中，给出可行的 shim 注入点 ===")
if not missing:
    print("  ✅ 9 个符号在 aiter 包内全部存在，只是没从 __init__ 导出")
    print("     → 可用 sitecustomize 垫片做显式 re-export，不改 site-packages")
else:
    print("  ✗ 仍缺:", missing)
PY

echo
echo "=== 压测状态 ==="
date +%H:%M:%S
wc -l "$W"/submission/bench/round*.jsonl 2>/dev/null
