#!/usr/bin/env bash
# aiter JIT 机制：infer_schema 为什么在 import 期就炸？有没有旁路开关？
W=/root/private_data/workspace/robo
D=/usr/local/lib/python3.10/dist-packages/aiter

echo "=== 1. jit/core.py 里 compile_ops / infer_schema ==="
python3 - <<'PY'
import ast, os
p = "/usr/local/lib/python3.10/dist-packages/aiter/jit/core.py"
src = open(p, encoding="utf-8", errors="replace").read()
tree = ast.parse(src)
lines = src.splitlines()
want = {"compile_ops", "infer_schema", "get_hip_bare_metal_version",
        "get_hip_device_properties", "is_binary_build", "compile_module"}
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in want:
        print(f"--- def {node.name}  (line {node.lineno}) ---")
        # 打印函数体（去掉 docstring）
        seg = lines[node.lineno-1:node.end_lineno]
        out = []
        skip = False
        for l in seg:
            s = l.strip()
            if s.startswith('"""') or s.startswith("'''"):
                skip = not skip
                continue
            if skip:
                continue
            out.append(l)
        print("\n".join(out[:70]))
        print()
PY

echo "=== 2. jit/build 下已编译的模块（看 pa_ragged / attention / mha 在不在） ==="
ls "$D/jit/build/" 2>/dev/null | head -60
echo "--- 目标模块 ---"
for m in module_pa_ragged module_pa module_attention module_attention_asm module_mha module_aiter_operator; do
  if [ -d "$D/jit/build/$m" ]; then echo "  $m: 存在"; else echo "  $m: 缺失"; fi
done

echo
echo "=== 3. jit 里有没有 env 开关 / is_binary_build 之类 ==="
grep -rnoE 'AITER_[A-Z0-9_]+' "$D/jit/core.py" 2>/dev/null | sort -u -t: -k2 | head -20
grep -nE 'def compile_ops|is_binary_build|binary_build|AITER_' "$D/jit/core.py" | head -25

echo
echo "=== 4. 试着用不同 env 组合 import aiter.ops.attention ==="
for combo in "AITER_JIT_ENABLE=0" "AITER_DISABLE_JIT=1" "AITER_USE_PRECOMPILED=1" "AITER_INSTALL_MODE=binary"; do
  echo "--- $combo ---"
  env $combo python3 -c "
import aiter.ops.attention as m
print('  OK import, paged_attention_ragged =', hasattr(m,'paged_attention_ragged'))
" 2>&1 | grep -vE 'NUMA balancing' | tail -3
done
