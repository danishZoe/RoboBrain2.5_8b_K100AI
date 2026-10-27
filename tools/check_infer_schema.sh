#!/usr/bin/env bash
# 决定性检查：infer_schema 为何在 import 期失败？das185 版是否修好？
W=/root/private_data/workspace/robo
TMP="$W/tmp/pypi_probe"
cd "$TMP" || exit 1

echo "=== 1. 镜像内 jit/core.py：compile_ops 尾部是否 eager 调用 infer_schema ==="
python3 - <<'PY'
import ast
p = "/usr/local/lib/python3.10/dist-packages/aiter/jit/core.py"
src = open(p, encoding="utf-8", errors="replace").read()
lines = src.splitlines()
tree = ast.parse(src)
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ("compile_ops", "infer_schema"):
        seg = lines[node.lineno-1:node.end_lineno]
        # 找 infer_schema 调用点及其所在行号
        print(f"--- {node.name} (lines {node.lineno}-{node.end_lineno}) ---")
        for i, l in enumerate(seg):
            if "infer_schema" in l or "arg_checked" in l or "schema" in l.lower():
                print(f"   {node.lineno+i}: {l.rstrip()}")
        print()
PY

echo "=== 2. 镜像内 aiter.ops.attention 的 paged_attention_ragged 签名（有返回注解吗） ==="
unzip -p "$(ls -t aiter-*das.opt1*torch290*.whl | head -1)" 'aiter/ops/attention.py' 2>/dev/null | sed -n '70,105p'
echo "--- 镜像内已装版本 ---"
sed -n '77,104p' /usr/local/lib/python3.10/dist-packages/aiter/ops/attention.py

echo
echo "=== 3. das185 版（torch2100）的同名函数签名 ==="
W2=$(ls -t aiter-*das185*.whl 2>/dev/null | head -1)
if [ -n "$W2" ]; then
  echo "  wheel: $W2"
  unzip -p "$W2" 'aiter/ops/attention.py' 2>/dev/null | grep -nA4 'def paged_attention_ragged' | head -12
  echo "  --- das185 的 infer_schema 是否放宽（关键片段） ---"
  unzip -p "$W2" 'aiter/jit/core.py' 2>/dev/null | grep -nE 'No return type annotation|unsupported type|def infer_schema' | head -8
fi

echo
echo "=== 4. das.opt1 wheel 版（2605211823）的 infer_schema 关键行 ==="
W1=$(ls -t aiter-*das.opt1*torch290*.whl 2>/dev/null | head -1)
unzip -p "$W1" 'aiter/jit/core.py' 2>/dev/null | grep -nE 'No return type annotation|unsupported type|def infer_schema' | head -8

echo
echo "=== 5. 镜像内 jit/core.py 对应行 ==="
grep -nE 'No return type annotation|unsupported type|def infer_schema' /usr/local/lib/python3.10/dist-packages/aiter/jit/core.py | head -8

echo
echo "=== 6. 压测状态 ==="
date +%H:%M:%S
wc -l "$W"/submission/bench/round*.jsonl 2>/dev/null
