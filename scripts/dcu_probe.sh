#!/bin/bash
W=/root/private_data/workspace/robo
cd "$W" || exit 1
echo "=== .revision ==="
cat .revision
echo
echo "=== env.sh ==="
cat env.sh
echo
echo "=== README.md (head 120) ==="
head -120 README.md
echo
echo "=== current python ==="
which python python3 pip pip3 2>/dev/null
python3 -V 2>&1
echo "=== installed vllm/torch in current env ==="
python3 - <<'PY' 2>&1 | tail -20
import importlib
for mod in ["torch","transformers","vllm","sglang","xformers","flash_attn"]:
    try:
        m = importlib.import_module(mod)
        print("%-14s %s" % (mod, getattr(m,"__version__","?")))
    except Exception as e:
        print("%-14s MISSING (%s)" % (mod, type(e).__name__))
PY
echo "=== pip list grep ==="
python3 -m pip list 2>/dev/null | grep -iE "^(vllm|torch|transformers|sglang|xformers|flash|triton|numpy) " 
echo "=== envs dir ==="
ls -la envs 2>&1 | head -20
echo "=== scripts dir ==="
find scripts -maxdepth 2 -type f 2>/dev/null | head -30
