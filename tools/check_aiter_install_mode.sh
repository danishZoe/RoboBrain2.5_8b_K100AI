#!/usr/bin/env bash
# aiter 的 install_mode 机制：是否会在安装/运行时生成 aiter.attention 布局
W=/root/private_data/workspace/robo
D=/usr/local/lib/python3.10/dist-packages/aiter

echo "=== 1. install_mode 是什么 ==="
file "$D/install_mode" 2>/dev/null || ls -ld "$D/install_mode"
if [ -f "$D/install_mode" ]; then
  echo "--- 内容 ---"
  cat "$D/install_mode"
fi
if [ -d "$D/install_mode" ]; then
  find "$D/install_mode" -type f | head -20
  for f in $(find "$D/install_mode" -type f -name '*.py' | head -5); do
    echo "--- $f ---"; head -60 "$f"
  done
fi

echo
echo "=== 2. aiter 里与 install_mode / 模式开关相关的环境变量 ==="
grep -rhoE 'AITER_[A-Z0-9_]+' "$D" 2>/dev/null | sort | uniq -c | sort -rn | head -25

echo
echo "=== 3. aiter 里有没有生成 attention 子包的代码 ==="
grep -rn --include='*.py' -E 'aiter\.attention|"attention"|'"'"'attention'"'"'' "$D/ops/attention.py" "$D/__init__.py" 2>/dev/null | head -15
echo "--- ops/attention.py 里的 def ---"
grep -nE '^def |^class ' "$D/ops/attention.py" 2>/dev/null | head -20

echo
echo "=== 4. aiter/paged_attn.py 里 paged_attention_v2 的实现来源 ==="
sed -n '1,40p' "$D/paged_attn.py" 2>/dev/null | grep -nE 'import|from|def |JIT|jit' | head -20

echo
echo "=== 5. JIT 目录内容（是否含 attention 实现源） ==="
find "$D/jit" -maxdepth 3 -type d 2>/dev/null | head -20
find "$D/jit" -name '*attention*' -o -name '*ragged*' 2>/dev/null | head -20
