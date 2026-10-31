#!/usr/bin/env bash
# 符号级核查：aiter 各版本是否真的导出 sglang 需要的符号
# （之前只查了文件名，而 paged_attention_ragged 是 aiter/__init__.py 里的符号 —— 必须查导出）
W=/root/private_data/workspace/robo
TMP="$W/tmp/pypi_probe"
cd "$TMP" || exit 1

echo "########## A. 镜像内已装 aiter 的导出 ##########"
python3 - <<'PY'
import os, re
d = "/usr/local/lib/python3.10/dist-packages/aiter"
init = os.path.join(d, "__init__.py")
src = open(init, encoding="utf-8", errors="replace").read() if os.path.exists(init) else ""
print("  __init__.py 行数:", len(src.splitlines()))
for sym in ("paged_attention_ragged", "flash_attn_varlen_func", "unified_attention"):
    hits = [l.strip() for l in src.splitlines() if sym in l]
    print(f"  {sym}: {len(hits)} 处")
    for h in hits[:4]:
        print("      ", h[:120])
print("  --- paged_attn.py 里定义的符号 ---")
pa = os.path.join(d, "paged_attn.py")
if os.path.exists(pa):
    t = open(pa, encoding="utf-8", errors="replace").read()
    for m in re.finditer(r"^def (\w+)", t, re.M):
        print("      def", m.group(1))
    print("      文件行数:", len(t.splitlines()))
print("  --- 是否有 ops/triton/attention 子目录 ---")
for root, dirs, files in os.walk(os.path.join(d, "ops")):
    rel = os.path.relpath(root, d)
    if "attention" in rel or rel.count(os.sep) <= 1:
        print("      ", rel, "->", sorted(files)[:6])
PY

echo
echo "########## B. das.opt1+torch290 wheel（0.1.3 ...2605211823）的导出 ##########"
W1=$(ls -t aiter-*das.opt1*torch290*.whl 2>/dev/null | head -1)
if [ -n "$W1" ]; then
  echo "  wheel: $W1"
  echo "  --- __init__.py 里关键符号 ---"
  unzip -p "$W1" 'aiter/__init__.py' 2>/dev/null | grep -nE 'paged_attention_ragged|flash_attn_varlen_func|unified_attention|from \.|^from ' | head -25
  echo "  --- paged_attn.py 的 def ---"
  unzip -p "$W1" 'aiter/paged_attn.py' 2>/dev/null | grep -nE '^def |^class ' | head -15
  echo "  --- ops/triton/attention 是否存在 ---"
  unzip -l "$W1" 2>/dev/null | awk '{print $4}' | grep -E 'ops/triton/(attention/)?[a-z_]*unified' | head -8
  echo "  --- ops/triton/attention 目录清单 ---"
  unzip -l "$W1" 2>/dev/null | awk '{print $4}' | grep '^aiter/ops/triton/attention/' | head -10
fi

echo
echo "########## C. das185+torch2100 wheel（0.1.5）的导出 ##########"
W2=$(ls -t aiter-*das185*.whl 2>/dev/null | head -1)
if [ -n "$W2" ]; then
  echo "  wheel: $W2"
  unzip -p "$W2" 'aiter/__init__.py' 2>/dev/null | grep -nE 'paged_attention_ragged|flash_attn_varlen_func|unified_attention' | head -15
  echo "  --- paged_attn.py 的 def ---"
  unzip -p "$W2" 'aiter/paged_attn.py' 2>/dev/null | grep -nE '^def ' | head -12
  echo "  --- ops/triton/attention 目录 ---"
  unzip -l "$W2" 2>/dev/null | awk '{print $4}' | grep '^aiter/ops/triton/attention/' | head -8
fi

echo
echo "########## D. sglang 0.5.10rc0+das.opt2.alpha+torch290 的 aiter_backend import ##########"
pip download --no-deps -d "$TMP" "sglang==0.5.10rc0+das.opt2.alpha.dtk2604.torch290.2605181752.g91be88" \
  -i "https://pypi.sourcefind.cn/release/dtk2604/" 2>&1 | tail -2
SW2=$(ls -t sglang-0.5.10rc0*.whl 2>/dev/null | head -1)
if [ -n "$SW2" ]; then
  echo "  wheel: $SW2"
  unzip -p "$SW2" 'sglang/srt/layers/attention/aiter_backend.py' 2>/dev/null \
    | sed -n '30,80p' | grep -nE 'from aiter|from sglang|paged_attention_ragged|flash_attn_varlen_func|unified_attention|except ImportError'
fi

echo
echo "########## E. 压测状态 ##########"
date +%H:%M:%S
wc -l "$W"/submission/bench/round*.jsonl 2>/dev/null
