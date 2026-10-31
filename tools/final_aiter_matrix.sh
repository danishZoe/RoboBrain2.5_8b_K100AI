#!/usr/bin/env bash
source /root/private_data/workspace/robo/tools/env_net.sh
W=/root/private_data/workspace/robo
TMP="$W/tmp/pypi_probe"
cd "$TMP" || exit 1

echo "=== 1. opt2alpha wheel 真实大小/完整性 ==="
ls -la sglang_opt2alpha.whl 2>/dev/null
unzip -l sglang_opt2alpha.whl 2>/dev/null | tail -3

echo
echo "=== 2. 三个 aiter 版本的 paged_attention_rocm 签名对比（有无返回注解） ==="
for label in "installed:/usr/local/lib/python3.10/dist-packages/aiter/ops/attention.py"; do
  name="${label%%:*}"; path="${label#*:}"
  echo "--- $name ---"
  grep -nA3 'def paged_attention_rocm' "$path" 2>/dev/null | head -5
  # 打印到下一个 def 之间的结尾（看有没有 -> None）
  awk '/def paged_attention_rocm/{f=1} f&&/^\)/{print; exit} f' "$path" 2>/dev/null | tail -3
done

for whl in aiter-*das.opt1*torch290*.whl aiter-*das185*.whl; do
  [ -f "$whl" ] || continue
  echo "--- $whl ---"
  unzip -p "$whl" 'aiter/ops/attention.py' 2>/dev/null | awk '/def paged_attention_rocm/{f=1} f&&/^\)/{print; exit} f' | tail -3
done

echo
echo "=== 3. das185 的 jit/torch_guard.py 是否也走 infer_schema ==="
W2=$(ls -t aiter-*das185*.whl 2>/dev/null | head -1)
unzip -p "$W2" 'aiter/jit/utils/torch_guard.py' 2>/dev/null | grep -nE 'infer_schema|wrapper_register' | head -8

echo
echo "=== 4. 厂商源里还有哪些 torch290 的 sglang/aiter 组合（最终可用性矩阵） ==="
for p in sglang sglang-kernel aiter flashinfer-python; do
  echo "--- $p ---"
  curl -sS -m 60 -L "https://pypi.sourcefind.cn/release/dtk2604/+simple/$p/" 2>/dev/null \
   | grep -oE '>[^<]*\.whl<' | sed -E 's/[><]//g' | sort -u | head -12
done

echo
echo "=== 5. 压测状态 ==="
date +%H:%M:%S
wc -l "$W"/submission/bench/round*.jsonl 2>/dev/null
