#!/usr/bin/env bash
# 1) 厂商源里 aiter / flashinfer 的 wheel 来自哪个 base（das_release 厂商自建 vs open/pypi 上游 CUDA）
# 2) 新版 sglang(das185) 的 aiter_backend.py 到底 import 哪些符号 —— 是否与 das185 aiter 对得上
source /root/private_data/workspace/robo/tools/env_net.sh
W=/root/private_data/workspace/robo
TMP="$W/tmp/pypi_probe"
mkdir -p "$TMP"
cd "$TMP" || exit 1

echo "=== 1. simple 页面里的 base 分布 ==="
for pkg in aiter flashinfer-python sglang; do
  echo "--- $pkg ---"
  curl -sS -m 60 -L "https://pypi.sourcefind.cn/release/dtk2604/+simple/$pkg/" 2>/dev/null \
   | grep -oE 'href="[^"]+"' | sed -E 's/href="//;s/"$//' | sed -E 's#(https?://[^/]+/)([^/]+/[^/]+/[^/]+)/.*#\1\2#' \
   | sort | uniq -c | sort -rn | head -12
done

echo
echo "=== 2. 下载 sglang das185 wheel，只取 aiter_backend.py 看 import ==="
pip download --no-deps -d "$TMP" "sglang==0.5.12+das185.dtk2604.torch2100.2608121934.gcbf025" \
  -i "https://pypi.sourcefind.cn/release/dtk2604/" 2>&1 | tail -3
SW=$(ls -t sglang-*.whl 2>/dev/null | head -1)
if [ -n "$SW" ]; then
  echo "--- $SW ---"
  unzip -p "$SW" 'sglang/srt/layers/attention/aiter_backend.py' 2>/dev/null \
    | grep -nE '^from |^import |^    from |^    import |except ImportError' | head -30
  echo
  echo "--- 该文件里用到的 paged_attention_ragged / flash_attn_varlen_func 定义处 ---"
  unzip -p "$SW" 'sglang/srt/layers/attention/aiter_backend.py' 2>/dev/null \
    | grep -nE 'paged_attention_ragged|flash_attn_varlen_func|unified_attention|aiter\.' | head -20
fi

echo
echo "=== 3. 当前镜像内 sglang 0.5.12+das.opt1 的同名文件 import（对照） ==="
python3 - <<'PY'
import os, re
p = "/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/attention/aiter_backend.py"
if not os.path.exists(p):
    print("  找不到", p); raise SystemExit
src = open(p, encoding="utf-8", errors="replace").read()
for i, line in enumerate(src.splitlines(), 1):
    if re.search(r"paged_attention_ragged|flash_attn_varlen_func|unified_attention|from aiter|except ImportError", line):
        print(f"  {i}: {line.strip()[:150]}")
PY

echo
echo "=== 4. 压测状态 ==="
date +%H:%M:%S
wc -l "$W"/submission/bench/round*.jsonl 2>/dev/null
tail -4 "$W"/logs/pipeline_after_bench.log 2>/dev/null
