#!/usr/bin/env bash
# 厂商内部 PyPI（DTK 26.04 源）探测：跟随重定向，找完整 aiter
source /root/private_data/workspace/robo/tools/env_net.sh

W=/root/private_data/workspace/robo
IDX="https://pypi.sourcefind.cn/release/dtk2604/"
TMP="$W/tmp/pypi_probe"
mkdir -p "$TMP"

echo "=== 1. 重定向链 ==="
curl -sS -m 40 -L -o "$TMP/index.html" -w 'final=%{url_effective}\ncode=%{http_code}\nsize=%{size_download}\n' "$IDX" 2>&1 | tail -5
echo

if [ -s "$TMP/index.html" ]; then
  head -c 400 "$TMP/index.html"; echo; echo "..."
  echo
  echo "=== 2. 包名清单 ==="
  grep -oE '>[A-Za-z0-9._+-]+</a>' "$TMP/index.html" | sed -E 's/[><]//g;s#/.*##' | sort -u > "$TMP/names.txt"
  wc -l < "$TMP/names.txt"
  echo "--- 前 60 ---"
  head -60 "$TMP/names.txt"
  echo
  echo "=== 3. 关键包命中 ==="
  for p in aiter sglang vllm flashinfer wave-lang wave_lang triton xformers transformers torch; do
    if grep -qiE "^${p}$" "$TMP/names.txt"; then echo "  ${p}: FOUND"; else
      m=$(grep -iE "^${p}" "$TMP/names.txt" | head -3 | tr '\n' ' ')
      echo "  ${p}: - ${m}"
    fi
  done
  echo
  echo "=== 4. aiter 相关条目全文 ==="
  grep -ioE '[^"<>]*aiter[^"<>]*' "$TMP/index.html" | sort -u | head -30
fi

echo
echo "=== 5. pip 解析 aiter ==="
pip index versions aiter -i "$IDX" 2>&1 | head -6
echo "--- sglang ---"
pip index versions sglang -i "$IDX" 2>&1 | head -6
