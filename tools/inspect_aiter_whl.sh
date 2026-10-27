#!/usr/bin/env bash
# 检查已下载的 aiter wheel 是否含 aiter/attention 子包（只读，不安装）
W=/root/private_data/workspace/robo
TMP="$W/tmp/pypi_probe"
cd "$TMP" || exit 1

for whl in aiter-*.whl; do
  [ -f "$whl" ] || continue
  echo "############################################################"
  echo "### $whl"
  echo "### size: $(du -h "$whl" | cut -f1)"
  echo "############################################################"
  echo "--- aiter/attention/ 条目数 ---"
  unzip -l "$whl" 2>/dev/null | awk '{print $4}' | grep -c '^aiter/attention/'
  echo "--- aiter/attention/ 清单 ---"
  unzip -l "$whl" 2>/dev/null | awk '{print $4}' | grep '^aiter/attention/' | head -25
  echo "--- 关键符号文件 ---"
  for k in paged_attention_ragged flash_attn_varlen_func unified_attention pa_ragged; do
    hits=$(unzip -l "$whl" 2>/dev/null | awk '{print $4}' | grep -c "$k")
    echo "    $k : $hits"
    unzip -l "$whl" 2>/dev/null | awk '{print $4}' | grep "$k" | head -3 | sed 's/^/       /'
  done
  echo "--- aiter 顶层子模块 ---"
  unzip -l "$whl" 2>/dev/null | awk '{print $4}' | grep -E '^aiter/[^/]+$' | sed 's/^/    /' | sort | head -40
  echo
done
