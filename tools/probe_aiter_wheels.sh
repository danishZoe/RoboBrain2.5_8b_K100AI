#!/usr/bin/env bash
# 厂商源 aiter / flashinfer / wave-lang 完整性核查
# 纪律：只下载与解包检查，**不安装**到任何环境（当前服务正在跑压测，绝不能动共享 site-packages）
source /root/private_data/workspace/robo/tools/env_net.sh

W=/root/private_data/workspace/robo
IDX="https://pypi.sourcefind.cn/release/dtk2604/"
TMP="$W/tmp/pypi_probe"
DL="$W/tmp/wheel_probe"
mkdir -p "$TMP" "$DL"
cd "$DL" || exit 1

echo "=== 1. 候选包版本一览 ==="
for p in aiter flashinfer-python wave-lang sglang-kernel sglang-router vllm; do
  echo "--- $p ---"
  pip index versions "$p" -i "$IDX" 2>&1 | grep -E 'Available versions|INSTALLED|LATEST|no matching|ERROR' | head -3
done

echo
echo "=== 2. 下载 aiter 候选 wheel 并检查是否含 aiter/attention ==="
for v in "0.1.3+das.opt1.dtk2604.torch290.2605211823.gccde69" \
         "0.1.2+das.opt1.dtk2604.torch290.2605141109.g76b669" \
         "0.1.5+das185.dtk2604.torch2100.2608131714.gb0979d" ; do
  echo "--- aiter==$v ---"
  rm -f aiter-*.whl
  if pip download --no-deps -d "$DL" "aiter==$v" -i "$IDX" 2>&1 | tail -2 | grep -qiE 'saved|downloading'; then
    whl=$(ls -t aiter-*.whl 2>/dev/null | head -1)
    echo "  文件: $whl ($(du -h "$whl" | cut -f1))"
    echo "  含 attention 子包? -> $(unzip -l "$whl" 2>/dev/null | grep -c 'aiter/attention/')"
    echo "  顶层内容:"
    unzip -l "$whl" 2>/dev/null | awk '{print $4}' | grep -E '^aiter/[^/]+$' | sed 's/^/    /' | head -25
    echo "  paged_attention_ragged 是否存在:"
    unzip -l "$whl" 2>/dev/null | awk '{print $4}' | grep -iE 'paged_attention|pa_ragged|attention/' | head -8 | sed 's/^/    /'
  else
    echo "  下载失败（该版本可能不提供 wheel 或仅 sdist）"
  fi
done

echo
echo "=== 3. 镜像内已装 aiter 的实际内容（对照基准） ==="
python3 -c "import aiter,os;d=os.path.dirname(aiter.__file__);print(' ',d);print(' ',sorted(os.listdir(d)))" 2>&1 | tail -3
