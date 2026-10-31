#!/usr/bin/env bash
# 1) 本机 DTK/HIP 运行时版本
# 2) 厂商源索引：torch / vllm / vllm-hcu / sglang / sglang-kernel / aiter 的 torch2100 构建
set -o pipefail
W=/root/private_data/workspace/robo
# shellcheck disable=SC1091
source "$W/tools/env_net.sh" >/dev/null 2>&1

echo "===== 1) 本机 DTK / HIP ====="
cat /opt/dtk/.info 2>/dev/null || true
ls /opt/dtk/ 2>/dev/null | head -20
for f in /opt/dtk/VERSION /opt/dtk/version.txt /opt/dtk/.version; do [ -e "$f" ] && { echo "-- $f"; cat "$f"; }; done
/opt/dtk/bin/hipcc --version 2>/dev/null | head -6 || true
echo "--- torch (当前环境)"
python3 -c "import torch;print('torch',torch.__version__,'hip',torch.version.hip,'cuda_avail',torch.cuda.is_available())" 2>&1 | tail -2
python3 -c "import aiter,sys;print('aiter',getattr(aiter,'__version__','?'),aiter.__file__)" 2>&1 | tail -1
python3 -c "import sglang;print('sglang',getattr(sglang,'__version__','?'),sglang.__file__)" 2>&1 | tail -1
python3 -c "import vllm;print('vllm',vllm.__version__)" 2>&1 | tail -1

echo
echo "===== 2) 厂商源 simple 索引探测 ====="
BASE="https://pypi.sourcefind.cn/release/dtk2604"
for shape in "/+simple/torch/" "/torch/+simple/" "/+simple/vllm/" "/+simple/sglang/"; do
  url="$BASE$shape"
  code=$(curl -sS -m 30 -o "$W/tmp/idx.html" -w '%{http_code}' "$url" 2>/dev/null)
  n=$(grep -o 'href="[^"]*"' "$W/tmp/idx.html" 2>/dev/null | wc -l)
  echo "--- $url  http=$code  links=$n"
  if [ "$code" = "200" ] && [ "$n" -gt 0 ]; then
    grep -o '>[^<]*\.\(whl\|tar\.gz\)<' "$W/tmp/idx.html" | sed -e 's/^>//' -e 's/<$//' | sort -u | head -25
  fi
done
