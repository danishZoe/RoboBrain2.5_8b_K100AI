#!/usr/bin/env bash
source /root/private_data/workspace/robo/tools/env_net.sh
W=/root/private_data/workspace/robo
TMP="$W/tmp/pypi_probe"
mkdir -p "$TMP"; cd "$TMP" || exit 1

echo "=== 1. import aiter.ops.attention 的准确 traceback ==="
python3 - <<'PY' 2>&1 | grep -v 'NUMA balancing' | tail -30
import traceback
try:
    import aiter.ops.attention  # noqa
    print("IMPORT OK")
except Exception:
    traceback.print_exc()
PY

echo
echo "=== 2. 精确确认 infer_schema 的调用栈位置 ==="
python3 - <<'PY' 2>&1 | grep -v 'NUMA balancing' | tail -25
import traceback, sys
try:
    import aiter.ops.attention  # noqa
except Exception:
    tb = sys.exc_info()[2]
    frames = traceback.extract_tb(tb)
    for f in frames[-8:]:
        print("  %s:%s in %s" % (f.filename.split('/')[-1], f.lineno, f.name))
        print("      " + (f.line or "")[:120])
PY

echo
echo "=== 3. sglang 0.5.10rc0+das.opt2.alpha+torch290 直接按 URL 下载 ==="
URL="https://pypi.sourcefind.cn/das_release/das1.8.2/+f/52e/42bb94d2e2a6a/sglang-0.5.10rc0%2Bdas.opt2.alpha.dtk2604.torch290.2605181752.g91be88-py3-none-any.whl"
curl -sSL -m 180 -o sglang_opt2alpha.whl "$URL" && echo "  下载完成: $(du -h sglang_opt2alpha.whl | cut -f1)"
if unzip -t sglang_opt2alpha.whl >/dev/null 2>&1; then
  echo "  --- 该版本的 aiter_backend.py 导入段 ---"
  unzip -p sglang_opt2alpha.whl 'sglang/srt/layers/attention/aiter_backend.py' 2>/dev/null \
    | sed -n '/^try:/,/^except ImportError/p' | head -40
  echo "  --- 它的 attention backend 文件清单 ---"
  unzip -l sglang_opt2alpha.whl 2>/dev/null | awk '{print $4}' | grep 'srt/layers/attention/' | head -20
else
  echo "  下载无效（不是 wheel）"
  head -c 200 sglang_opt2alpha.whl
fi

echo
echo "=== 4. 压测状态 ==="
date +%H:%M:%S
wc -l "$W"/submission/bench/round*.jsonl 2>/dev/null
