#!/usr/bin/env bash
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
echo "=== run_cv_bench.py 的 --image-protocol 默认值 ==="
grep -n -A3 'image-protocol' submission/bench_public/run_cv_bench.py | head -12
echo
echo "=== common.py 里各协议的处理函数 ==="
grep -nE 'def (canonical|up2|up4|raw|long1500|maxpix)|image_protocol ==|PROTOCOLS' submission/bench_public/common.py | head -20
echo
echo "=== 是否存在把 up2 当默认的迹象 ==="
grep -rn 'up2_png' submission/bench_public/*.py scripts/*.py 2>/dev/null | head -10
echo
echo "=== 提交包 scripts/ 里的图像规范化默认值 ==="
grep -rnE 'canonical|up2' scripts/image_canonical.py 2>/dev/null | head -10
