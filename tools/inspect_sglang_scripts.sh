#!/usr/bin/env bash
# 现有 SGLang 探针脚本盘点（准备复用，不重写）
W=/root/private_data/workspace/robo
cd "$W" || exit 1
echo "=== scripts/ 下与 sglang 相关的脚本 ==="
ls -la scripts/ | grep -iE 'sglang|probe|attn' || echo "(无)"
echo
echo "=== scripts/probe_attn_backend.sh 内容 ==="
[ -f scripts/probe_attn_backend.sh ] && cat scripts/probe_attn_backend.sh || echo "不存在"
echo
echo "=== scripts/probe_one_request.py 关键部分 ==="
[ -f scripts/probe_one_request.py ] && sed -n '1,80p' scripts/probe_one_request.py || echo "不存在"
echo
echo "=== 已有 SGLang 启动脚本 ==="
[ -f scripts/serve_sglang.sh ] && cat scripts/serve_sglang.sh || echo "不存在"
