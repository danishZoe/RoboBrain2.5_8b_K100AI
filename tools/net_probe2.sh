#!/usr/bin/env bash
# 联网验证 + 官方材料检索入口探测
source /root/private_data/workspace/robo/tools/env_net.sh

echo "=== 1. 代理变量键名（不打印值） ==="
env | grep -iE '^(http|https|ftp|no)_proxy' | cut -d= -f1
echo "(空=未生效)"

echo
echo "=== 2. 连通性 ==="
for u in https://www.scnet.cn/ \
         https://gitlab.scnet.cn:9002/ \
         https://developer.sourcefind.cn/ \
         https://www.modelscope.cn/ \
         https://huggingface.co/ \
         https://pypi.tuna.tsinghua.edu.cn/simple/ ; do
  code=$(curl -sS -m 25 -o /dev/null -w '%{http_code}' "$u" 2>/dev/null)
  echo "$u -> ${code:-FAIL}"
done

echo
echo "=== 3. DCU 运行时 ==="
python3 -c "import torch;print('torch', torch.__version__, '| hip', torch.version.hip, '| cuda_avail', torch.cuda.is_available())" 2>&1 | tail -2

echo
echo "=== 4. 官方材料可能的主机：可达性 ==="
for u in https://developer.sourcefind.cn/api/v4/projects?per_page=1 \
         https://www.scnet.cn/api/aihub/image/list \
         https://gitlab.scnet.cn:9002/api/v4/groups ; do
  code=$(curl -sS -m 25 -o /dev/null -w '%{http_code}' "$u" 2>/dev/null)
  echo "$u -> ${code:-FAIL}"
done
