#!/usr/bin/env bash
source /root/private_data/workspace/robo/tools/env_net.sh >/dev/null 2>&1
echo "=== gitlab 根路径 ==="
curl -sS -m 20 -o /dev/null -w 'root            %{http_code}\n' https://gitlab.scnet.cn:9002/ 2>&1
echo "=== 仓库 refs 端点（git upload-pack）==="
for m in RoboBrain2.5-8B-NV RoboBrain2-5-8B-NV; do
  code=$(curl -sS -m 25 -o /dev/null -w '%{http_code}' \
    "https://gitlab.scnet.cn:9002/model/sugon_scnet/${m}.git/info/refs?service=git-upload-pack" 2>/dev/null)
  echo "$code  ${m}"
done
echo "=== 任意一个已知仓库的 refs 端点（用于确认该端点语义）==="
curl -sS -m 25 -o /dev/null -w 'MiMo-V2.6       %{http_code}\n' \
  'https://gitlab.scnet.cn:9002/model/sugon_scnet/MiMo-V2.6.git/info/refs?service=git-upload-pack' 2>&1
echo "=== done ==="
