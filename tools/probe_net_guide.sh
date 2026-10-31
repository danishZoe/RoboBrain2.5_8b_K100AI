#!/usr/bin/env bash
# 核对：指南引用的 env.sh 内容 + 代理是否真的生效 + 直连是否仍不通
echo "=== A. /root/private_data/workspace/scripts/env.sh 内容 ==="
cat /root/private_data/workspace/scripts/env.sh

echo
echo "=== B. source env_net.sh 后代理变量是否在位（只看键名，不打印口令） ==="
set +u
source /root/private_data/workspace/robo/tools/env_net.sh
for v in http_proxy https_proxy ftp_proxy no_proxy; do
  val="${!v}"
  if [ -z "$val" ]; then
    echo "$v = <EMPTY>"
  else
    # 脱敏：隐藏 user:pass@
    echo "$v = $(echo "$val" | sed -E 's#//[^/@]*@#//***:***@#')"
  fi
done

echo
echo "=== C. 代理出口连通性（状态码） ==="
for u in https://www.scnet.cn/ \
         https://gitlab.scnet.cn:9002/ \
         https://developer.sourcefind.cn/ \
         https://huggingface.co/ \
         https://pypi.tuna.tsinghua.edu.cn/simple/ ; do
  code=$(curl -sS -m 20 -o /dev/null -w '%{http_code}' "$u" 2>/dev/null || echo "ERR")
  echo "$code  $u"
done

echo
echo "=== D. torch / DCU 运行时 ==="
python3 -c "import torch;print(torch.__version__,'| hip',torch.version.hip,'| cuda_avail',torch.cuda.is_available(),'| ndev',torch.cuda.device_count())" 2>&1 | tail -3

echo
echo "=== E. 不经代理直连 modelscope（应超时/不可达） ==="
env -u http_proxy -u https_proxy -u ftp_proxy curl -sS -m 8 -o /dev/null -w 'direct modelscope %{http_code}\n' https://www.modelscope.cn/ 2>&1 | tail -2

echo
echo "=== F. 找 network_egress_investigation.md ==="
find /root/private_data/workspace -maxdepth 3 -name 'network_egress_investigation.md' 2>/dev/null
ls -l /root/private_data/workspace/reports/ 2>&1 | head -20
echo "=== done ==="
