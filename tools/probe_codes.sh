#!/usr/bin/env bash
source /root/private_data/workspace/robo/tools/env_net.sh >/dev/null 2>&1
for u in https://www.modelscope.cn/ \
         https://www.scnet.cn/ \
         https://developer.sourcefind.cn/ \
         https://huggingface.co/ \
         https://pypi.tuna.tsinghua.edu.cn/simple/ \
         https://gitlab.scnet.cn:9002/ ; do
  code=$(curl -sS -m 20 -o /dev/null -w '%{http_code}' "$u" 2>/dev/null)
  [ -z "$code" ] && code=ERR
  echo "$code  $u"
done
