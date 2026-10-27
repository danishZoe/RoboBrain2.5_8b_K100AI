#!/usr/bin/env bash
# SCNet 容器统一环境：DCU 运行时 + 平台出口代理
#
# 用法：在任何需要联网 / 需要 torch 的脚本或交互 shell 里先 source 它
#     source /root/private_data/workspace/robo/tools/env_net.sh
#
# 说明：
#   - 平台代理文件由平台生成，本容器里位于 /public/home/scnjz2s4uc/.ai_user_info/ai_proxy
#     （容器内 /root/private_data 挂载的就是该家目录，所以还有 /root/private_data/.ai_user_info/ai_proxy）。
#     指南里的 ~/.ai_user_info/ai_proxy 在本容器中不存在，因为 root 的 $HOME 是 /root 而不是家目录挂载点。
#   - 代理文件含明文口令，**不要**把内容写进任何脚本、日志、报告或提交物；只 source 一次即可。
#   - /opt/dtk/env.sh 内部引用了未定义变量（CMAKE_PREFIX_PATH 等），在 set -u 下会打死脚本，
#     所以先临时关闭 nounset 再 source。

__robo_nounset_was_on=0
case "$-" in *u*) __robo_nounset_was_on=1; set +u ;; esac

# 1) DCU 运行时（不 source 的话 import torch 报 libgalaxyhip.so.5 缺失）
[ -f /opt/dtk/env.sh ] && source /opt/dtk/env.sh >/dev/null 2>&1

# 2) 平台出口代理
for _p in /root/private_data/.ai_user_info/ai_proxy \
          /public/home/scnjz2s4uc/.ai_user_info/ai_proxy ; do
  if [ -f "$_p" ]; then
    . "$_p" >/dev/null 2>&1
    break
  fi
done
unset _p

[ "$__robo_nounset_was_on" = "1" ] && set -u
unset __robo_nounset_was_on
