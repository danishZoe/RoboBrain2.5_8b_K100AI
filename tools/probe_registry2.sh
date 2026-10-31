#!/usr/bin/env bash
# 1) 当前容器自身的镜像/运行时元数据
# 2) 从镜像仓库取 sf_b075 SGLang 镜像 manifest/config labels
set -o pipefail
W=/root/private_data/workspace/robo
# shellcheck disable=SC1091
source "$W/tools/env_net.sh" >/dev/null 2>&1

echo "===== 1) 当前容器元数据 ====="
hostname
echo "--- env 中的镜像/运行时相关变量"
env | grep -iE 'image|dtk|torch|scnet|dcu|registry' | sed -E 's/(PASS|TOKEN|KEY|SECRET)=.*/\1=[REDACTED]/I' | head -40
echo "--- /etc 下的镜像信息文件"
ls -la /etc/*version* /etc/image* /etc/*release* 2>/dev/null | head
for f in /etc/os-release /etc/image-version /etc/dtk-version /opt/dtk/version; do
  [ -e "$f" ] && { echo "--- $f"; head -20 "$f"; }
done
echo "--- cgroup/hostname"
cat /etc/hostname
head -3 /proc/1/cgroup 2>/dev/null

echo
echo "===== 2) registry manifest 探测 ====="
REPO="aihub/dcu/jsyadmin/sglang0.5.18-ubuntu22.04-dtk2604-py3.10-sf_b075-minimax-h3"
for host in "appstore.scnet.cn:5000" "appstore.scnet.cn"; do
  for scheme in https http; do
    url="$scheme://$host/v2/$REPO/tags/list"
    echo "--- $url"
    curl -sS -m 25 -o "$W/tmp/reg_tags.json" -w 'http=%{http_code}\n' "$url" 2>&1 | head -3
    head -c 500 "$W/tmp/reg_tags.json" 2>/dev/null
    echo
  done
done

echo
echo "===== 3) v2 root ====="
curl -sS -m 25 -o "$W/tmp/reg_root.txt" -w 'http=%{http_code}\n' "https://appstore.scnet.cn:5000/v2/" 2>&1 | head -3
head -c 400 "$W/tmp/reg_root.txt" 2>/dev/null
echo
