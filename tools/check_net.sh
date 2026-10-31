#!/usr/bin/env bash
# SCNet 容器联网状态体检（不打印任何代理口令）
set -o pipefail

echo "=== 1. 平台代理文件探测 ==="
for p in "$HOME/.ai_user_info/ai_proxy" \
         /root/.ai_user_info/ai_proxy \
         /public/home/scnjz2s4uc/.ai_user_info/ai_proxy \
         /root/private_data/.ai_user_info/ai_proxy ; do
  if [ -f "$p" ]; then echo "FOUND   $p"; else echo "missing $p"; fi
done
for d in "$HOME/.ai_user_info" /public/home/scnjz2s4uc/.ai_user_info ; do
  [ -d "$d" ] && echo "DIR     $d" || echo "no dir  $d"
done

echo
echo "=== 2. 当前 shell 的代理变量（仅键名） ==="
env | grep -iE '^(http|https|ftp|no)_proxy' | cut -d= -f1 || true
echo "(以上为空即无代理变量)"

echo
echo "=== 3. 登录 shell 里的代理变量（仅键名） ==="
bash -lc 'env | grep -iE "^(http|https|ftp|no)_proxy" | cut -d= -f1' 2>/dev/null || true
echo "(以上为空即无代理变量)"

echo
echo "=== 4. 平台内网出口代理端口探测 ==="
for hp in 10.15.100.43:3120 10.15.100.43:8080 169.254.20.10:3120 ; do
  h=${hp%%:*}; p=${hp##*:}
  if timeout 5 bash -c "exec 3<>/dev/tcp/$h/$p" 2>/dev/null; then
    echo "$hp  OPEN"
  else
    echo "$hp  closed/filtered"
  fi
done

echo
echo "=== 5. 直连公网（不经代理） ==="
for u in https://www.scnet.cn/ https://www.modelscope.cn/ https://huggingface.co/ \
         https://pypi.tuna.tsinghua.edu.cn/simple/ https://gitlab.scnet.cn:9002/ \
         https://developer.sourcefind.cn/ ; do
  code=$(curl -sS -m 15 -o /dev/null -w '%{http_code}' "$u" 2>/dev/null)
  echo "$u -> ${code:-FAIL}"
done

echo
echo "=== 6. DNS 解析 ==="
for d in www.scnet.cn gitlab.scnet.cn www.modelscope.cn ; do
  echo -n "$d -> "; getent hosts "$d" 2>/dev/null | head -1 || echo "no resolve"
done
