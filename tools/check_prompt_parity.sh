#!/usr/bin/env bash
# 核对：服务侧使用的任务提示词 / 解析正则，是否与官方 upstream 逐字一致
cd /root/private_data/workspace/robo || exit 1

echo "=== 1. 官方 upstream 提示词模板在远端出现的位置 ==="
for pat in "Please provide its 2D coordinates" "end-effector-centric" "bounding box coordinate of the region"; do
  echo "--- $pat ---"
  grep -rn --include='*.py' --include='*.sh' --include='*.json' -F "$pat" . 2>/dev/null | head -6
done

echo
echo "=== 2. 远端解析正则实现 ==="
grep -rn --include='*.py' -E "re\.(findall|search|compile)" scripts/ submission/scripts/ 2>/dev/null | head -20

echo
echo "=== 3. upstream demo 资产在远端的落位与哈希 ==="
sha256sum samples/images/official_*.jpg 2>/dev/null

echo
echo "=== 4. upstream README 里 Case 7 (navigation2.jpg) 是否存在 ==="
ls -la samples/images/ | grep -i navigation
echo "--- upstream 仓库实际资产 ---"
ls -la third_party/RoboBrain2.5/assets/demo/ 2>/dev/null || ls -la /root/private_data/workspace/robo/third_party/ 2>/dev/null | head

echo
echo "=== 5. 上游 inference.py 是否已在工作目录内 ==="
find . -name 'inference.py' -not -path './envs/*' 2>/dev/null | head
