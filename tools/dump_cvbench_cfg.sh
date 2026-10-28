#!/usr/bin/env bash
set -uo pipefail
WS=/root/private_data/workspace/robo
FE="$WS/third_party/upstream_eval/FlagEvalMM"

echo "########## 1. cv_bench_test.py ##########"
cat "$FE/tasks/cv_bench/cv_bench_test.py" 2>/dev/null

echo
echo "########## 2. 全仓 yaml/yml 文件 ##########"
find "$FE" -maxdepth 4 \( -name "*.yaml" -o -name "*.yml" \) 2>/dev/null | head -60

echo
echo "########## 3. 含 cv_bench 的配置文件 ##########"
grep -rl "cv_bench" "$FE" --include="*.yaml" --include="*.yml" --include="*.json" 2>/dev/null | head -30

echo
echo "########## 4. 含 robobrain 的配置文件 ##########"
grep -rl "robobrain" "$FE" --include="*.yaml" --include="*.yml" --include="*.json" 2>/dev/null | head -30

echo
echo "########## 5. HttpClient 的 max_image_size / thinking 处理 ##########"
grep -rn "max_image_size\|min_short_side\|max_long_side" "$FE/flagevalmm/models/" 2>/dev/null | head -30
grep -rn "def build_message" -A 40 "$FE/flagevalmm/models/http_client.py" 2>/dev/null | head -80

echo
echo "########## 6. thinking / reasoning 相关 ##########"
grep -rn "thinking\|reasoning" "$FE/flagevalmm/models/http_client.py" 2>/dev/null | head -30
