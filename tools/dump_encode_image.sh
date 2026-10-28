#!/usr/bin/env bash
set -uo pipefail
WS=/root/private_data/workspace/robo
FE="$WS/third_party/upstream_eval/FlagEvalMM"

echo "########## encode_image 实现 ##########"
grep -n "def encode_image" -A 45 "$FE/flagevalmm/prompt/prompt_tools.py" 2>/dev/null
echo "--- resize_image / smart_resize 相关 ---"
grep -n "max_long_side\|min_short_side\|def resize\|smart_resize\|def encode" "$FE/flagevalmm/prompt/prompt_tools.py" 2>/dev/null | head -40
echo "--- FlagEvalMM 内 Qwen 系 model_config 的长边设置 ---"
grep -rn "max_long_side" "$FE/model_zoo/" 2>/dev/null | head -20
grep -rn "min_short_side\|max_long_side" "$FE/configs/" "$FE/exp_configs/" 2>/dev/null | head -20
find "$FE" -name "*.py" -path "*config*" 2>/dev/null | head -20

echo
echo "########## qwen_vl_utils 是否可用 + smart_resize 默认 ##########"
python3 -c "
import sys
sys.path.insert(0, '$WS/envs/pylibs_core')
try:
    import qwen_vl_utils
    print('qwen_vl_utils OK', getattr(qwen_vl_utils,'__file__',None))
    import inspect
    src = inspect.getsource(qwen_vl_utils.vision_process.fetch_image)
    print(src[:3000])
except Exception as e:
    print('ERR', repr(e))
" 2>&1 | head -60
