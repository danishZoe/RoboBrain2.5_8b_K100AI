#!/usr/bin/env bash
set -o pipefail
W=/root/private_data/workspace/robo

echo "===== 上游 draw_on_image 如何使用坐标（判定尺度的关键） ====="
sed -n "$(grep -n 'def draw_on_image' "$W/third_party/RoboBrain2.5/inference.py" | cut -d: -f1),+55p" "$W/third_party/RoboBrain2.5/inference.py"

echo
echo "===== 上游 inference.py 返回结构（answer 字段如何组织） ====="
sed -n "$(grep -n '# Return unified format' "$W/third_party/RoboBrain2.5/inference.py" | cut -d: -f1),+40p" "$W/third_party/RoboBrain2.5/inference.py"

echo
echo "===== 我们样例图的真实尺寸 ====="
python3 - <<'PY'
import os
from PIL import Image
D = "/root/private_data/workspace/robo/samples/images"
for f in sorted(os.listdir(D)):
    if f.lower().endswith((".jpg", ".jpeg", ".png")):
        p = os.path.join(D, f)
        try:
            im = Image.open(p)
            print("  %-32s %s" % (f, im.size))
        except Exception as e:
            print("  %-32s ERR %s" % (f, e))
PY

echo
echo "===== 我们的 answer_adapter.py 做了什么换算 ====="
grep -nE "1000|normal|scale|resiz|width|height|/ 1000|0-1" "$W/submission/scripts/answer_adapter.py" | head -25
