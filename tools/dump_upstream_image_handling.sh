#!/usr/bin/env bash
set -uo pipefail
WS=/root/private_data/workspace/robo

echo "########## 1. 上游 RoboBrain2.5 inference.py 的图像处理 ##########"
find "$WS/third_party" -maxdepth 3 -name "inference.py" 2>/dev/null
for f in $(find "$WS/third_party" -maxdepth 3 -name "inference.py" 2>/dev/null | head -3); do
  echo "--- $f"
  grep -n "resize\|max_pixels\|min_pixels\|smart_resize\|Image.open\|image_size\|encode\|base64\|processor\|process_vision\|28\b" "$f" | head -40
done

echo
echo "########## 2. 上游仓库里与图像分辨率相关的常量 ##########"
grep -rn "max_pixels\|min_pixels\|smart_resize\|MAX_PIXELS\|image_max" "$WS/third_party/RoboBrain2.5" 2>/dev/null | head -30

echo
echo "########## 3. 模型 README 里的推理示例（图像处理部分）##########"
M=/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV
grep -n "resize\|max_pixels\|min_pixels\|smart_resize\|image\|Image" "$M/README.md" 2>/dev/null | head -40

echo
echo "########## 4. readme.ipynb 里的图像处理 ##########"
python3 - <<'PY'
import json, os, re
M = "/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV"
p = os.path.join(M, "readme.ipynb")
try:
    nb = json.load(open(p, encoding="utf-8"))
except Exception as e:
    print("ERR", e); raise SystemExit
for cell in nb.get("cells", []):
    src = "".join(cell.get("source", []))
    if any(k in src for k in ("resize", "max_pixels", "min_pixels", "smart_resize", "Image", "image")):
        print("-----")
        print(src[:1200])
PY
