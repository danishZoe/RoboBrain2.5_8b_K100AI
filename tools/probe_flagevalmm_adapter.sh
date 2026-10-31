#!/usr/bin/env bash
# 1) FlagEvalMM 里的 RoboBrain2.5 适配器（产生论文分数的原始协议）
# 2) point_utils / common_types / pre_process 的关键函数
# 3) robo_spatial_home 任务配置
# 4) RefSpatial / TraceSpatial 图片实际尺寸
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
FE=third_party/upstream_eval/FlagEvalMM

echo "===== A) FlagEvalMM 里 RoboBrain 相关文件 ====="
grep -ril "robobrain" "$FE" --include='*.py' --include='*.md' --include='*.yaml' --include='*.yml' --include='*.json' 2>/dev/null | head -n 40

echo
echo "===== B) point_utils.text2pts_norm1k / calculate_mask_score ====="
sed -n '1,140p' "$FE/flagevalmm/evaluator/point_utils.py" 2>/dev/null

echo
echo "===== C) common_types.evaluate_multiple_choice ====="
grep -n "def evaluate_multiple_choice" -A 60 "$FE/flagevalmm/evaluator/common_types.py" 2>/dev/null | head -n 80

echo
echo "===== D) pre_process.normalize_string ====="
grep -n "def normalize_string" -A 25 "$FE/flagevalmm/evaluator/pre_process.py" 2>/dev/null | head -n 40

echo
echo "===== E) robo_spatial_home 任务目录 ====="
ls -la "$FE/tasks/robo_spatial_home/" 2>/dev/null
for f in "$FE/tasks/robo_spatial_home/"*.py; do echo "--- $f ---"; cat "$f"; done 2>/dev/null | head -n 80

echo
echo "===== F) RefSpatial 图片尺寸抽样 ====="
python3 - <<'PY'
import json, os
from PIL import Image
B = "third_party/benchmarks/RefSpatial-Bench"
for s in ("Location", "Placement", "Unseen"):
    q = json.load(open(os.path.join(B, s, "question.json"), encoding="utf-8"))
    sizes = set()
    for it in q[:12]:
        p = os.path.join(B, s, it["rgb_path"])
        try:
            with Image.open(p) as im:
                sizes.add(im.size)
        except Exception as e:
            sizes.add(("ERR", str(e)[:40]))
    print(f"{s}: n={len(q)} 前12张尺寸集合={sorted(sizes, key=str)}")
    mp = os.path.join(B, s, q[0]["mask_path"])
    with Image.open(mp) as im:
        print(f"   mask[0] size={im.size} mode={im.mode}")
PY

echo
echo "===== G) TraceSpatial 图片/深度尺寸抽样 ====="
python3 - <<'PY'
import json
from PIL import Image
B = "third_party/benchmarks/TraceSpatial-Bench"
d = json.load(open(f"{B}/trajectory_dataset.json", encoding="utf-8"))
print("n =", len(d))
for it in d[:5]:
    ip = f"{B}/raw_data/{it['image_path']}"
    dp = f"{B}/raw_data/{it['gt_depth_path']}"
    with Image.open(ip) as im:
        s1 = (im.size, im.mode)
    with Image.open(dp) as im:
        s2 = (im.size, im.mode)
    print(f"id={it['id']} image={s1} depth={s2}")
PY
