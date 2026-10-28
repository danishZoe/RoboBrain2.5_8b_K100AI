#!/usr/bin/env bash
# 诊断：CV-Bench 精度差距的输入侧根因
#   1) 上游 RoboBrain2.5 inference.py 的图像预处理（max_pixels 等）
#   2) FlagEvalMM tasks/cv_bench/process.py 如何准备图像与标注
#   3) 模型侧 processor 的 min_pixels/max_pixels
#   4) upstream_eval 目录里是否还有 RoboBrain2.5 官方评测脚本
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1

echo "########## 0. upstream_eval 目录清单 ##########"
ls third_party/upstream_eval/

echo
echo "########## 1. 上游 RoboBrain2.5 inference.py 的图像预处理 ##########"
find third_party -name 'inference.py' -path '*RoboBrain*' | head
for f in $(find third_party -name 'inference.py' -path '*RoboBrain*' | head -2); do
  echo "--- $f ---"
  grep -n 'max_pixels\|min_pixels\|fetch_image\|process_vision_info\|smart_resize\|qwen_vl_utils\|MAX_PIXELS\|resize' "$f" | head -30
done

echo
echo "########## 2. FlagEvalMM tasks/cv_bench/process.py ##########"
sed -n '1,140p' third_party/upstream_eval/FlagEvalMM/tasks/cv_bench/process.py

echo
echo "########## 3. 模型 processor 的像素预算 ##########"
python3 - <<'PY'
import json
p = "/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV/preprocessor_config.json"
try:
    d = json.load(open(p, encoding="utf-8"))
    for k in ("min_pixels", "max_pixels", "patch_size", "merge_size", "size",
              "image_processor_type", "do_resize", "do_rescale"):
        if k in d:
            print("  %-22s = %s" % (k, d[k]))
    ip = d.get("image_processor") or {}
    for k in ("min_pixels", "max_pixels", "patch_size", "merge_size", "size"):
        if k in ip:
            print("  image_processor.%-12s = %s" % (k, ip[k]))
except Exception as e:
    print("  读取失败:", e)
PY

echo
echo "########## 4. 当前进度 ##########"
date
ps -ef | grep -E 'summarize_acc|run_tracespatial|stability|rescore' | grep -v grep | tr -s ' ' | cut -d' ' -f2,9,10,11,12,13
tail -3 logs/public_bench_baseline-eager/tracespatial.log
