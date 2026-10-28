#!/usr/bin/env bash
set -uo pipefail
WS=/root/private_data/workspace/robo
cd "$WS"
export PYTHONPATH="$WS/envs/pylibs_core:$WS/third_party/upstream_eval/FlagEvalMM:$WS/scripts:${PYTHONPATH:-}"
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

echo "=== py_compile ==="
python3 -m py_compile submission/bench_public/common.py submission/bench_public/datasets.py \
  submission/bench_public/run_cv_bench.py && echo COMPILE_OK

echo "=== 两种提示词样例 ==="
python3 - <<'PY'
import sys
sys.path.insert(0, "/root/private_data/workspace/robo/submission/bench_public")
import common as C
import datasets as D
print("--- flagevalmm ---")
print(C.prompt_cv_bench_flag_eval_mm("How many organs are in the image?", ["3", "2", "1", "0"]))
print("--- dataset ---")
print(C.prompt_cv_bench_dataset(
    "How many organs are in the image? Select from the following choices.\n"
    "(A) 3\n(B) 2\n(C) 1\n(D) 0"))
print("--- 数据集字段抽样 ---")
n = 0
for ann, _ in D.iter_cvbench():
    if n >= 2:
        break
    print("idx=%s prompt_official=%r" % (ann["question_id"], (ann.get("prompt_official") or "")[:80]))
    n += 1
PY
echo "=== 进程/进度 ==="
echo "3d chunks 完成: $(grep -c '完成' logs/ts_score_3d.log)"
echo "upscale: $(grep -c 'protocol=' logs/exp_cvbench_upscale.log) 个协议开始"
tail -n 3 logs/exp_cvbench_upscale.log
