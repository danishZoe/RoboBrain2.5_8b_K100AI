#!/usr/bin/env bash
# 上线前自检：语法编译、模块导入、提示词一致性、数据装载抽样
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONPATH="$WS/envs/pylibs_core:$WS/third_party/upstream_eval/FlagEvalMM:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

echo "===== 1) py_compile ====="
python3 -m py_compile submission/bench_public/*.py && echo "  语法 OK"

echo
echo "===== 2) 模块导入 + 提示词自检 ====="
cd "$WS/submission/bench_public" && python3 - <<'PY'
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "../scripts")
import common as C, datasets as D, engine as E
print("  common/datasets/engine 导入 OK")
print("  prompt parity:", C.selfcheck_prompt_parity())
print()
print("  --- RefSpatial 官方 prompt (逐字) ---")
print("  ", repr(C.prompt_refspatial_official("Please point out the orange box.")))
print("  --- RefSpatial 上游 prompt ---")
print("  ", repr(C.prompt_upstream(C.strip_one_trailing_period("Please point out the orange box."), "pointing")))
print("  --- TraceSpatial 官方 3D prompt ---")
print("  ", repr(C.prompt_tracespatial_official("3D", "the pillow", "the stool", "pick up the pillow")))
print("  --- CV-Bench prompt ---")
print("  ", repr(C.prompt_cv_bench_flag_eval_mm("How many organs are in the image?", ["3", "2", "1", "0"])))
PY

echo
echo "===== 3) 数据装载抽样 ====="
python3 - <<'PY'
import sys, json
sys.path.insert(0, ".")
sys.path.insert(0, "../scripts")
import datasets as D, common as C
s = D.load_refspatial("Location")
print("  RefSpatial Location n=%d, 首题 image=%s" % (len(s), s[0]["image_paths"][0]))
print("  首题 fields:", json.dumps(s[0]["fields"], ensure_ascii=False)[:200])
t = D.load_tracespatial("3D")
print("  TraceSpatial n=%d, 首题 image=%s" % (len(t), t[0]["image_paths"][0]))
print("  首题 fields keys:", list(t[0]["fields"].keys()))
n = 0
for ann, data in D.iter_cvbench():
    n += 1
    if n == 1:
        print("  CV-Bench 首题 ann:", json.dumps(ann, ensure_ascii=False)[:300])
        print("  CV-Bench 首题 image bytes:", len(data))
    if n >= 5:
        break
print("  CV-Bench 抽样条数:", n)
PY

echo
echo "===== 4) open3d / torch_fidelity ====="
python3 -c "import importlib.util as u; print('  open3d:', u.find_spec('open3d') is not None); print('  torch_fidelity:', u.find_spec('torch_fidelity') is not None)"

echo
echo "===== 5) 服务状态 ====="
(ss -ltnp 2>/dev/null | grep -E ':(8000|30000)') || echo "  无监听"
tail -n 5 logs/serve_eager_20261027-213945.log 2>/dev/null
