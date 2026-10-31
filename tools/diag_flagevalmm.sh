#!/usr/bin/env bash
# 诊断 FlagEvalMM 判分链路的导入失败
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONPATH="$WS/envs/pylibs_core:$WS/third_party/upstream_eval/FlagEvalMM:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

echo "===== 1) 逐级 import ====="
python3 - <<'PY'
import importlib, traceback
for m in ("flagevalmm",
          "flagevalmm.evaluator",
          "flagevalmm.evaluator.pre_process",
          "flagevalmm.evaluator.common_types",
          "flagevalmm.evaluator.point_utils"):
    try:
        importlib.import_module(m)
        print("  OK  ", m)
    except Exception as e:
        print("  FAIL", m, "->", type(e).__name__, str(e)[:200])
        tb = traceback.format_exc().strip().splitlines()
        for line in tb[-6:]:
            print("        ", line)
        break
PY

echo
echo "===== 2) flagevalmm/evaluator/__init__.py 的 import ====="
head -n 25 third_party/upstream_eval/FlagEvalMM/flagevalmm/evaluator/__init__.py 2>/dev/null

echo
echo "===== 3) 直接按文件加载官方 evaluate.py（绕开包 __init__）====="
python3 - <<'PY'
import importlib.util, traceback
try:
    spec = importlib.util.spec_from_file_location(
        "cv_bench_evaluate",
        "/root/private_data/workspace/robo/third_party/upstream_eval/FlagEvalMM/tasks/cv_bench/evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("  OK   官方 tasks/cv_bench/evaluate.py 加载成功")
    print("  get_result:", mod.get_result)
except Exception as e:
    print("  FAIL", type(e).__name__, str(e)[:300])
    print(traceback.format_exc()[-1500:])
PY

echo
echo "===== 4) 官方 common_types 直接按文件加载 ====="
python3 - <<'PY'
import importlib.util, traceback, sys
sys.path.insert(0, "/root/private_data/workspace/robo/third_party/upstream_eval/FlagEvalMM")
try:
    spec = importlib.util.spec_from_file_location(
        "fe_common_types",
        "/root/private_data/workspace/robo/third_party/upstream_eval/FlagEvalMM/flagevalmm/evaluator/common_types.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("  OK   common_types 加载成功:", mod.evaluate_multiple_choice)
except Exception as e:
    print("  FAIL", type(e).__name__, str(e)[:300])
    print(traceback.format_exc()[-1200:])
PY
