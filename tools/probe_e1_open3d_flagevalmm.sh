#!/usr/bin/env bash
# 1) E1 确认实验各档结果汇总  2) open3d 安装状态  3) FlagEvalMM 依赖链可行性
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1

echo "===== A) E1 确认实验（non-eager）各档结果 ====="
D=outputs/phase5_ab_noneager/noneager_confirm_20261027_210402
for f in "$D"/ab_*.jsonl; do
  echo "--- $(basename "$f") ---"
  python3 - "$f" <<'PY'
import json, sys
p = sys.argv[1]
rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
if not rows:
    print("  (empty)"); raise SystemExit
dur = rows[0].get("duration", 0)
ok = sum(1 for r in rows if r.get("success", True))
gen = sum(r.get("output_tokens", 0) or 0 for r in rows)
print(f"  n={len(rows)} ok={ok} duration={dur:.2f}s "
      f"gen_tok={gen} tok/s={gen/dur if dur else 0:.2f}")
PY
done
echo "--- 基线对照（eager，来自 phase4/phase5 原始数据）---"
for f in outputs/phase4_bench/*4096_c1*.jsonl outputs/phase4_bench/*4096_c8*.jsonl \
         outputs/phase4_bench/*16384_c1*.jsonl outputs/phase4_bench/*16384_c8*.jsonl; do
  [ -f "$f" ] || continue
  echo "--- $f ---"
  python3 - "$f" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
if not rows: raise SystemExit
dur = rows[0].get("duration", 0)
gen = sum(r.get("output_tokens", 0) or 0 for r in rows)
print(f"  n={len(rows)} duration={dur:.2f}s gen_tok={gen} tok/s={gen/dur if dur else 0:.2f}")
PY
done

echo
echo "===== B) open3d ====="
ls -d envs/pylibs/open3d* 2>/dev/null || echo "  未安装"
ps -ef | grep -c '[p]ip install' | sed 's/^/  pip 进程数: /'
tail -n 30 logs/pip_open3d.log 2>/dev/null

echo
echo "===== C) FlagEvalMM 依赖链 ====="
FE=third_party/upstream_eval/FlagEvalMM
echo "--- flagevalmm/__init__.py ---"
head -n 30 "$FE/flagevalmm/__init__.py" 2>/dev/null
echo "--- common_types.py 头部 import ---"
head -n 12 "$FE/flagevalmm/evaluator/common_types.py" 2>/dev/null
echo "--- pre_process.py 头部 import ---"
head -n 12 "$FE/flagevalmm/evaluator/pre_process.py" 2>/dev/null
echo "--- 试 import（加 PYTHONPATH 指向仓库根）---"
PYTHONPATH="$WS/third_party/upstream_eval/FlagEvalMM:$WS/envs/pylibs:${PYTHONPATH:-}" python3 - <<'PY'
mods = ["flagevalmm.evaluator.common_types", "flagevalmm.evaluator.pre_process",
        "flagevalmm.evaluator.point_utils"]
for m in mods:
    try:
        __import__(m)
        print("  OK  ", m)
    except Exception as e:
        print("  FAIL", m, type(e).__name__, str(e)[:120])
PY
echo "--- 缺哪些 pip 包 ---"
python3 -c "import mmengine" 2>&1 | tail -n 1
python3 -c "import word2number" 2>&1 | tail -n 1
python3 -c "import easydict" 2>&1 | tail -n 1
python3 -c "import flask" 2>&1 | tail -n 1

echo
echo "===== D) TraceSpatial 上游参考预测文件 ====="
ls -la third_party/upstream_eval/RoboTracer/Evaluation/outputs/ 2>/dev/null | head -n 12
ls -la third_party/benchmarks/TraceSpatial-Bench/Model_Prediction_*.jsonl 2>/dev/null
