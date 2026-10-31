#!/usr/bin/env bash
# 探查 TraceSpatial-Bench 的 trajectory_dataset.json / data/ 结构，以及 open3d 可用性
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1

echo "===== 1) open3d / 依赖可用性 ====="
python3 - <<'PY'
import importlib.util as u
for m in ("open3d", "plyfile", "trimesh", "opencv", "cv2", "skimage", "scipy", "numpy"):
    try:
        print(f"{m:10s} -> {u.find_spec(m) is not None}")
    except Exception as e:
        print(f"{m:10s} -> ERR {e}")
import numpy, scipy, PIL
print("numpy", numpy.__version__, "scipy", scipy.__version__, "PIL", PIL.__version__)
PY

echo
echo "===== 2) TraceSpatial-Bench 顶层 ====="
cd third_party/benchmarks/TraceSpatial-Bench || exit 1
ls -la | head -n 20
echo "--- data/ ---"
ls -la data/ 2>/dev/null | head -n 20
echo "--- raw_data/0 ---"
ls -la raw_data/0/ 2>/dev/null | head -n 20

echo
echo "===== 3) trajectory_dataset.json 结构 ====="
python3 - <<'PY'
import json, os
p = "trajectory_dataset.json"
print("size(B):", os.path.getsize(p) if os.path.exists(p) else "MISSING")
if os.path.exists(p):
    with open(p) as f:
        d = json.load(f)
    print("type:", type(d).__name__)
    if isinstance(d, dict):
        print("keys:", list(d.keys())[:20])
        for k in list(d.keys())[:2]:
            v = d[k]
            print(f"  [{k}] type={type(v).__name__}",
                  (f"len={len(v)}" if hasattr(v, "__len__") else ""))
            if isinstance(v, list) and v:
                print("   first item:", json.dumps(v[0], ensure_ascii=False)[:1200])
            elif isinstance(v, dict):
                print("   sample:", json.dumps(v, ensure_ascii=False)[:1200])
    elif isinstance(d, list):
        print("len:", len(d))
        if d:
            print("item0 keys:", list(d[0].keys()) if isinstance(d[0], dict) else type(d[0]))
            print("item0:", json.dumps(d[0], ensure_ascii=False)[:1500])
PY

echo
echo "===== 4) 上游参考预测（2D/3D）样例 ====="
for f in Model_Prediction_TraceSpatial_2D.jsonl Model_Prediction_TraceSpatial_3D.jsonl; do
  echo "--- $f  lines=$(wc -l < "$f") ---"
  head -c 900 "$f"; echo
done

echo
echo "===== 5) requirements.txt ====="
cat requirements.txt 2>/dev/null | head -n 40
