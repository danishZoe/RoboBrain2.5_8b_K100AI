#!/usr/bin/env bash
set -uo pipefail
WS=/root/private_data/workspace/robo
cd "$WS"
echo "=== 语法检查 ==="
bash -n tools/verify_3d_merge.sh && echo "verify_3d_merge.sh SYNTAX_OK"
python3 -m py_compile submission/bench_public/score_3d_chunked.py && echo "score_3d_chunked.py COMPILE_OK"

echo
echo "=== 把被阶段A 覆盖的垃圾产物改名留存（不删除）==="
for F in outputs/bench_public/tracespatial/official/scores_3dchunk_3D.json \
         outputs/bench_public/tracespatial/upstream/scores_3dchunk_3D.json \
         outputs/bench_public_noneager/tracespatial/official/scores_3dchunk_3D.json \
         outputs/bench_public_noneager/tracespatial/upstream/scores_3dchunk_3D.json; do
  if [ -f "$F" ]; then
    mv "$F" "$F.garbage_stageA_20261028"
    echo "  moved: $F -> $F.garbage_stageA_20261028"
  fi
done

echo
echo "=== 分片写文件的换行回归（多样本分片必须能被官方 json.loads 解析）==="
python3 - <<'PY'
import json, os, sys, tempfile
WS = "/root/private_data/workspace/robo"
sys.path.insert(0, WS + "/submission/bench_public")
import score_3d_chunked as S3
src = (WS + "/outputs/bench_public/tracespatial/official/outputs/"
       "RoboTracer_RoboBrain2.5-8B-NV_3D.jsonl")
lines = [l for l in open(src, encoding="utf-8").read().splitlines() if l.strip()][:5]
d = tempfile.mkdtemp()
p = os.path.join(d, "x.jsonl")
with open(p, "w", encoding="utf-8") as fh:
    fh.write("".join(l if l.endswith("\n") else l + "\n" for l in lines))
n = 0
for line in open(p, encoding="utf-8"):
    if line.strip():
        json.loads(line)
        n += 1
print("  写入 %d 条 → 官方 json.loads 解析成功 %d 条  ✅" % (len(lines), n))
print("  产物:", p)
PY

echo
echo "=== 96 兜底判分进度 ==="
grep -cE '完成' logs/ts_score_3d_96.log
