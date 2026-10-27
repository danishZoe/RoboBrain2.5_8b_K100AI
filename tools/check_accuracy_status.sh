#!/usr/bin/env bash
set -o pipefail
W=/root/private_data/workspace/robo

echo "===== 时间 ====="; date

echo
echo "===== 官方精度材料是否到位 ====="
for p in "$W/submission/baselines" "$W/submission/data"; do
  if [ -d "$p" ]; then echo "目录存在: $p"; ls -la "$p"; else echo "目录不存在: $p"; fi
done
find "$W" -name 'BAAI__RoboBrain2.5-8B-NV.json' 2>/dev/null | sed 's/^/  found: /'
find "$W" -name 'judge.py' -not -path '*/tmp/*' 2>/dev/null | sed 's/^/  judge: /'

echo
echo "===== submission/verify_result.json（当前状态） ====="
python3 - <<'PY'
import json, os
p = "/root/private_data/workspace/robo/submission/verify_result.json"
d = json.load(open(p, encoding="utf-8"))
print("generated_at =", d.get("generated_at"))
print("status       =", d.get("status"), "| passed =", d.get("passed"))
print("reason       =", (d.get("reason") or "")[:220])
im = d.get("internal_metric")
if im:
    print("内部指标:", im.get("name"))
    print("  一致/总数 =", im.get("identical"), "/", im.get("total"))
    for k, v in (im.get("detail") or {}).items():
        if isinstance(v, dict) and "identical" in v:
            print(f"   - {k:<22} identical={v['identical']}")
PY

echo
echo "===== 规范化 serving 探针：6 样例逐坐标比对（精度侧最接近官方口径的内部证据） ====="
for f in "$W/outputs/phase5_probe_canonical/probe_result.json" \
         "$W/outputs/phase5_ab_noneager/noneager_20261027_194104/probe/probe_result.json"; do
  [ -e "$f" ] || continue
  echo "--- $f"
  python3 - "$f" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
res = d.get("results") or d.get("samples") or []
if not res and isinstance(d, dict):
    for v in d.values():
        if isinstance(v, list) and v and isinstance(v[0], dict) and "sample_id" in v[0]:
            res = v
same = 0
for r in res:
    if not isinstance(r, dict): continue
    sid = r.get("sample_id") or r.get("id") or "?"
    ok = r.get("identical")
    if ok is None:
        b = r.get("baseline"); s = r.get("serving")
        ok = (b == s) if (b is not None and s is not None) else None
    if ok: same += 1
    print(f"   {sid:<22} identical={ok}")
print(f"   -> 完全一致 {same}/{len(res)}")
PY
done

echo
echo "===== agreement.md（与 transformers 直跑基线的逐坐标差异） ====="
head -40 "$W/outputs/phase5_probe_canonical/agreement.md" 2>/dev/null

echo
echo "===== clean_repro 进度 ====="
ls -l "$W/logs/clean_repro/" 2>/dev/null
tail -n 20 "$W/logs/clean_repro_stdout.log" 2>/dev/null
tail -n 12 "$W/logs/pipeline_phase6.log" 2>/dev/null
