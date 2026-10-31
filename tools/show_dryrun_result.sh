#!/usr/bin/env bash
set -o pipefail
W=/root/private_data/workspace/robo
echo "===== 沙箱 verify_result.json（演练结论原文） ====="
cat "$W/tmp/dryrun_verify/verify_result.json" 2>/dev/null | head -c 2000
echo
echo "===== infer_kind / REF_KEYS / pick_field ====="
grep -n "REF_KEYS\s*=" "$W/submission/verify.py"
sed -n "$(grep -n 'def infer_kind' "$W/submission/verify.py" | cut -d: -f1),+30p" "$W/submission/verify.py"
echo "----- pick_field -----"
sed -n "$(grep -n 'def pick_field' "$W/submission/verify.py" | cut -d: -f1),+14p" "$W/submission/verify.py"
echo
echo "===== 沙箱 data/internal_demo.jsonl 的 reference 字段覆盖 ====="
python3 - <<'PY'
import json
p='/root/private_data/workspace/robo/tmp/dryrun_verify/data/internal_demo.jsonl'
for i,l in enumerate(open(p,encoding='utf-8')):
    r=json.loads(l)
    print(i, r.get('sample_id'), '| reference 存在:', 'reference' in r, '| keys:', sorted(r.keys()))
PY
