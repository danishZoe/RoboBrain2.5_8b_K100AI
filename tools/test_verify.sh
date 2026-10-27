#!/usr/bin/env bash
set -uo pipefail
cd /root/private_data/workspace/robo/submission

echo "===== 1) selfcheck ====="
python3 verify.py --selfcheck; echo "exit=$?"

echo
echo "===== 2) no baseline -> BLOCKED ====="
source ./env.sh
python3 verify.py --out /root/private_data/workspace/robo/tmp/vr_nobase.json; echo "exit=$?"
python3 -c "
import json;d=json.load(open('/root/private_data/workspace/robo/tmp/vr_nobase.json'))
print('status=',d.get('status'),'passed=',d.get('passed'))
print('reason=',d.get('reason'))
print('internal=',{k:v for k,v in (d.get('internal_metric') or {}).items() if k in ('name','identical','total')})
"

echo
echo "===== 3) synthetic baseline, subset missing -> must be BLOCKED (not PASS) ====="
mkdir -p baselines
cat > baselines/TEST__Synthetic.json <<'JSON'
{"model":"TEST/Synthetic","extra_body":{},
 "metrics":{"agreement_test":{"subset":"data/does_not_exist.jsonl","n":2,"max_tokens":16,
   "ref":100.0,"scale":100,"judge_type":"agreement","threshold":98.0}}}
JSON
MODEL_ID="TEST/Synthetic" python3 verify.py --out /root/private_data/workspace/robo/tmp/vr_nosubset.json --evidence /root/private_data/workspace/robo/tmp/vr_ev.json; echo "exit=$?"
python3 -c "
import json;d=json.load(open('/root/private_data/workspace/robo/tmp/vr_nosubset.json'))
print('status=',d.get('status'),'passed=',d.get('passed'))
print('summary=',d.get('summary'))
print('detail=',json.dumps(d.get('detail'),ensure_ascii=False))
"
rm -f baselines/TEST__Synthetic.json
rmdir baselines/data 2>/dev/null || true
echo "cleaned: $(ls baselines 2>/dev/null | wc -l) files left in baselines/"
