#!/usr/bin/env bash
# 压测结束后刷新提交包的三个产物，使其与最终配置（MAX_LEN=32768）对齐：
#   env_check.sh -> env.log
#   smoke.sh     -> smoke.log (+ smoke_results.json)
#   verify.py    -> verify_result.json (+ verify_evidence.json)
# 要求：服务已在 127.0.0.1:30000 上以最终配置运行。
set -uo pipefail
cd /root/private_data/workspace/robo/submission
source ./env.sh

echo "===== service check ====="
curl -sf --noproxy '*' "http://127.0.0.1:$PORT/health" -o /dev/null -w 'health=%{http_code}\n' || {
  echo "service not up on $PORT"; exit 1; }
curl -sf --noproxy '*' "http://127.0.0.1:$PORT/v1/models" | python3 -c "
import json,sys
d=json.load(sys.stdin)['data'][0]
print('served=',d['id'],'max_model_len=',d.get('max_model_len'))
"

echo
echo "===== 1) env_check.sh -> env.log ====="
mkdir -p "$WORK_DIR/tmp"
bash env_check.sh > "$WORK_DIR/tmp/env_check_out.txt" 2>&1
rc=$?
tail -5 "$WORK_DIR/tmp/env_check_out.txt"
echo "exit=$rc"
[ -s env.log ] && echo "env.log = $(wc -c < env.log) B" || echo "env.log MISSING"

echo
echo "===== 2) smoke.sh -> smoke.log ====="
bash smoke.sh > "$WORK_DIR/tmp/smoke_out.txt" 2>&1
rc=$?
tail -8 "$WORK_DIR/tmp/smoke_out.txt"
echo "exit=$rc"
[ -s smoke.log ] && echo "smoke.log = $(wc -c < smoke.log) B" || echo "smoke.log MISSING"

echo
echo "===== 3) verify.py --selfcheck ====="
python3 verify.py --selfcheck; echo "exit=$?"

echo
echo "===== 4) verify.py (full) ====="
python3 verify.py; rc=$?
echo "exit=$rc (2 = BLOCKED，预期：官方基线未下发)"
python3 -c "
import json;d=json.load(open('verify_result.json'))
print('status=',d.get('status'),'passed=',d.get('passed'))
im=d.get('internal_metric') or {}
print('internal:',im.get('identical'),'/',im.get('total'))
"

echo
echo "===== artifacts ====="
ls -l env.log smoke.log verify_result.json bench/summary.md 2>&1
