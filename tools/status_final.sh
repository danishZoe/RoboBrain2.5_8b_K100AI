#!/usr/bin/env bash
set -o pipefail
W=/root/private_data/workspace/robo

echo "===== 12 轮压测的请求总量与口径 ====="
python3 - <<'PY'
import json, os
B="/root/private_data/workspace/robo/submission/bench"
tot=0; rows=[]
for rf in ("round1.jsonl","round2.jsonl"):
    p=os.path.join(B,rf)
    if not os.path.exists(p): continue
    for line in open(p,encoding="utf-8"):
        line=line.strip()
        if not line: continue
        d=json.loads(line)
        c=d.get("completed"); tot+=int(c or 0)
        rows.append((rf,d.get("random_input_len"),d.get("max_concurrency"),c,
                     round(d.get("output_throughput") or 0,2),
                     round(d.get("mean_ttft_ms") or 0,1),
                     d.get("total_output_tokens"),d.get("total_output_tokens_retokenized")))
print(f"{'轮次':<9}{'输入':>7}{'并发':>5}{'完成':>6}{'tok/s':>9}{'TTFT_ms':>11}{'生成tok':>10}{'重分词':>9}")
for r in rows:
    print(f"{r[0]:<9}{r[1]:>7}{r[2]:>5}{r[3]:>6}{r[4]:>9}{r[5]:>11}{r[6]:>10}{r[7]:>9}")
print("总请求数 =", tot, " 总生成token =", sum(int(r[6] or 0) for r in rows))
PY

echo
echo "===== pipeline_phase6.sh 在做什么 ====="
tail -n 30 "$W/logs/pipeline_phase6.log" 2>/dev/null || ls -l "$W/logs/" | grep -i phase6

echo
echo "===== clean_repro 目录 ====="
ls -l "$W/logs/clean_repro/" 2>/dev/null | head -20

echo
echo "===== 各阶段产物新鲜度 ====="
for p in outputs/phase4_parse_coverage/parse_coverage.json \
         outputs/phase5_smoke/smoke_results.json \
         outputs/phase5_ab_noneager/noneager_20261027_194104/probe/probe_result.json \
         outputs/phase5_token_accounting/result.json \
         submission/verify_result.json \
         submission/bench/summary.md; do
  printf '%-62s ' "$p"
  [ -e "$W/$p" ] && stat -c '%y  %s B' "$W/$p" || echo "MISSING"
done

echo
echo "===== verify_result.json 状态 ====="
python3 -c "
import json;d=json.load(open('$W/submission/verify_result.json'))
print('status =',d.get('status'),'| passed =',d.get('passed'))
print('reason =',(d.get('reason') or '')[:150])
"

echo
echo "===== parse_coverage 实验（§? ）是什么 ====="
head -c 1200 "$W/outputs/phase4_parse_coverage/parse_coverage.json" 2>/dev/null
