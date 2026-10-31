#!/usr/bin/env bash
# 等待官方压测六档跑完（轮询进程），结束后自动生成 bench/summary.md
set -uo pipefail
cd /root/private_data/workspace/robo/submission

DEADLINE=$(( $(date +%s) + 14400 ))   # 最多等 4 小时
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  if ! pgrep -f 'bash bench.sh' > /dev/null 2>&1 && ! pgrep -f 'bench_serving' > /dev/null 2>&1; then
    echo "[watcher] bench loop finished at $(date -Is)"
    break
  fi
  sleep 20
done

echo
echo "===== per-tier status ====="
for f in bench/round*.log; do
  [ -e "$f" ] || continue
  n=$(basename "$f" .log)
  if grep -qiE 'Traceback|^Error' "$f"; then st="FAILED"; else st="OK"; fi
  ot=$(grep -E 'Output token throughput' "$f" | tail -1 | awk '{print $NF}')
  cp=$(grep -E '^Successful requests' "$f" | tail -1 | awk '{print $NF}')
  fl=$(grep -E '^Failed requests' "$f" | tail -1 | awk '{print $NF}')
  printf '  %-30s %-7s ok=%-5s failed=%-5s out_tps=%-9s\n' "$n" "$st" "${cp:-?}" "${fl:-?}" "${ot:-?}"
done

echo
echo "===== generate summary ====="
bash /root/private_data/workspace/robo/tools/summarize_official_bench.sh 2>/dev/null || {
  python3 - <<'PY' | tee bench/summary.md
import json, os, glob, re
ROWS=[(4096,1024,1,8),(4096,1024,8,16),(4096,1024,64,128),(16384,1024,1,8),(16384,1024,8,16),(16384,1024,64,128)]
def load(p):
    return [json.loads(l) for l in open(p,encoding="utf-8") if l.strip()] if os.path.exists(p) else []
r1,r2=load("bench/round1.jsonl"),load("bench/round2.jsonl")
def pick(rows,i,c):
    for r in rows:
        if r.get("random_input_len")==i and r.get("max_concurrency")==c: return r
    return None
print("| 输入档 | 并发 | 请求数 | 轮次 | completed | duration (s) | 输出吞吐 (tok/s) | 总吞吐 (tok/s) | 平均 TTFT (ms) | P99 TTFT (ms) | 平均 TPOT (ms) | P99 TPOT (ms) | 平均端到端 (ms) | 输入token(文本) | 输入token(视觉) | 输出token |")
print("| --- |"*16)
for (i,o,c,n) in ROWS:
    for rn,rows in (("R1",r1),("R2",r2)):
        r=pick(rows,i,c)
        if r is None:
            print("| %d | %d | %d | %s | - | - | - | - | - | - | - | - | - | - | - | - |"%(i,c,n,rn)); continue
        print("| {} | {} | {} | {} | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {} | {} | {} |".format(
            i,c,n,rn,r.get("completed","-"),r.get("duration",0),r.get("output_throughput",0),r.get("total_throughput",0),
            r.get("mean_ttft_ms",0),r.get("p99_ttft_ms",0),r.get("mean_tpot_ms",0),r.get("p99_tpot_ms",0),
            r.get("mean_e2e_latency_ms",0),r.get("total_input_text_tokens","-"),r.get("total_input_vision_tokens","-"),r.get("total_output_tokens","-")))
PY
}

echo
echo "===== summary.md ====="
cat bench/summary.md
