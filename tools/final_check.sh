#!/usr/bin/env bash
# =============================================================================
# 最终验收与同步（远端）
#   1) 等流水线结束
#   2) 跑提交包自检
#   3) 核对六档是否都有两轮数据
#   4) 打印产物清单
# 用法: bash tools/final_check.sh
# =============================================================================
set -uo pipefail
W=/root/private_data/workspace/robo
cd "$W/submission" || exit 1

echo "=== 1) 等待收尾流水线结束 ==="
DEADLINE=$(( $(date +%s) + 21600 ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  if ! pgrep -f 'pipeline_after_bench.sh' >/dev/null 2>&1; then
    echo "pipeline finished at $(date -Is)"; break
  fi
  sleep 30
done

echo
echo "=== 2) 回填最终报告表格（reports/05_final.md）==="
python3 "$W/tools/inject_final_report.py" "$W/reports/05_final.md" 2>&1
python3 "$W/tools/inject_optimization_gate.py" "$W/reports/05_final.md" 2>&1
python3 "$W/tools/inject_tuning_results.py" "$W/reports/11_tuning_experiments.md" 2>&1

echo
echo "=== 3) 提交包自检 ==="
bash "$W/tools/check_submission.sh" 2>&1 | tail -70

echo
echo "=== 4) 六档两轮完整性 ==="
python3 - <<'PY'
import json, os
ROWS = [(4096,1),(4096,8),(4096,64),(16384,1),(16384,8),(16384,64)]
def load(p):
    return [json.loads(l) for l in open(p,encoding="utf-8") if l.strip()] if os.path.exists(p) else []
r1, r2 = load("bench/round1.jsonl"), load("bench/round2.jsonl")
allok = True
print("| 档位 | R1 | R2 |")
print("| --- | --- | --- |")
for (i,c) in ROWS:
    a = [r for r in r1 if r.get("random_input_len")==i and r.get("max_concurrency")==c]
    b = [r for r in r2 if r.get("random_input_len")==i and r.get("max_concurrency")==c]
    sa = "%.2f tok/s (%d done/%d failed)" % (a[0]["output_throughput"], a[0]["completed"], a[0].get("failed",0)) if a else "缺失"
    sb = "%.2f tok/s (%d done/%d failed)" % (b[0]["output_throughput"], b[0]["completed"], b[0].get("failed",0)) if b else "缺失"
    if not (a and b): allok = False
    print("| %d / conc %d | %s | %s |" % (i, c, sa, sb))
print("\n六档两轮齐全:", "YES" if allok else "NO（缺失档位必须写进报告的限制）")
PY

echo
echo "=== 5) 关键产物 ==="
for f in env.log server.log smoke.log verify_result.json bench/summary.md bench/summary_detailed.md; do
  if [ -s "$f" ]; then printf '  %-28s %8s B\n' "$f" "$(wc -c < "$f")"; else printf '  %-28s MISSING\n' "$f"; fi
done
echo
echo "=== 6) 调优实验产物 ==="
ls -1 "$W/outputs/phase5_ab_noneager" 2>/dev/null | sed 's/^/  noneager: /'
ls -1 "$W/outputs/phase5_ab_bnb" 2>/dev/null | sed 's/^/  bnb: /'
ls -1 "$W/outputs/phase5_gemm_trace" 2>/dev/null | sed 's/^/  trace: /'
ls -1 "$W/outputs/phase5_kv_bnb" 2>/dev/null | sed 's/^/  kv_by_bnt: /'
echo
echo "=== 7) 样例产物（验收 §9）==="
for f in samples/MANIFEST.md samples/upstream_provenance.json samples/parsed_outputs.json; do
  if [ -s "$f" ]; then printf '  %-34s %8s B\n' "$f" "$(wc -c < "$f")"; else printf '  %-34s MISSING\n' "$f"; fi
done
ls -1 samples/annotated/*.png 2>/dev/null | sed 's/^/  overlay: /'
echo
echo "=== 8) 交付 revision ==="
if [ -s "$W/.revision" ]; then
  grep -E '^(revision|revision_short|recorded_at|tracked_files)' "$W/.revision" | sed 's/^/  /'
else
  echo "  .revision 尚未固化（阶段 A 未跑）"
fi
[ -s MANIFEST.sha256 ] && printf '  MANIFEST.sha256                  %8s 行\n' "$(wc -l < MANIFEST.sha256)"
