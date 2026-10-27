#!/usr/bin/env bash
# 抽取 E1（非 eager）确认实验 vs 基线（eager）的同口径指标
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1

echo "===== 找基线（eager）压测结果目录 ====="
ls -d outputs/phase4* outputs/phase5* 2>/dev/null
echo
echo "--- 候选 jsonl ---"
find outputs -name '*.jsonl' -newermt '2026-10-01' 2>/dev/null | grep -iE 'bench|round|ab_' | head -n 40

echo
echo "===== 汇总：逐文件提取 output_throughput / TTFT ====="
python3 - <<'PY'
import json, os, glob

def summarize(path):
    try:
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    except Exception as e:
        return None
    if not rows:
        return None
    r = rows[0]
    if "output_throughput" not in r:
        return None
    return {
        "file": path,
        "in_len": r.get("random_input_len"),
        "conc": r.get("max_concurrency"),
        "completed": r.get("completed"),
        "duration": round(r.get("duration", 0), 2),
        "out_tok": r.get("total_output_tokens"),
        "out_tps": round(r.get("output_throughput", 0), 2),
        "req_tps": round(r.get("request_throughput", 0), 4),
        "ttft_mean_ms": round(r.get("mean_ttft_ms", 0), 1),
        "ttft_p99_ms": round(r.get("p99_ttft_ms", 0), 1),
        "e2e_mean_ms": round(r.get("mean_e2e_latency_ms", 0), 1),
        "tpot_mean_ms": round(r.get("mean_tpot_ms", 0), 2),
    }

files = []
for pat in ("outputs/phase4_bench/*.jsonl", "outputs/phase4*/**/*.jsonl",
            "outputs/phase5_ab_noneager/**/ab_*.jsonl",
            "outputs/phase5*/ab_*.jsonl", "outputs/bench*/**/*.jsonl"):
    files += glob.glob(pat, recursive=True)
files = sorted(set(files))
rows = []
for f in files:
    s = summarize(f)
    if s:
        rows.append(s)
rows.sort(key=lambda r: (str(r["in_len"]), r["conc"] or 0, r["file"]))
for r in rows:
    print(json.dumps(r, ensure_ascii=False))
PY
