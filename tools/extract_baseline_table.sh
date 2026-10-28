#!/usr/bin/env bash
# 提取 eager 基线（首次提交口径）六档同口径指标
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1

echo "===== 基线目录结构 ====="
find outputs/phase5_bench outputs/phase4 -type f | head -n 40

echo
echo "===== 逐文件提取 ====="
python3 - <<'PY'
import json, os, glob
files = []
for pat in ("outputs/phase5_bench/**/*.jsonl", "outputs/phase5_bench/**/*.json",
            "outputs/phase4/*.json", "outputs/phase5_ab_noneager/**/*.jsonl"):
    files += glob.glob(pat, recursive=True)
files = sorted(set(files))
for f in files:
    try:
        rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    except Exception:
        try:
            d = json.load(open(f, encoding="utf-8"))
            rows = d if isinstance(d, list) else [d]
        except Exception:
            continue
    if not rows:
        continue
    r = rows[0]
    if not isinstance(r, dict) or "output_throughput" not in r:
        continue
    print(json.dumps({
        "file": f,
        "in_len": r.get("random_input_len"),
        "conc": r.get("max_concurrency"),
        "completed": r.get("completed"),
        "duration": round(r.get("duration", 0), 2),
        "out_tok": r.get("total_output_tokens"),
        "out_tps": round(r.get("output_throughput", 0), 2),
        "ttft_mean_ms": round(r.get("mean_ttft_ms", 0), 1),
        "ttft_p99_ms": round(r.get("p99_ttft_ms", 0), 1),
        "e2e_mean_ms": round(r.get("mean_e2e_latency_ms", 0), 1),
        "tpot_mean_ms": round(r.get("mean_tpot_ms", 0), 2),
    }, ensure_ascii=False))
PY

echo
echo "===== submission/bench/summary.md 里的基线汇总 ====="
sed -n '1,60p' submission/bench/summary.md 2>/dev/null

echo
echo "===== 服务状态 ====="
(ss -ltnp 2>/dev/null | grep 30000 || echo 'not listening')
curl -s --max-time 5 http://127.0.0.1:30000/v1/models | head -c 200; echo
