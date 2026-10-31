#!/usr/bin/env bash
# 官方压测汇总：
#   bench/summary.md          -> 完全按官方 bench.sh 的 13 列格式（提交 README §7 直接引用）
#   bench/summary_detailed.md -> 增补列（失败数、输入 token 文本/视觉拆分、实测输出 token）
set -uo pipefail
cd /root/private_data/workspace/robo/submission

python3 - <<'PY'
import json, os, re, glob

ROWS = [(4096, 1024, 1, 8), (4096, 1024, 8, 16), (4096, 1024, 64, 128),
        (16384, 1024, 1, 8), (16384, 1024, 8, 16), (16384, 1024, 64, 128)]

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []

def pick(rows, i, c):
    for r in rows:
        if r.get("random_input_len") == i and r.get("max_concurrency") == c:
            return r
    return None

def log_failed(tag):
    """从各档日志里取 Failed requests；没有该行说明客户端未报失败"""
    p = "bench/{}.log".format(tag)
    if not os.path.exists(p):
        return None
    m = re.search(r"Failed requests:\s+(\d+)", open(p, encoding="utf-8", errors="replace").read())
    return int(m.group(1)) if m else 0

r1, r2 = load("bench/round1.jsonl"), load("bench/round2.jsonl")

# ---------- 官方格式（与官方 bench.sh 的 summary.md 完全一致）----------
OFFICIAL_COLS = ["random_input_len", "random_output_len", "max_concurrency",
                 "completed", "duration", "request_throughput", "output_throughput",
                 "total_throughput", "mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms",
                 "p99_tpot_ms", "mean_e2e_latency_ms"]

with open("bench/summary.md", "w", encoding="utf-8") as f:
    print("| " + " | ".join(OFFICIAL_COLS) + " |", file=f)
    print("|" + " --- |" * len(OFFICIAL_COLS), file=f)
    for (i, o, c, n) in ROWS:
        r = pick(r2, i, c)
        if r is None:
            print("| " + " | ".join(str(x) for x in
                  (i, o, c, "-", "-", "-", "-", "-", "-", "-", "-", "-", "-")) + " |", file=f)
            continue
        cells = []
        for col in OFFICIAL_COLS:
            v = r.get(col, "-")
            cells.append("{:.2f}".format(v) if isinstance(v, float) else str(v))
        print("| " + " | ".join(cells) + " |", file=f)

# ---------- 详细表 ----------
with open("bench/summary_detailed.md", "w", encoding="utf-8") as f:
    print("## 官方格式压测结果（SGLang 压测客户端 image 数据集 → vLLM 服务）\n", file=f)
    print("| 输入档 | 并发 | 请求数 | 轮次 | completed | failed | duration s | 输出吞吐 tok/s | 总吞吐 tok/s | 平均TTFT ms | P99 TTFT ms | 平均TPOT ms | P99 TPOT ms | 平均端到端 ms | 输入token(文本) | 输入token(视觉) | 输出token | 实际输出token(重分词) |",
          file=f)
    print("| --- |" * 18, file=f)
    for (i, o, c, n) in ROWS:
        for rnd, rows in (("R1", r1), ("R2", r2)):
            tag = "round{}_{}_{}_c{}".format(1 if rnd == "R1" else 2, i, o, c)
            r = pick(rows, i, c)
            if r is None:
                print("| {} | {} | {} | {} | - | - | - | - | - | - | - | - | - | - | - | - | - | - |".format(
                    i, c, n, rnd), file=f)
                continue
            failed = r.get("failed")
            if failed is None:
                failed = log_failed(tag)
            print("| {} | {} | {} | {} | {} | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {} | {} | {} | {} |".format(
                i, c, n, rnd, r.get("completed", "-"), failed if failed is not None else "-",
                r.get("duration", 0), r.get("output_throughput", 0), r.get("total_throughput", 0),
                r.get("mean_ttft_ms", 0), r.get("p99_ttft_ms", 0),
                r.get("mean_tpot_ms", 0), r.get("p99_tpot_ms", 0),
                r.get("mean_e2e_latency_ms", 0),
                r.get("total_input_text_tokens", "-"), r.get("total_input_vision_tokens", "-"),
                r.get("total_output_tokens", "-"),
                r.get("total_output_tokens_retokenized", "-")), file=f)
    print("\n### 每档客户端数据构造统计（#Input tokens = 文本+视觉）\n", file=f)
    print("| 档位 | #Input tokens | #Output tokens | #Total images |", file=f)
    print("| --- | --- | --- | --- |", file=f)
    for p in sorted(glob.glob("bench/round2_*.log")):
        txt = open(p, encoding="utf-8", errors="replace").read()
        gi = re.search(r"#Input tokens:\s+(\d+)", txt)
        go = re.search(r"#Output tokens:\s+(\d+)", txt)
        gm = re.search(r"#Total images:\s+(\d+)", txt)
        if gi:
            print("| {} | {} | {} | {} |".format(
                os.path.basename(p)[len("round2_"):-len(".log")], gi.group(1),
                go.group(1) if go else "-", gm.group(1) if gm else "-"), file=f)
    print()

print("wrote bench/summary.md and bench/summary_detailed.md")
PY

echo
echo "===== bench/summary.md (官方格式) ====="
cat bench/summary.md
echo
echo "===== bench/summary_detailed.md ====="
cat bench/summary_detailed.md
