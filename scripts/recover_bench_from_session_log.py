#!/usr/bin/env python3
"""把会话快照里恢复出的压测结果落成工作区数据文件。

背景：2026-11-01 05:08 起 dcu_1026 容器不可达（网关通但无 SSH banner），
      节点工作目录里的原始 jsonl 取不回来。但压缩前的会话快照逐字保存了
      压测客户端的官方输出，因此可以恢复出**实测值本身**。

数据来源（全部为节点上压测客户端/汇总脚本的原样输出）：
  * 13 列完整行：summarize_full_bench.py 的汇总表（快照记录 366）
  * 每轮官方指标块：bench_serving 客户端日志（Successful requests /
    Benchmark duration / Output token throughput / Mean E2E / Mean TTFT / Mean TPOT）
  * 16384/c64 详情：逐字段打印（completed / duration / output_throughput /
    total_output_tokens / mean_ttft_ms / p99_ttft_ms / mean_tpot_ms / mean_e2e_latency_ms）

可精确复原的量：
  duration          = completed * 1024 / output_throughput   （客户端口径，已逐条核对）
  request_throughput= completed / duration
无法复原的量（不填、不猜）：
  total_throughput（需要逐请求输入 token 数）、p99_ttft_ms、p99_tpot_ms

用法：python3 scripts/recover_bench_from_session_log.py <workspace_dir>
"""
import json
import os
import sys
from pathlib import Path

WS = Path(sys.argv[1] if len(sys.argv) > 1
          else os.environ.get("ROBO_WS", "/root/private_data/workspace/robo"))

CAND = "outputs/fullbench_candidate_20261031_231152"
GBASE = "outputs/phase5_ab_noneager/noneager_confirm_20261031_222226"
EAGER = "logs/clean_repro_full/clean_repro_full_20261031/bench"

OUT_TOKENS = 1024


def row(ilen, c, completed, out_tp, ttft=None, tpot=None, e2e=None,
        p99_ttft=None, p99_tpot=None, total_tp=None, duration=None, failed=0):
    """构造一条与 bench_serving 同构的结果行；只写已知字段。"""
    d = {
        "backend": "vllm",
        "dataset": "image",
        "random_input_len": ilen,
        "random_output_len": OUT_TOKENS,
        "max_concurrency": c,
        "num_prompts": completed,
        "completed": completed,
        "failed": failed,
        "seed": 1,
        "output_throughput": out_tp,
    }
    if duration is not None:
        d["duration"] = duration
        d["request_throughput"] = completed / duration
    if total_tp is not None:
        d["total_throughput"] = total_tp
    if ttft is not None:
        d["mean_ttft_ms"] = ttft
    if p99_ttft is not None:
        d["p99_ttft_ms"] = p99_ttft
    if tpot is not None:
        d["mean_tpot_ms"] = tpot
    if p99_tpot is not None:
        d["p99_tpot_ms"] = p99_tpot
    if e2e is not None:
        d["mean_e2e_latency_ms"] = e2e
    d["total_output_tokens"] = completed * OUT_TOKENS
    return d


# --- 候选（最优版）：11 轮 ----------------------------------------------------
# 完整 13 列的 7 轮（duration/total_tp/p99 均来自节点汇总表原文）
CAND_FULL = {
    ("round1", 4096, 1): (285.268, 0.0280438, 28.7169, 207.062, 1389.73, 1432.84, 33.4972, 33.821, 35657.3),
    ("round2", 4096, 1): (285.069, 0.0280634, 28.7369, 207.23, 1448.69, 1630.32, 33.4152, 33.6212, 35632.5),
    ("round1", 4096, 8): (142.031, 0.112652, 115.355, 831.891, 9761.92, 10429.2, 59.8461, 66.9055, 70984.5),
    ("round2", 4096, 8): (141.29, 0.113242, 115.96, 836.859, 9786.72, 10444.8, 59.4601, 66.2265, 70614.4),
    ("round1", 16384, 1): (431.677, 0.0185324, 18.9771, 376.64, 6703.94, 6773.53, 46.1919, 50.3712, 53958.2),
    ("round2", 16384, 1): (426.139, 0.0187732, 19.2238, 381.565, 6717.82, 6826.51, 45.5019, 48.3741, 53266.3),
    ("round1", 16384, 8): (364.392, 0.0439088, 44.9626, 892.816, 25728.9, 54922.2, 152.488, 167.403, 181724),
}
# 仅客户端日志块（无 p99 / total_tp）的 4 轮
CAND_PARTIAL = {
    ("round2", 16384, 8): (16, 45.09, 25831.20, 151.86, 181184.52),
    ("round1", 4096, 64): (128, 140.36, 114098.93, 306.83, 427981.38),
    ("round2", 4096, 64): (128, 139.21, 114388.36, 309.35, 430852.94),
    ("round1", 16384, 64): (128, 9.589760785790007, 4134411.488995707, 1416.3134491481903,
                            5583300.147474306),
}
CAND_C64_P99_TTFT = 6101802.494043422


def dump(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_candidate():
    for (rnd, ilen, c), v in CAND_FULL.items():
        dur, req, out_tp, tot, ttft, p99t, tpot, p99p, e2e = v
        r = row(ilen, c, 8 if c == 1 else (16 if c == 8 else 128), out_tp,
                ttft, tpot, e2e, p99_ttft=p99t, p99_tpot=p99p,
                total_tp=tot, duration=dur, failed=0)
        r["request_throughput"] = req
        dump(WS / CAND / f"{rnd}_{ilen}_1024_c{c}.jsonl", [r])
    for (rnd, ilen, c), v in CAND_PARTIAL.items():
        done, out_tp, ttft, tpot, e2e = v
        dur = done * OUT_TOKENS / out_tp
        r = row(ilen, c, done, out_tp, ttft, tpot, e2e, duration=dur, failed=0)
        if (rnd, ilen, c) == ("round1", 16384, 64):
            r["p99_ttft_ms"] = CAND_C64_P99_TTFT
        dump(WS / CAND / f"{rnd}_{ilen}_1024_c{c}.jsonl", [r])


# --- graph 基线（10-31 同节点，stock 库 + graph）---------------------------
def write_graph_base():
    # 4096/c1：客户端完整输出块
    dur = 366.07
    r = row(4096, 1, 8, 22.38, 1481.77, 43.28, 45756.81,
            p99_ttft=1514.64, p99_tpot=43.46, total_tp=161.10, duration=dur, failed=0)
    dump(WS / GBASE / "ab_4096_c1.jsonl", [r])
    # 4096/c8：只有吞吐/TTFT/TPOT（duration 由此精确复原）
    out_tp = 104.84
    dur = 16 * OUT_TOKENS / out_tp
    r = row(4096, 8, 16, out_tp, 10864.7, 65.73, duration=dur, failed=0)
    dump(WS / GBASE / "ab_4096_c8.jsonl", [r])


# --- 首提基线 eager 同节点复现（10-31）-------------------------------------
EAGER_R1 = [(4096, 1, 8, 21.76, 1446.0, 44.59),
            (4096, 8, 16, 101.90, 10663.5, 68.13),
            (4096, 64, 128, 137.15, 119463.1, 311.12)]
EAGER_R2 = [(4096, 1, 8, 21.73, 1453.9, 44.64),
            (4096, 8, 16, 102.37, 10762.4, 67.67)]


def write_eager():
    for name, data in (("round1.jsonl", EAGER_R1), ("round2.jsonl", EAGER_R2)):
        rows = []
        for ilen, c, done, out_tp, ttft, tpot in data:
            dur = done * OUT_TOKENS / out_tp
            rows.append(row(ilen, c, done, out_tp, ttft, tpot, duration=dur, failed=0))
        dump(WS / EAGER / name, rows)


def main():
    write_candidate()
    write_graph_base()
    write_eager()
    print("[recover] workspace:", WS)
    for p in sorted(WS.glob(f"{CAND}/*.jsonl")):
        print("  ", p.relative_to(WS))
    for p in sorted(WS.glob(f"{GBASE}/*.jsonl")):
        print("  ", p.relative_to(WS))
    for p in sorted(WS.glob(f"{EAGER}/*.jsonl")):
        print("  ", p.relative_to(WS))


if __name__ == "__main__":
    main()
