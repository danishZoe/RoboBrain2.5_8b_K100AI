#!/usr/bin/env python3
"""在 reports/25 的限制表里补充实测口径说明（幂等）。

新增两行：
  1) 16384/c64 档只跑了单轮（两轮需 7.5 h+）
  2) 16384/c64 档没有同节点 graph 基线对照（基线补测只覆盖另外三档）

用法：
    python3 scripts/patch_report25_limits.py [reports/25_tuning_delivery_20261031.md]
"""
import json
import os
import re
import sys
from pathlib import Path

WS = Path(os.environ.get("ROBO_WS", "/root/private_data/workspace/robo"))
CHAIN = "outputs/fullbench_baseline_20261101_043223"
ORDER = [(4096, 1), (4096, 8), (4096, 64), (16384, 1), (16384, 8), (16384, 64)]


def chain_tiers():
    """链式基线补测实际覆盖到的档位。"""
    got = set()
    d = WS / CHAIN
    if not d.is_dir():
        return got
    for f in d.glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            got.add((r.get("random_input_len"), r.get("max_concurrency")))
    return got


def main():
    p = Path(sys.argv[1] if len(sys.argv) > 1
             else WS / "reports/25_tuning_delivery_20261031.md")
    text = p.read_text(encoding="utf-8")
    got = chain_tiers()

    rows = []
    if "16384/c64 档轮次" not in text:
        rows.append(
            "| 16384/c64 档轮次 | **单轮** | 该档单轮耗时 3 h 48 min"
            "（128 请求排队 69 min，TTFT 均值 4,134,411 ms），两轮需 7.5 h+；"
            "其余五档均为两轮。表格已标注轮数 |")
    if "同节点 graph 基线补测" not in text:
        covered = "、".join(f"{i}/{c}" for (i, c) in ORDER if (i, c) in got) or "无"
        if got:
            state, why = "已完成", f"覆盖 {covered}"
        else:
            state = "NOT_RUN"
            why = ("补测于 2026-11-01 04:36 启动（graph 基线服务 READY after 230 s），"
                   "预期 05:05 完成，但节点在 05:08 断开，结果未能取回；"
                   "因此同节点 graph 基线对照只覆盖 4096/并发1、4096/并发8 两档")
        rows.append(f"| 同节点 graph 基线补测（16384/并发1、16384/并发8、4096/并发64） | {state} | {why} |")
    if "16384/c64 档的同节点 graph 基线对照" not in text:
        if (16384, 64) in got:
            state = "已完成"
            why = "基线补测覆盖了该档"
        else:
            state = "NOT_RUN"
            why = ("同节点 graph 基线补测未覆盖该档（补测结果因节点中断未能取回）；"
                   "该档基线单轮同样需 ~3.8 h")
        rows.append(
            f"| 16384/c64 档的同节点 graph 基线对照 | {state} | {why} |")
    if "本轮压测结果的取回方式" not in text:
        rows.append(
            "| 本轮压测结果的取回方式 | **会话日志转录** | "
            "2026-11-01 05:08 起 dcu_1026 容器不可达（网关端口可连但收不到 SSH banner，"
            "同网关其他容器正常），节点工作目录无法访问；"
            "最优版六档结果由压测客户端与汇总脚本的**原样输出**转录恢复"
            "（见 `reports/evidence/bench_recovered_20261101/`），"
            "因此**不是客户端原始 jsonl 文件**，其中 3 个档位的 "
            "`total_throughput` / `p99_ttft_ms` / `p99_tpot_ms` 未被留存，表中以 `—` 标注 |")

    if not rows:
        print("already present, skip")
        return

    anchor = "| c64 以上并发摸高 | NOT_RUN |"
    if anchor not in text:
        sys.exit("ERROR: anchor not found")
    text = text.replace(anchor, "\n".join(rows) + "\n" + anchor, 1)
    p.write_text(text, encoding="utf-8")
    print(f"updated: {p} (+{len(rows)} rows)")


if __name__ == "__main__":
    main()
