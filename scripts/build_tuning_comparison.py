#!/usr/bin/env python3
"""生成「首提基线 eager / graph 基线 / 最优版」的同口径对比表。

口径说明（必须一起读）：
  * 所有数值都是 vLLM serving + 官方 bench_serving 客户端（image 数据集，1×1080p 图/请求，
    输出 1024 token，seed 1，请求数 max(8, 2×并发)）实测。
  * **绝对值只在同一张卡上可比。** 容器被平台回收重建后卡会更换，新卡上测的
    绝对值与旧卡不可直接相减；因此收益结论一律用**同一张卡上三套配置**的 A/B：
        eager     首提基线（--enforce-eager + stock 库 + 默认 attention）
        baseline  graph 基线（非 eager + stock 库 + 默认 attention）
        candidate 最优版（非 eager + GEMM 调优库 + ROCM_AITER_FA）
  * 「同一张卡」由 scripts/bench_set.py 的会话模型判定：当前卡的运行目录
    （run_config.txt 里 card_id 与 outputs/CARD_ID.txt 一致）优先；
    当前卡凑不齐时回退到「历史同节点会话」（卡指纹未记录），并显式标注为历史。
    **两个会话之间绝不互相相减。**

用法：
    python3 scripts/build_tuning_comparison.py --out reports/tuning_comparison.md
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_set  # noqa: E402

WS = bench_set.WS
ORDER = bench_set.ORDER

# 跨卡存档（历史实验，永不参与同卡相减）
ARCHIVE_FIRST = ["submission/bench/round1.jsonl", "submission/bench/round2.jsonl"]
ARCHIVE_GRAPH30 = ["tmp/graph_bench/baseline_oldlib/round1.jsonl",
                    "tmp/graph_bench/baseline_oldlib/round2.jsonl"]
ARCHIVE_TUNED30 = ["tmp/graph_bench/tuned_085/round1.jsonl",
                   "tmp/graph_bench/tuned_085/round2.jsonl"]
ARCHIVE_E7 = ["tmp/gate_aiterfa.jsonl"]


def load(paths):
    """按 (输入长度, 并发) 聚合（静态文件路径列表，相对工作目录）。"""
    rows = {}
    for rel in paths:
        if "*" in rel:
            files = sorted(WS.glob(rel))
        else:
            p = WS / rel
            files = sorted(p.glob("*.jsonl")) if p.is_dir() else ([p] if p.exists() else [])
        for f in files:
            seen = set()
            try:
                text = f.read_text(encoding="utf-8")
            except OSError:
                continue
            for line in text.splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (r.get("random_input_len"), r.get("max_concurrency"))
                if key in seen:
                    continue
                seen.add(key)
                rows.setdefault(key, []).append(r)
    return rows


def mean(rs, field):
    return sum(float(r.get(field, 0) or 0) for r in rs) / len(rs) if rs else 0.0


def fmt_delta(base, cand, field):
    b, c = mean(base, field), mean(cand, field)
    if b <= 0:
        return "—"
    return f"{(c / b - 1) * 100:+.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=WS / "reports/tuning_comparison.md")
    ap.add_argument("--node-label", default=os.environ.get("BENCH_NODE_LABEL", "当前节点（同一张卡）"))
    args = ap.parse_args()

    card = bench_set.current_card_id()
    card_info = bench_set.current_card_info()
    sessions = bench_set.sessions()

    lines = ["# 调优对比表：首提基线 eager / graph 基线 / 最优版", ""]
    lines.append("> 口径：单卡 K100AI（K500SM_AI / gfx928 / 63.98 GiB）/ vLLM 0.15.1+das.opt1.alpha.dtk2604 /")
    lines.append("> BF16 / TP=1 / `--gpu-memory-utilization 0.85 --max-model-len 32768` /")
    lines.append("> `--max-num-batched-tokens 4096 --max-num-seqs 64 --no-enable-prefix-caching` /")
    lines.append("> `--limit-mm-per-prompt '{\"image\":8}'`；官方 bench_serving 客户端（image 数据集，")
    lines.append("> 1×1080p 图/请求，输出 1024 token，seed 1，请求数 max(8, 2×并发)）。")
    lines.append(">")
    lines.append("> **换卡警告（重要）**：SCNet 容器被平台回收重建后**物理卡会更换**。")
    lines.append("> 换卡前后的绝对值**不可直接相减**（实测同配置跨卡可差 30%），")
    lines.append("> 因此收益结论**只用同一张卡上三套配置的 A/B**。")
    if card:
        lines.append(">")
        lines.append(f"> 卡指纹 `card_id = {card}`")
        lines.append(f"> （来源：{card_info.get('card_id_source', '—')}；"
                     f"卡名 {card_info.get('card_name', '—')}；架构 {card_info.get('card_gfx', '—')}；"
                     f"节点 {card_info.get('node_host', '—')}）")
    else:
        lines.append(">")
        lines.append("> ⚠️ **未读到当前卡的指纹**（缺少 `outputs/CARD_ID.txt`），"
                     "本次未做任何同卡判定；同卡段可能为空。")
    lines.append("")

    # --- 1) 逐配置逐档（按会话分组）---
    lines.append("## 1. 各配置的逐档实测")
    lines.append("")
    lines.append("| 会话 | 配置 | 输入 | 并发 | 轮数 | 输出吞吐 tok/s | TTFT ms | TPOT ms | 失败 |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |")

    def dump_rows(session_label, config_label, rows):
        for key in ORDER:
            rs = rows.get(key)
            if not rs:
                continue
            tp = [r.get("output_throughput", 0) for r in rs]
            ttft = [r.get("mean_ttft_ms", 0) for r in rs]
            tpot = [r.get("mean_tpot_ms", 0) for r in rs]
            failed = sum(r.get("failed", 0) or 0 for r in rs)
            lines.append(
                f"| {session_label} | {config_label} | {key[0]} | {key[1]} | {len(rs)} | "
                f"{'/'.join(f'{v:.2f}' for v in tp)} | {'/'.join(f'{v:.0f}' for v in ttft)} | "
                f"{'/'.join(f'{v:.2f}' for v in tpot)} | {failed} |")

    for label, cfgmap in sessions:
        for cfg, cfg_label in (("eager", "首提基线 eager"), ("baseline", "graph 基线"),
                               ("candidate", "最优版")):
            if cfg not in cfgmap:
                continue
            dump_rows(label, cfg_label, bench_set.load_dirs(cfgmap[cfg]))

    dump_rows("跨卡存档", "首提基线 eager", load(ARCHIVE_FIRST))
    dump_rows("跨卡存档", "graph 基线", load(ARCHIVE_GRAPH30))
    dump_rows("跨卡存档", "调优库 E5", load(ARCHIVE_TUNED30))
    dump_rows("跨卡存档", "调优库+aiterFA E7", load(ARCHIVE_E7))
    lines.append("")

    # --- 2) 同卡 A/B ---
    lines.append("## 2. 同卡 A/B 提升（唯一可直接相减的列）")
    lines.append("")

    def ab_table(title, base, cand_rows, base_label, cand_label, note=""):
        if not base or not cand_rows:
            return False
        lines.append(title)
        lines.append("")
        if note:
            lines.append(note)
            lines.append("")
        lines.append("| 输入 | 并发 | " + base_label + " tok/s | " + cand_label +
                     " tok/s | 吞吐提升 | TTFT 变化 | TPOT 变化 | 轮次 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        hit = False
        for key in ORDER:
            b, c = base.get(key), cand_rows.get(key)
            if not b or not c:
                continue
            bt, ct = mean(b, "output_throughput"), mean(c, "output_throughput")
            if bt <= 0:
                continue
            hit = True
            lines.append(
                f"| {key[0]} | {key[1]} | {bt:.2f} | {ct:.2f} | {(ct/bt-1)*100:+.2f}% | "
                f"{fmt_delta(b, c, 'mean_ttft_ms')} | {fmt_delta(b, c, 'mean_tpot_ms')} | "
                f"{len(b)}/{len(c)} |")
        lines.append("")
        return hit

    main_hit = False
    for label, cfgmap in sessions:
        eager = bench_set.load_dirs(cfgmap["eager"]) if "eager" in cfgmap else {}
        graph = bench_set.load_dirs(cfgmap["baseline"]) if "baseline" in cfgmap else {}
        cand = bench_set.load_dirs(cfgmap["candidate"]) if "candidate" in cfgmap else {}
        if eager and cand:
            main_hit = ab_table(
                f"### 2.1 同卡：首提基线 eager → 最优版（赛事口径的正式对比）〔{label}〕",
                eager, cand, "首提基线 eager", "最优版",
                "> 两者在同一张卡、同一负载、同一客户端下测得，只差配置"
                "（eager+stock 库 vs graph+调优库+aiter FA）。这是本轮**主结论**。") or main_hit
        if graph and cand:
            ab_table(
                f"### 2.2 同卡：graph 基线 → 最优版（调优库 + aiter FA 的净贡献）〔{label}〕",
                graph, cand, "graph 基线", "最优版",
                "> 只开 graph、不带任何调优开关的对照，用于把收益拆成「graph」与「调优库+aiter FA」两部分。")
        if eager and graph:
            ab_table(
                f"### 2.3 同卡：首提基线 eager → graph 基线（graph 本身的贡献）〔{label}〕",
                eager, graph, "首提基线 eager", "graph 基线")

    arch_base30 = load(ARCHIVE_GRAPH30)
    arch_tuned30 = load(ARCHIVE_TUNED30)
    arch_e7 = load(ARCHIVE_E7)
    if arch_base30 and arch_tuned30:
        lines.append("### 2.4 跨卡存档（不可与上表相减）")
        lines.append("")
        lines.append("> 下面是旧卡（已被平台回收）上的历史实验，仅作过程存档。")
        lines.append("")
        lines.append("| 输入 | 并发 | graph 基线 tok/s | 调优库 tok/s | 吞吐提升 | TTFT 变化 | TPOT 变化 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for key in ORDER:
            b, t = arch_base30.get(key), arch_tuned30.get(key)
            if not b or not t:
                continue
            bt, tt = mean(b, "output_throughput"), mean(t, "output_throughput")
            lines.append(
                f"| {key[0]} | {key[1]} | {bt:.2f} | {tt:.2f} | {(tt/bt-1)*100:+.2f}% | "
                f"{fmt_delta(b, t, 'mean_ttft_ms')} | {fmt_delta(b, t, 'mean_tpot_ms')} |")
        lines.append("")
    if arch_base30 and arch_e7:
        lines.append("### 2.5 跨卡存档：graph 基线 → 调优库 + aiter FA（旧卡 E7 门槛）")
        lines.append("")
        lines.append("| 输入 | 并发 | graph 基线 tok/s | 最优版 tok/s | 吞吐提升 |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for key in ORDER:
            b, t = arch_base30.get(key), arch_e7.get(key)
            if not b or not t:
                continue
            bt, tt = mean(b, "output_throughput"), mean(t, "output_throughput")
            lines.append(f"| {key[0]} | {key[1]} | {bt:.2f} | {tt:.2f} | {(tt/bt-1)*100:+.2f}% |")
        lines.append("")

    if not main_hit:
        lines.append("### 2.0 同卡主结论缺失")
        lines.append("")
        lines.append("> 没有任何一个会话同时具备「首提基线 eager」与「最优版」的数据，"
                     "主结论留待同卡重测后补。")
        lines.append("> **不用跨卡数据顶替**。")
        lines.append("")

    # --- 3) 数据来源 ---
    lines.append("## 3. 数据来源")
    lines.append("")
    lines.append("| 会话 | 配置 | 运行目录 |")
    lines.append("| --- | --- | --- |")
    for label, cfgmap in sessions:
        for cfg, dirs in cfgmap.items():
            for d in dirs:
                rel = Path(d).relative_to(WS) if str(d).startswith(str(WS)) else d
                lines.append(f"| {label} | {bench_set.CONFIG_LABEL.get(cfg, cfg)} | `{rel}` |")
    for cfg in ("candidate", "baseline", "eager"):
        for d in bench_set.resolve(cfg, card)[1]:
            rel = Path(d).relative_to(WS) if str(d).startswith(str(WS)) else d
            lines.append(f"| 跨卡存档（未参与同卡 A/B） | {bench_set.CONFIG_LABEL.get(cfg, cfg)} | `{rel}` |")
    lines.append("")

    out = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
