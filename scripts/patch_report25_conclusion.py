#!/usr/bin/env python3
"""按原始 JSONL 数据重写 reports/25 的结论数值与差异表「依据」列。

背景：
  reports/25 §1 结论表里残留了两组**跨节点**旧值（+28.5%/+22%、+64%/+25%），
  与本轮「只用同节点同轮 A/B」的口径冲突；§2 差异表第 1 行还把
  「graph 基线 → 调优库」的 TPOT 变化（29.4 → 23.4 ms，10-30）误挂在
  「去掉 --enforce-eager」上。

本脚本只做**可由原始数据复算**的替换：
  * 所有百分比、tok/s、TTFT/TPOT 都从 JSONL 现场读、现场算；
  * 无法由本节点数据复算的跨节点结论保留原文，并显式标注来源节点；
  * 幂等：重复执行结果一致。

用法：
    python3 scripts/patch_report25_conclusion.py
"""
import json
import os
import re
from pathlib import Path

WS = Path(os.environ.get("ROBO_WS", "/root/private_data/workspace/robo"))

CAND_DIRS = ["outputs/fullbench_candidate_20261031_231152"]
GRAPH_BASE_FILES = [
    "outputs/phase5_ab_noneager/noneager_confirm_20261031_222226/ab_4096_c1.jsonl",
    "outputs/phase5_ab_noneager/noneager_confirm_20261031_222226/ab_4096_c8.jsonl",
]
GRAPH_BASE_DIRS = ["outputs/fullbench_baseline_20261101_043223"]
EAGER_FIRST_FILES = [
    "logs/clean_repro_full/clean_repro_full_20261031/bench/round1.jsonl",
    "logs/clean_repro_full/clean_repro_full_20261031/bench/round2.jsonl",
]

ORDER = [(4096, 1), (4096, 8), (4096, 64), (16384, 1), (16384, 8), (16384, 64)]
TIER_NAME = {(4096, 1): "4096/并发1", (4096, 8): "4096/并发8", (4096, 64): "4096/并发64",
             (16384, 1): "16384/并发1", (16384, 8): "16384/并发8", (16384, 64): "16384/并发64"}


def load(paths):
    """返回 {(ilen, conc): [row, ...]}，同一文件内同档位只取第一次出现。"""
    rows = {}
    for rel in paths:
        if "*" in rel:
            files = sorted(WS.glob(rel))
        else:
            p = WS / rel
            if p.is_dir():
                # 目录：取其下的 *.jsonl
                files = sorted(p.glob("*.jsonl"))
            else:
                files = [p] if p.exists() else []
        for f in files:
            seen = set()
            for line in f.read_text(encoding="utf-8").splitlines():
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


def fmt_pct(b, c):
    return f"{(c / b - 1) * 100:+.2f}%"


def deltas(base_rows, cand_rows):
    """返回 {tier: (base_tp, cand_tp, pct)}，只保留两边都有数据的档位。"""
    out = {}
    for key in ORDER:
        b, c = base_rows.get(key), cand_rows.get(key)
        if not b or not c:
            continue
        bt, ct = mean(b, "output_throughput"), mean(c, "output_throughput")
        if bt <= 0:
            continue
        out[key] = (bt, ct, fmt_pct(bt, ct))
    return out


def pick(d, tiers):
    parts = []
    for t in tiers:
        if t in d:
            parts.append(f"{TIER_NAME[t]} **{d[t][2]}**")
    return "、".join(parts)


def replace_row(text: str, prefix_regex: str, new_row: str, label: str) -> str:
    pat = re.compile(prefix_regex, re.M)
    if not pat.search(text):
        raise SystemExit(f"ERROR: 未找到待替换行：{label}")
    out, n = pat.subn(lambda _m: new_row, text, count=1)
    if n != 1:
        raise SystemExit(f"ERROR: 替换次数异常 {n}：{label}")
    return out


def main():
    cand = load(CAND_DIRS)
    gbase = load(GRAPH_BASE_FILES + GRAPH_BASE_DIRS)
    eager = load(EAGER_FIRST_FILES)
    if not cand:
        raise SystemExit("ERROR: 候选压测数据缺失")
    if not gbase:
        raise SystemExit("ERROR: graph 基线数据缺失")
    if not eager:
        raise SystemExit("ERROR: 首提基线复现数据缺失")

    d_graph = deltas(gbase, cand)
    d_eager = deltas(eager, cand)
    if not d_graph:
        raise SystemExit("ERROR: graph 基线与候选无共同档位")

    tiers_g = [t for t in ORDER if t in d_graph]
    tiers_e = [t for t in ORDER if t in d_eager]

    print("=== graph 基线 → 最优版（同节点） ===")
    for t in tiers_g:
        print(f"  {TIER_NAME[t]}: {d_graph[t][0]:.2f} → {d_graph[t][1]:.2f} tok/s  {d_graph[t][2]}")
    print("=== 首提基线 eager → 最优版（同节点） ===")
    for t in tiers_e:
        print(f"  {TIER_NAME[t]}: {d_eager[t][0]:.2f} → {d_eager[t][1]:.2f} tok/s  {d_eager[t][2]}")

    # ---------------------------------------------------------------- reports/25
    rp = WS / "reports/25_tuning_delivery_20261031.md"
    txt = rp.read_text(encoding="utf-8")

    row_graph = (f"| 相对 graph 基线 | {pick(d_graph, tiers_g)}"
                 f"（2026-10-31 同节点同轮实测，见 §5.2） |")
    row_eager = (f"| 相对首提 eager 基线 | {pick(d_eager, tiers_e)}"
                 f"（2026-10-31 同节点同轮实测，赛事口径正式对比，见 §5.2） |")
    txt = replace_row(txt, r"^\| 相对 graph 基线 \|.*$", row_graph, "§1 graph 基线行")
    txt = replace_row(txt, r"^\| 相对首提 eager 基线 \|.*$", row_eager, "§1 首提基线行")

    # §2 差异表「依据」列：按实测重写归因
    g41 = d_graph.get((4096, 1))
    e41 = d_eager.get((4096, 1))
    g48 = d_graph.get((4096, 8))
    e48 = d_eager.get((4096, 8))

    if e41 and g41:
        # g41 = (graph 基线吞吐, 最优版吞吐, 提升%)；e41 = (eager 基线吞吐, 最优版吞吐, 提升%)
        eff = (g41[0] / e41[0] - 1) * 100
        c1_basis = (f"graph（torch.compile + graph capture）：同节点 4096/并发1 "
                    f"首提 eager {e41[0]:.2f} → graph 基线 {g41[0]:.2f} tok/s（{eff:+.1f}%），"
                    f"单项收益有限；叠加调优库与 aiter FA 后到 {g41[1]:.2f} tok/s，"
                    f"主体收益来自第 2、3 项")
    else:
        c1_basis = "graph（torch.compile + graph capture），收益见 §5.2 同节点 A/B"

    if g41:
        c2_basis = (f"decode 形状 N=1~64 的 GEMM 换成 GEMV 专用 kernel"
                    f"（算子级 107 size 中位 +19.1%；10-30 单独隔离 E5：4096/并发1 +23.8%）；"
                    f"同节点 graph 基线 → 最优版 4096/并发1 {g41[2]}、"
                    + (f"4096/并发8 {g48[2]}" if g48 else "4096/并发8 —"))
    else:
        c2_basis = "GEMV 专用 kernel，算子级 107 size 中位 +19.1%"

    c3_basis = ("16384 长上下文 decode 走 aiter FA kernel"
                "（10-30 同轮 E7 gate：16384/并发1 +17.9%，TPOT 34.66 → 28.85 ms）；"
                "本节点未单独隔离此项，未计入上表")

    txt = replace_row(
        txt, r"^\| 1 \| 去掉 `--enforce-eager` \|.*$",
        f"| 1 | 去掉 `--enforce-eager` | {c1_basis} |", "§2 差异表第 1 行")
    txt = replace_row(
        txt, r"^\| 2 \| `ROCBLAS_TENSILE_LIBPATH`.*$",
        f"| 2 | `ROCBLAS_TENSILE_LIBPATH` → 工作目录内的调优库 | {c2_basis} |", "§2 差异表第 2 行")
    txt = replace_row(
        txt, r"^\| 3 \| `VLLM_ATTENTION_BACKEND=ROCM_AITER_FA`.*$",
        f"| 3 | `VLLM_ATTENTION_BACKEND=ROCM_AITER_FA` | {c3_basis} |", "§2 差异表第 3 行")

    rp.write_text(txt, encoding="utf-8")
    print(f"[patch] updated {rp}")

    # --------------------------------------------------------- submission/README
    sp = WS / "submission/README.md"
    txt = sp.read_text(encoding="utf-8")
    if e41 and g41:
        eff = (g41[0] / e41[0] - 1) * 100
        readme_c1 = (f"启用 torch.compile + graph capture（同节点 4096/并发1 "
                     f"首提 eager {e41[0]:.2f} → graph 基线 {g41[0]:.2f} tok/s，{eff:+.1f}%；"
                     f"单项收益有限，主体收益来自 GEMM 调优库与 aiter FA）")
    else:
        readme_c1 = "启用 torch.compile + graph capture（收益见 ../reports/25 §5.2）"
    txt = replace_row(
        txt, r"^\| 去掉 `--enforce-eager` \| `env\.sh` 的 `EXTRA_ARGS` \|.*$",
        f"| 去掉 `--enforce-eager` | `env.sh` 的 `EXTRA_ARGS` | {readme_c1} |",
        "submission/README §3.5 第 1 行")
    # 跨节点的 +17.9%（aiter FA）必须标明来源节点，避免被当成本节点实测
    txt = txt.replace(
        "（16384 输入档 +17.9%）",
        "（10-30 同轮 E7 gate 隔离测得：16384/并发1 +17.9%；本节点未单独隔离该项）")
    sp.write_text(txt, encoding="utf-8")
    print(f"[patch] updated {sp}")

    # ------------------------------------------------------------- env.sh 注释
    ep = WS / "submission/env.sh"
    txt = ep.read_text(encoding="utf-8")
    if e41 and g41:
        eff = (g41[0] / e41[0] - 1) * 100
        env_comment = (f"# 最优版**不再**带 --enforce-eager：启用 torch.compile + graph capture 后，\n"
                       f"# 4096/并发1 输出吞吐 首提 eager {e41[0]:.2f} → graph 基线 {g41[0]:.2f} tok/s"
                       f"（{eff:+.1f}%，同节点实测）。\n"
                       f"# 单看 graph 的收益有限，主体收益来自 GEMM 调优库与 aiter FA（见 reports/25 §2）。")
        txt = re.sub(r"# 最优版\*\*不再\*\*带 --enforce-eager：.*?# 作为保护。",
                     env_comment.replace("\\", "\\\\") + "\n# 2026-10-27 首次非 eager 启动曾在 inductor 编译期触顶容器 cgroup 上限被 OOM killer\n"
                     "# 杀掉；当前实测启动稳定（ready_after 170~240 s），run.sh 仍保留 70 GiB 内存看门狗\n# 作为保护。",
                     txt, count=1, flags=re.S)
        # 跨节点的 +17.9%（aiter FA）标明来源节点
        txt = txt.replace(
            "（16384 输入档 +17.9%）",
            "（10-30 同轮 E7 gate 隔离测得：16384/并发1 +17.9%；本节点未单独隔离该项）")
        ep.write_text(txt, encoding="utf-8")
        print(f"[patch] updated {ep}")

    # ------------------------------------------------- reports/05_final.md §4.1
    # 提交配置已切换为最优版，最终报告的「提交口径压测表」应展示最优版，
    # 首提基线表保留在下方并明确标注。
    fp = WS / "reports/05_final.md"
    ftxt = fp.read_text(encoding="utf-8")

    best_block = "\n".join([
        "<!-- BEST_BENCH_TABLE:BEGIN -->",
        "**最优版（当前提交配置）六档实测**",
        "",
        official_table(cand),
        "",
        "> 原始数据 `outputs/fullbench_candidate_20261031_231152/`，"
        "对比与逐轮波动见 `reports/25_tuning_delivery_20261031.md` §4 与 `reports/tuning_comparison.md`。",
        "<!-- BEST_BENCH_TABLE:END -->",
    ])
    # §4.1 的口径段原本写的是首提基线（带 --enforce-eager），与「最优版在上」矛盾。
    # 这里把「口径段起点 → FINAL_BENCH_TABLE 锚点」整段替换为确定内容，
    # 因此无论跑多少次结果都一致（幂等）。
    new_kj = ("口径（**最优版 = 当前提交配置**：graph 非 eager + GEMM 调优库 + `ROCM_AITER_FA`）：\n"
              "单卡 K100AI / vLLM 0.15.1 / BF16 / `--max-model-len 32768` /\n"
              "`--gpu-memory-utilization 0.85` / `--max-num-batched-tokens 4096` / `--max-num-seqs 64` /\n"
              "`--no-enable-prefix-caching`；客户端为官方指定的 `sglang.bench_serving`，\n"
              "每请求 1 张 1080p 图，`--seed 1`，请求数 `max(8, 2×并发)`，输出实际打满 1024 token。")
    baseline_head = ("**首提基线（eager，冻结基准）同口径原始结果**"
                     "（口径同上，仅多 `--enforce-eager`、stock rocBLAS 库、默认 attention）：")

    m = re.search(r"(?m)^口径(?:：单卡 K100AI|（\*\*最优版)", ftxt)
    anchor = "<!-- FINAL_BENCH_TABLE -->"
    if not m or anchor not in ftxt:
        raise SystemExit("ERROR: reports/05_final.md 找不到 §4.1 口径段或 FINAL_BENCH_TABLE 锚点")
    a = ftxt.index(anchor)
    canonical = new_kj + "\n\n" + best_block + "\n\n" + baseline_head + "\n\n"
    ftxt = ftxt[:m.start()] + canonical + ftxt[a:]
    fp.write_text(ftxt, encoding="utf-8")
    print(f"[patch] updated {fp}")


def official_table(cand_rows):
    cols = ["random_input_len", "random_output_len", "max_concurrency", "completed", "duration",
            "request_throughput", "output_throughput", "total_throughput",
            "mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms", "mean_e2e_latency_ms"]
    header = ("| random_input_len | random_output_len | max_concurrency | completed | duration | "
              "request_throughput | output_throughput | total_throughput | mean_ttft_ms | "
              "p99_ttft_ms | mean_tpot_ms | p99_tpot_ms | mean_e2e_latency_ms |")
    lines = [header, "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key in ORDER:
        rs = cand_rows.get(key)
        if not rs:
            continue
        r = rs[-1]
        vals = [fmt_val(r.get(c)) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def fmt_val(v):
    """官方 13 列表的数值格式：整数列原样，浮点列保留 2 位小数，不用科学计数法。"""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


if __name__ == "__main__":
    main()
