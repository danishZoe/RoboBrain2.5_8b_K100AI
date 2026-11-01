#!/usr/bin/env python3
"""按原始 JSONL 数据重写 reports/25 的结论数值与差异表「依据」列。

背景：
  * 容器被平台回收重建后**物理卡会更换**，跨卡绝对值不可比；
    本脚本只使用**同一张卡**上的三套数据（scripts/bench_set.py 判定），
    没有同卡对照时如实写「未测」，绝不用跨卡数据顶替。
  * §2 差异表第 1 行曾把「graph 基线 → 调优库」的 TPOT 变化误挂在
    「去掉 --enforce-eager」上，这里按实测重写归因。

本脚本只做**可由原始数据复算**的替换：
  * 所有百分比、tok/s、TTFT/TPOT 都从 JSONL 现场读、现场算；
  * 无法由本卡数据复算的跨节点结论保留原文，并显式标注来源节点；
  * 幂等：重复执行结果一致。

用法：
    python3 scripts/patch_report25_conclusion.py
    python3 scripts/patch_report25_conclusion.py --run-dir outputs/fullbench_candidate_xxx \
        --baseline-dirs outputs/fullbench_baseline_yyy --first-dirs outputs/fullbench_eager_zzz
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_set  # noqa: E402

WS = bench_set.WS
ORDER = bench_set.ORDER
TIER_NAME = bench_set.TIER_NAME
LEGACY_LABEL = bench_set.LEGACY_LABEL


def replace_marker(text: str, name: str, body: str) -> str:
    """在成对标记之间替换正文（标记本身保留，脚本可重复执行）。"""
    begin = f"<!-- {name}:BEGIN -->"
    end = f"<!-- {name}:END -->"
    if begin not in text or end not in text:
        raise SystemExit(f"ERROR: marker pair not found: {name}")
    pat = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.S)
    block = f"{begin}\n{body}\n{end}"
    return pat.sub(lambda _m: block, text, count=1)

# 跨卡存档（历史实验，永不参与同卡相减）
ARCHIVE_FIRST = ["submission/bench/round1.jsonl", "submission/bench/round2.jsonl"]
ARCHIVE_GRAPH30 = ["tmp/graph_bench/baseline_oldlib/round1.jsonl",
                    "tmp/graph_bench/baseline_oldlib/round2.jsonl"]
ARCHIVE_E7 = ["tmp/gate_aiterfa.jsonl"]


def load(paths):
    """返回 {(ilen, conc): [row, ...]}，同一文件内同档位只取第一次出现。"""
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None, help="最优版运行目录（默认取同卡最新一份）")
    ap.add_argument("--baseline-dirs", default="", help="graph 基线目录（逗号分隔）")
    ap.add_argument("--first-dirs", default="", help="首提基线 eager 目录（逗号分隔）")
    ap.add_argument("--node-label", default=os.environ.get("BENCH_NODE_LABEL", "当前节点（同一张卡）"))
    args = ap.parse_args()

    card = bench_set.current_card_id()

    if args.run_dir:
        cand_dirs = [args.run_dir]
        cand_dir, cand_label = args.run_dir, args.node_label
    else:
        cand_label, cand_dirs = bench_set.pick_single("candidate")
        cand_dir = cand_dirs[0] if cand_dirs else None

    if args.baseline_dirs:
        base_dirs = [Path(p) for p in args.baseline_dirs.split(",") if p.strip()]
        base_label = args.node_label
    else:
        base_label, base_dirs, _ = bench_set.pick_pair("baseline", "candidate")

    if args.first_dirs:
        first_dirs = [Path(p) for p in args.first_dirs.split(",") if p.strip()]
        first_label = args.node_label
    else:
        first_label, first_dirs, _ = bench_set.pick_pair("eager", "candidate")

    if cand_dir is None:
        raise SystemExit("ERROR: 找不到最优版运行目录（同卡或显式 --run-dir）")

    # 同一配置被拆成多次服务实例跑时，按档位合并全部同卡目录
    cand = bench_set.load_dirs(cand_dirs)
    gbase = bench_set.load_dirs(base_dirs)
    eager = bench_set.load_dirs(first_dirs)

    print(f"[patch] card_id       : {card or '（未知）'}")
    print(f"[patch] candidate     : {[str(d) for d in cand_dirs]}  [{cand_label}]")
    print(f"[patch] baseline dirs : {[str(d) for d in base_dirs]}  [{base_label}]")
    print(f"[patch] eager dirs    : {[str(d) for d in first_dirs]}  [{first_label}]")

    if not cand:
        raise SystemExit("ERROR: 候选压测数据缺失")
    if not gbase:
        print("[patch] WARN: 当前卡上没有同卡 graph 基线数据，§2 归因只能给部分结论")
    if not eager:
        print("[patch] WARN: 当前卡上没有同卡首提基线数据，§1 赛事口径对比行将标注未测")

    # --- 归因必须三套同卡 ---------------------------------------------------
    # 换卡后若没在新卡上补跑 graph 基线，graph 基线会落在历史会话里。此时
    # **绝不能**拿历史会话的 graph 去减当前卡的最优版 —— 那是跨卡相减，
    # 正是换卡后最容易产生的错误结论。正确做法：归因整段改用 graph 基线
    # 所在会话自己的三套数据（同一张卡），并把该会话标签写进结论文字。
    _sessions = bench_set.sessions()

    def _rows(label, cfg):
        for lbl, cfgmap in _sessions:
            if lbl == label and cfg in cfgmap:
                return bench_set.load_dirs(cfgmap[cfg])
        return {}

    attr_label = cand_label if base_label == cand_label else base_label
    attr_cand = _rows(attr_label, "candidate") or cand
    attr_eager = _rows(attr_label, "eager")
    attr_graph = _rows(attr_label, "baseline") or gbase
    if attr_label == cand_label:
        attr_note = "同卡同轮实测"
        print(f"[patch] 归因会话 = {attr_label}（同卡三套齐全）")
    else:
        attr_note = "2026-10-31 节点同卡实测（换卡后未重跑 graph 基线，属跨卡存档）"
        print(f"[patch] 归因会话 = {attr_label}（当前卡未补跑 graph 基线，归因整段用该会话三套数据）")

    d_graph = deltas(attr_graph, attr_cand)        # 归因：graph → 最优版（同卡）
    d_eager_attr = deltas(attr_eager, attr_cand)  # 归因：eager → 最优版（同卡）
    d_eager = deltas(eager, cand)                 # 主体结论：当前卡 eager → 最优版（同卡）

    tiers_g = [t for t in ORDER if t in d_graph]
    tiers_e = [t for t in ORDER if t in d_eager]

    print("=== 同卡 graph 基线 → 最优版 ===")
    for t in tiers_g:
        print(f"  {TIER_NAME[t]}: {d_graph[t][0]:.2f} → {d_graph[t][1]:.2f} tok/s  {d_graph[t][2]}")
    print("=== 同卡 首提基线 eager → 最优版 ===")
    for t in tiers_e:
        print(f"  {TIER_NAME[t]}: {d_eager[t][0]:.2f} → {d_eager[t][1]:.2f} tok/s  {d_eager[t][2]}")

    # ---------------------------------------------------------------- reports/25
    rp = WS / "reports/25_tuning_delivery_20261031.md"
    txt = rp.read_text(encoding="utf-8")

    if tiers_g:
        row_graph = (f"| 相对 graph 基线 | {pick(d_graph, tiers_g)}"
                     f"（{attr_note}，见 §5.2） |")
    else:
        row_graph = "| 相对 graph 基线 | **未测**（当前卡上没有同卡 graph 基线数据） |"
    if tiers_e:
        row_eager = (f"| 相对首提 eager 基线 | {pick(d_eager, tiers_e)}"
                     f"（{first_label}，同卡同轮实测，赛事口径正式对比，见 §5.3） |")
    else:
        row_eager = "| 相对首提 eager 基线 | **未测**（当前卡上没有同卡 eager 基线数据） |"
    txt = replace_row(txt, r"^\| 相对 graph 基线 \|.*$", row_graph, "§1 graph 基线行")
    txt = replace_row(txt, r"^\| 相对首提 eager 基线 \|.*$", row_eager, "§1 首提基线行")

    # §2 差异表「依据」列：按实测重写归因
    # 注意：eager→graph 这一步必须用**归因会话自己的** eager，
    # 不能拿当前卡的 eager 去减历史会话的 graph（跨卡相减）。
    g41 = d_graph.get((4096, 1))
    e41 = d_eager.get((4096, 1))
    e41_attr = d_eager_attr.get((4096, 1))
    g48 = d_graph.get((4096, 8))

    if e41_attr and g41:
        eff = (g41[0] / e41_attr[0] - 1) * 100
        c1_basis = (f"graph（torch.compile + graph capture）：{attr_note}，"
                    f"4096/并发1 首提 eager {e41_attr[0]:.2f} → graph 基线 {g41[0]:.2f} tok/s（{eff:+.1f}%），"
                    f"单项收益有限；叠加调优库与 aiter FA 后到 {g41[1]:.2f} tok/s，"
                    f"主体收益来自第 2、3 项")
    elif g41:
        c1_basis = (f"graph（torch.compile + graph capture）：{attr_note}，"
                    f"4096/并发1 graph 基线 {g41[0]:.2f} → 最优版 {g41[1]:.2f} tok/s（{g41[2]}）；"
                    f"同卡 eager 基线未测，graph 自身收益未能单独隔离")
    else:
        c1_basis = "graph（torch.compile + graph capture），收益见 §5.2 同卡 A/B"

    if g41:
        c2_basis = (f"decode 形状 N=1~64 的 GEMM 换成 GEMV 专用 kernel"
                    f"（算子级 107 size 中位 +19.1%；10-30 单独隔离 E5：4096/并发1 +23.8%）；"
                    f"同卡 graph 基线 → 最优版 4096/并发1 {g41[2]}、"
                    + (f"4096/并发8 {g48[2]}" if g48 else "4096/并发8 —"))
    else:
        c2_basis = "GEMV 专用 kernel，算子级 107 size 中位 +19.1%"

    c3_basis = ("16384 长上下文 decode 走 aiter FA kernel"
                "（10-30 同轮 E7 gate：16384/并发1 +17.9%，TPOT 34.66 → 28.85 ms）；"
                "该数值来自 **10-30 节点**，与本卡数据不同卡，未计入上表")

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

    # ------------------------------- §6 卡片等效性对照表（由实测数据生成）-------
    # 换卡后要回答「旧卡的结论能不能沿用」，靠的是**同配置跨卡对照**：
    # 同一个最优版配置在两张卡上各测一遍，看绝对值差多少。这是**验证手段**，
    # 收益结论本身仍然只用同卡相减。
    legacy_cand = _rows(LEGACY_LABEL, "candidate")
    legacy_graph = _rows(LEGACY_LABEL, "baseline")
    if legacy_cand and cand:
        equiv_rows = []
        for t in ORDER:
            cur = list(cand.get(t) or [])
            old = list(legacy_cand.get(t) or [])
            if not cur or not old:
                continue
            o = sum(float(r.get("output_throughput") or 0) for r in old) / len(old)
            n = sum(float(r.get("output_throughput") or 0) for r in cur) / len(cur)
            if o <= 0:
                continue
            equiv_rows.append((t, o, n, (n - o) / o * 100))
        if equiv_rows:
            lines = [
                "| 档位 | 旧卡最优版（10-31 节点） | 新卡最优版（11-01 换卡后） | 偏差 |",
                "| --- | ---: | ---: | ---: |",
            ]
            for t, o, n, dev in equiv_rows:
                lines.append(f"| {TIER_NAME[t]} | {o:.2f} tok/s | {n:.2f} tok/s | {dev:+.2f}% |")
            mx = max(abs(d) for _, _, _, d in equiv_rows)
            lines.append("")
            lines.append(
                f"最大偏差 **{mx:.2f}%**（{len(equiv_rows)} 个档位）。"
                + ("两卡性能等价，旧卡数据可作为新卡结论的佐证；**收益结论仍只用同卡相减**。"
                   if mx <= 5 else
                   "偏差超过 5%，旧卡数据只能作历史存档，不得用于新卡结论。"))
            block = "\n".join(lines)
            txt = rp.read_text(encoding="utf-8")
            txt = replace_marker(txt, "CARD_EQUIV_TABLE", block)
            rp.write_text(txt, encoding="utf-8")
            print(f"[patch] updated {rp} (§6 卡片等效性对照表，{len(equiv_rows)} 档)")
        else:
            print("[patch] WARN: 无可比档位，§6 卡片等效性对照表未生成")

    # --------------------------------------------------------- submission/README
    sp = WS / "submission/README.md"
    txt = sp.read_text(encoding="utf-8")
    if e41_attr and g41:
        eff = (g41[0] / e41_attr[0] - 1) * 100
        readme_c1 = (f"启用 torch.compile + graph capture（{attr_note}，4096/并发1 "
                     f"首提 eager {e41_attr[0]:.2f} → graph 基线 {g41[0]:.2f} tok/s，{eff:+.1f}%；"
                     f"单项收益有限，主体收益来自 GEMM 调优库与 aiter FA）")
    elif g41:
        readme_c1 = (f"启用 torch.compile + graph capture（{attr_note}，4096/并发1 "
                     f"graph 基线 {g41[0]:.2f} → 最优版 {g41[1]:.2f} tok/s，{g41[2]}）")
    else:
        readme_c1 = "启用 torch.compile + graph capture（收益见 ../reports/25 §5.2）"
    txt = replace_row(
        txt, r"^\| 去掉 `--enforce-eager` \| `env\.sh` 的 `EXTRA_ARGS` \|.*$",
        f"| 去掉 `--enforce-eager` | `env.sh` 的 `EXTRA_ARGS` | {readme_c1} |",
        "submission/README §3.5 第 1 行")
    # 跨节点的 +17.9%（aiter FA）必须标明来源节点，避免被当成本节点实测
    txt = txt.replace(
        "（16384 输入档 +17.9%）",
        "（10-30 同轮 E7 gate 隔离测得：16384/并发1 +17.9%；本卡未单独隔离该项）")
    sp.write_text(txt, encoding="utf-8")
    print(f"[patch] updated {sp}")

    # ------------------------------------------------------------- env.sh 注释
    ep = WS / "submission/env.sh"
    txt = ep.read_text(encoding="utf-8")
    if e41_attr and g41:
        eff = (g41[0] / e41_attr[0] - 1) * 100
        env_comment = (f"# 最优版**不再**带 --enforce-eager：启用 torch.compile + graph capture 后，\n"
                       f"# 4096/并发1 输出吞吐 首提 eager {e41_attr[0]:.2f} → graph 基线 {g41[0]:.2f} tok/s"
                       f"（{eff:+.1f}%，{attr_note}）。\n"
                       f"# 单看 graph 的收益有限，主体收益来自 GEMM 调优库与 aiter FA（见 reports/25 §2）。")
        txt = re.sub(r"# 最优版\*\*不再\*\*带 --enforce-eager：.*?# 作为保护。",
                     env_comment.replace("\\", "\\\\") + "\n# 2026-10-27 首次非 eager 启动曾在 inductor 编译期触顶容器 cgroup 上限被 OOM killer\n"
                     "# 杀掉；当前实测启动稳定（ready_after 170~240 s），run.sh 仍保留 70 GiB 内存看门狗\n# 作为保护。",
                     txt, count=1, flags=re.S)
        # 跨节点的 +17.9%（aiter FA）标明来源节点
        txt = txt.replace(
            "（16384 输入档 +17.9%）",
            "（10-30 同轮 E7 gate 隔离测得：16384/并发1 +17.9%；本卡未单独隔离该项）")
        ep.write_text(txt, encoding="utf-8")
        print(f"[patch] updated {ep}")

    # ------------------------------------------------- reports/05_final.md §4.1
    fp = WS / "reports/05_final.md"
    ftxt = fp.read_text(encoding="utf-8")

    best_block = "\n".join([
        "<!-- BEST_BENCH_TABLE:BEGIN -->",
        "**最优版（当前提交配置）六档实测**",
        "",
        official_table(cand),
        "",
        f"> 原始数据 " + "、".join(f"`outputs/{Path(d).name}/`" for d in cand_dirs) + "，"
        "对比与逐轮波动见 `reports/25_tuning_delivery_20261031.md` §4 与 `reports/tuning_comparison.md`。",
        "<!-- BEST_BENCH_TABLE:END -->",
    ])

    eager_block = "\n".join([
        "<!-- SAME_CARD_EAGER_TABLE:BEGIN -->",
        "**首提基线 eager（同一张卡，赛事「首次提交」口径）六档实测**",
        "",
        official_table(eager) if eager else "（当前卡上没有同卡 eager 基线数据，未测）",
        "",
        (f"> 原始数据 `outputs/{Path(first_dirs[0]).name}/`；"
         f"与最优版的同卡对比见 `reports/tuning_comparison.md` §2.1。")
        if first_dirs else "> 未测：当前卡上没有同卡 eager 基线数据。",
        "<!-- SAME_CARD_EAGER_TABLE:END -->",
    ])

    new_kj = ("口径（**最优版 = 当前提交配置**：graph 非 eager + GEMM 调优库 + `ROCM_AITER_FA`）：\n"
              "单卡 K100AI / vLLM 0.15.1 / BF16 / `--max-model-len 32768` /\n"
              "`--gpu-memory-utilization 0.85` / `--max-num-batched-tokens 4096` / `--max-num-seqs 64` /\n"
              "`--no-enable-prefix-caching`；客户端为官方指定的 `sglang.bench_serving`，\n"
              "每请求 1 张 1080p 图，`--seed 1`，请求数 `max(8, 2×并发)`，输出实际打满 1024 token。\n"
              "**以下「最优版」与「首提基线 eager」两张表来自同一张物理卡**（容器重建会换卡，"
              "跨卡绝对值不可比，见 `reports/25_tuning_delivery_20261031.md` §6）。")
    baseline_head = ("**首提基线（eager，冻结基准，2026-10-27 节点）同口径原始结果**"
                     "（跨卡存档，仅作历史参考；口径同上，仅多 `--enforce-eager`、stock rocBLAS 库、默认 attention）：")

    m = re.search(r"(?m)^口径(?:：单卡 K100AI|（\*\*最优版)", ftxt)
    anchor = "<!-- FINAL_BENCH_TABLE -->"
    if not m or anchor not in ftxt:
        raise SystemExit("ERROR: reports/05_final.md 找不到 §4.1 口径段或 FINAL_BENCH_TABLE 锚点")
    a = ftxt.index(anchor)
    canonical = new_kj + "\n\n" + best_block + "\n\n" + eager_block + "\n\n" + baseline_head + "\n\n"
    ftxt = ftxt[:m.start()] + canonical + ftxt[a:]
    fp.write_text(ftxt, encoding="utf-8")
    print(f"[patch] updated {fp}")


def official_table(cand_rows):
    cols = ["random_input_len", "random_output_len", "max_concurrency", "completed", "duration",
            "request_throughput", "output_throughput", "total_throughput",
            "mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms", "mean_e2e_latency_ms"]
    header = ("| random_input_len | random_output_len | max_concurrency | completed | duration | "
              "| request_throughput | output_throughput | total_throughput | mean_ttft_ms | "
              "| p99_ttft_ms | mean_tpot_ms | p99_tpot_ms | mean_e2e_latency_ms |")
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
