#!/usr/bin/env python3
"""把完整压测结果回填到报告与提交包 README 的占位符位置。

回填位置：
  reports/25_tuning_delivery_20261031.md
      <!-- FULL_BENCH_TABLE -->    ← 最优版六档表（官方 13 列）
      <!-- SAME_NODE_AB -->        ← 同卡 A/B（graph 基线 vs 最优版）
      <!-- SAME_NODE_AB_EAGER -->  ← 同卡 A/B（首提基线 eager vs 最优版）
  submission/README.md
      <!-- BENCH_SUMMARY:BEGIN --> ... <!-- BENCH_SUMMARY:END -->  ← 换成最优版表
      并在其后追加「与基线对比」小节（<!-- TUNING_COMPARISON -->）

★ 同卡口径（scripts/bench_set.py）
  容器被平台回收重建后物理卡会更换，跨卡绝对值不可比。本脚本**只把记录了
  同一 card_id 的运行目录**当作同卡对照；老目录（无 card_id）一律不参与同卡 A/B。
  没有同卡对照数据时，相应小节写明「未测」，绝不用跨卡数据顶替。

用法：
    python3 scripts/fill_tuning_tables.py
    python3 scripts/fill_tuning_tables.py --run-dir outputs/fullbench_candidate_xxx \
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
_BASE = Path(__file__).resolve().parent.parent


def _resolve(rel):
    """解析相对路径到实际存在的位置。

    优先字面路径 <base>/rel（节点工作目录布局：submission/...），
    再回落去掉 ``submission/`` 前缀（仓库提交视图布局：提交包已上提到根）。
    保证节点侧行为不变，克隆后的仓库也能跑。
    """
    rel = str(rel)
    p = _BASE / rel
    if p.exists():
        return p
    if rel.startswith("submission/"):
        alt = _BASE / rel[len("submission/"):]
        if alt.exists():
            return alt
    return p


WS = bench_set.WS
ORDER = bench_set.ORDER
TIER_NAME = bench_set.TIER_NAME

# 跨卡存档（历史实验，永不参与同卡相减）
ARCHIVE_FIRST = [str(_resolve("submission/bench/round1.jsonl")),
                  str(_resolve("submission/bench/round2.jsonl"))]

COLS = [
    "random_input_len", "random_output_len", "max_concurrency", "completed", "duration",
    "request_throughput", "output_throughput", "total_throughput",
    "mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms", "mean_e2e_latency_ms",
]
# 表头与分隔行**从 COLS 派生**：早期版本用字符串字面量拼接，拼接处多出一个 "|"
# → 表头 15 列（2 个空单元格）而数据行 13 列，Markdown 表格渲染失败。
HEADER = "| " + " | ".join(COLS) + " |"
SEPARATOR = "| " + " | ".join(["---"] * 3 + ["---:"] * (len(COLS) - 3)) + " |"



def check_table_shape(lines):
    """自检：表头 / 分隔行 / 数据行的列数必须一致，否则 Markdown 渲染会退化成纯文本。"""
    widths = [len(l.split("|")) - 2 for l in lines if l.strip().startswith("|")]
    if len(set(widths)) > 1:
        raise SystemExit("ERROR: 表格列数不一致 widths=%s\n%s" % (widths, "\n".join(lines)))
    return True


def load_dir(d):
    """读取一个目录下的 *.jsonl，按 (输入长度, 并发) 聚合（同一文件内同档位只取第一次）。"""
    rows = {}
    d = Path(d)
    if not d.is_dir():
        return rows
    for f in sorted(d.glob("*.jsonl")):
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


def missing_cells_note(rows):
    """若某些档位缺少官方 13 列中的字段，生成明确的缺口说明，避免把 '—' 误读成 0。"""
    gaps = {}
    for key in ORDER:
        rs = rows.get(key)
        if not rs:
            continue
        r = rs[-1]
        miss = [c for c in COLS if r.get(c) is None]
        if miss:
            gaps[key] = miss
    if not gaps:
        return []
    lines = [
        "> **缺口说明**：`—` 表示该字段在本次数据来源中未被留存，**不是 0**。"
        "以下档位存在未留存字段：",
        "",
    ]
    for key, miss in gaps.items():
        lines.append(f"> - {TIER_NAME.get(key, key)}：{', '.join(miss)}")
    lines.append(">")
    lines.append("> 原因与影响见 §9；吞吐、TTFT、TPOT、时长、完成数、失败数等核心指标齐全。")
    return lines


def fmt_official(rows):
    lines = [HEADER, SEPARATOR]
    for key in ORDER:
        rs = rows.get(key)
        if not rs:
            lines.append(f"| {key[0]} | 1024 | {key[1]} | - | - | - | - | - | - | - | - | - | - |")
            continue
        r = rs[-1]  # 记录轮（最后一轮）
        vals = [fmt_val(r.get(c)) for c in COLS]
        lines.append("| " + " | ".join(vals) + " |")
    check_table_shape(lines)
    return "\n".join(lines)


def fmt_rounds(rows):
    lines = ["| 输入 | 并发 | 轮数 | 输出吞吐 tok/s（各轮） | 轮间极差 | 失败 |",
             "| --- | ---: | ---: | --- | ---: | ---: |"]
    for key in ORDER:
        rs = rows.get(key)
        if not rs:
            continue
        tp = [r.get("output_throughput", 0) for r in rs]
        spread = (max(tp) - min(tp)) / min(tp) * 100 if len(tp) > 1 and min(tp) else 0.0
        failed = sum(r.get("failed", 0) for r in rs)
        lines.append(f"| {key[0]} | {key[1]} | {len(rs)} | {', '.join(f'{v:.2f}' for v in tp)} | "
                     f"{spread:.2f}% | {failed} |")
    check_table_shape(lines)
    return "\n".join(lines)


def _mean(rs, field):
    return sum(r.get(field, 0) or 0 for r in rs) / len(rs) if rs else 0.0


def fmt_ab(base_rows, cand_rows, base_label, cand_label):
    lines = [f"| 输入 | 并发 | {base_label} tok/s | {cand_label} tok/s | 吞吐提升 | TTFT 变化 | TPOT 变化 | 基线轮/候选轮 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for key in ORDER:
        b, c = base_rows.get(key), cand_rows.get(key)
        if not b or not c:
            continue
        bt, ct = _mean(b, "output_throughput"), _mean(c, "output_throughput")
        btt, ctt = _mean(b, "mean_ttft_ms"), _mean(c, "mean_ttft_ms")
        btp, ctp = _mean(b, "mean_tpot_ms"), _mean(c, "mean_tpot_ms")
        if bt <= 0 or btt <= 0 or btp <= 0:
            continue
        lines.append(
            f"| {key[0]} | {key[1]} | {bt:.2f} | {ct:.2f} | {(ct/bt-1)*100:+.2f}% | "
            f"{(ctt/btt-1)*100:+.1f}% | {(ctp/btp-1)*100:+.1f}% | {len(b)}/{len(c)} |")
    check_table_shape(lines)
    return "\n".join(lines)


NOT_MEASURED = ("（当前卡上无同卡对照数据，未做同卡 A/B；"
                "跨卡历史数据见 `reports/tuning_comparison.md` 的存档段）")


def replace_marker(text: str, name: str, body: str) -> str:
    """在成对标记之间替换正文（标记本身保留，脚本可重复执行）。"""
    begin = f"<!-- {name}:BEGIN -->"
    end = f"<!-- {name}:END -->"
    if begin not in text or end not in text:
        raise SystemExit(f"ERROR: marker pair not found: {name}")
    pat = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.S)
    block = f"{begin}\n\n{body}\n\n{end}"
    return pat.sub(lambda _m: block, text, count=1)


def replace_block(text: str, body: str) -> str:
    pat = re.compile(r"<!-- BENCH_SUMMARY:BEGIN -->.*?<!-- BENCH_SUMMARY:END -->", re.S)
    if not pat.search(text):
        raise SystemExit("ERROR: BENCH_SUMMARY block not found")
    return pat.sub(f"<!-- BENCH_SUMMARY:BEGIN -->\n{body}\n<!-- BENCH_SUMMARY:END -->", text, count=1)


def replace_cmp(text: str, body: str) -> str:
    pat = re.compile(r"<!-- TUNING_COMPARISON:BEGIN -->.*?<!-- TUNING_COMPARISON:END -->", re.S)
    block = f"<!-- TUNING_COMPARISON:BEGIN -->\n\n{body}\n\n<!-- TUNING_COMPARISON:END -->"
    if pat.search(text):
        return pat.sub(lambda _m: block, text, count=1)
    anchor = "<!-- BENCH_SUMMARY:END -->"
    if anchor not in text:
        raise SystemExit("ERROR: BENCH_SUMMARY:END not found")
    return text.replace(anchor, anchor + "\n\n" + block, 1)


def resolve_dirs(args):
    """决定三套配置各用哪个目录。

    规则（scripts/bench_set.py）：
      * 显式参数优先；
      * 否则取「同一张卡」上最新的一份；当前卡凑不齐时回退到历史同节点会话，
        并在表里标注该会话标签；
      * **绝不跨会话配对**（A 侧和 B 侧必须来自同一张卡/同一段测量窗口）。
    """
    card = bench_set.current_card_id()

    if args.run_dir:
        # 同一配置可能被拆成多次服务实例跑（重档单独一轮），
        # 因此 --run-dir 允许**逗号分隔的多个目录**，全部参与按档位合并。
        cand_dirs = [Path(p) for p in str(args.run_dir).split(",") if p.strip()]
        cand_dir = cand_dirs[0] if cand_dirs else None
        cand_label = args.node_label
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

    return cand_dir, cand_label, cand_dirs, base_dirs, base_label, first_dirs, first_label, card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="",
                    help="最优版运行目录（逗号分隔，可多个；默认取同卡全部服务实例并按档位合并）")
    ap.add_argument("--baseline-dirs", default="", help="graph 基线目录（逗号分隔；默认取同卡最新一份）")
    ap.add_argument("--first-dirs", default="", help="首提基线 eager 目录（逗号分隔；默认取同卡最新一份）")
    ap.add_argument("--report", type=Path, default=WS / "reports/25_tuning_delivery_20261031.md")
    ap.add_argument("--readme", type=Path, default=_resolve("submission/README.md"))
    ap.add_argument("--node-label", default=os.environ.get("BENCH_NODE_LABEL", "当前节点（同一张卡）"))
    args = ap.parse_args()

    cand_dir, cand_label, cand_dirs, base_dirs, base_label, first_dirs, first_label, card = resolve_dirs(args)
    if cand_dir is None:
        raise SystemExit("ERROR: 找不到最优版运行目录（同卡或显式 --run-dir）")
    print(f"[fill] card_id      : {card or '（未知）'}")
    print(f"[fill] candidate    : {[str(d) for d in cand_dirs]}  [{cand_label}]")
    print(f"[fill] baseline dirs: {[str(d) for d in base_dirs]}  [{base_label}]")
    print(f"[fill] eager dirs   : {[str(d) for d in first_dirs]}  [{first_label}]")

    # 同一配置可能被拆成多次服务实例跑（重档单独一轮），按档位合并全部目录
    cand = {}
    for d in cand_dirs:
        for k, v in load_dir(d).items():
            cand.setdefault(k, []).extend(v)
    if not cand:
        raise SystemExit(f"ERROR: no result rows in {cand_dirs}")
    base = {}
    for d in base_dirs:
        for k, v in load_dir(d).items():
            base.setdefault(k, []).extend(v)
    first = {}
    for d in first_dirs:
        for k, v in load_dir(d).items():
            first.setdefault(k, []).extend(v)

    official = fmt_official(cand)
    rounds = fmt_rounds(cand)

    # --- 归因（graph → 最优版）必须同卡 ---------------------------------------
    # 换卡后若没在新卡上补跑 graph 基线，graph 基线落在历史会话里，
    # 此时**不能**拿历史会话的 graph 去减当前卡的最优版（跨卡相减）。
    # 正确做法：归因整段改用 graph 基线所在会话自己的三套数据，并在表头标注。
    _sessions = bench_set.sessions()

    def _rows(label, cfg):
        for lbl, cfgmap in _sessions:
            if lbl == label and cfg in cfgmap:
                return bench_set.load_dirs(cfgmap[cfg])
        return {}

    attr_label = cand_label if base_label == cand_label else base_label
    attr_cand = _rows(attr_label, "candidate") or cand
    attr_graph = _rows(attr_label, "baseline") or base
    if attr_label == cand_label:
        ab_base_label, ab_cand_label = "graph 基线", "最优版"
    else:
        ab_base_label = "graph 基线（2026-10-31 节点存档）"
        ab_cand_label = "最优版（2026-10-31 节点存档）"

    ab = fmt_ab(attr_graph, attr_cand, ab_base_label, ab_cand_label) if attr_graph else NOT_MEASURED
    ab_eager = fmt_ab(first, cand, "首提基线 eager", "最优版") if first else NOT_MEASURED

    full_block = "\n".join([
        f"**最优版（提交配置）六档实测结果**（{cand_label}；官方 13 列格式，取记录轮）：",
        "",
        official,
        "",
    ] + missing_cells_note(cand) + [
        "各档轮次与波动：",
        "",
        rounds,
        "",
        f"> 原始数据：" + "、".join(f"`outputs/{Path(d).name}/`" for d in cand_dirs) +
        "（每档 `round*.jsonl` 与同名 `.log`；重档单独起服务实例时按档位合并）；",
        "> 驱动脚本 `scripts/run_full_bench.sh`，汇总 `scripts/summarize_full_bench.py`。",
        "> 本次运行的卡指纹见该目录 `run_config.txt` 的 `card_id=`；",
        "> **跨节点的绝对值不可直接比较**（容器重建会换卡），见 §6 换卡说明。",
    ])

    # --- reports/25 ---
    txt = args.report.read_text(encoding="utf-8")
    sections_before = txt.count("\n## ")
    txt = replace_marker(txt, "FULL_BENCH_TABLE", full_block)
    txt = replace_marker(txt, "SAME_NODE_AB", ab)
    txt = replace_marker(txt, "SAME_NODE_AB_EAGER", ab_eager)
    sections_after = txt.count("\n## ")
    if sections_after < sections_before:
        raise SystemExit(
            f"ERROR: 章节数减少（{sections_before} -> {sections_after}），已中止写入。"
            "请检查 replace_marker 的正则是否越界。")
    args.report.write_text(txt, encoding="utf-8")
    print(f"[fill] updated {args.report} (sections {sections_before} -> {sections_after})")

    # --- submission/README.md ---
    txt = args.readme.read_text(encoding="utf-8")
    txt = replace_block(txt, official)
    cmp_lines = [
        "### 7.1 与基线的同口径对比",
        "",
        f"**同卡 A/B：首提基线 eager → 最优版**（{first_label}，同一负载、同一客户端；"
        "赛事口径的正式对比）",
        "",
        ab_eager,
        "",
        f"**同卡 A/B：graph 基线 → 最优版**（{base_label}，只开 graph、不带任何调优开关的对照；"
        "用于把收益拆成「graph」与「调优库 + aiter FA」两部分）",
        "",
        ab,
        "",
        "> 首提基线在 **2026-10-27 节点** 的原始六档记录保留在 `bench/round1.jsonl` /",
        "> `bench/round2.jsonl`（官方口径基准，见第 8 节的节点差异说明）；",
        "> 最优版的原始数据在 `bench/tuned/`。逐档对比表见 `reports/tuning_comparison.md`，",
        "> 完整交付报告见 `reports/25_tuning_delivery_20261031.md`。",
        "> 容器重建会换物理卡，跨卡绝对值不可比，上表只含**同一张卡**上的数据。",
    ]
    txt = replace_cmp(txt, "\n".join(cmp_lines))
    args.readme.write_text(txt, encoding="utf-8")
    print(f"[fill] updated {args.readme}")

    print("\n=== 同卡 A/B：graph 基线 → 最优版 ===")
    print(ab)
    print("\n=== 同卡 A/B：首提基线 eager → 最优版 ===")
    print(ab_eager)


if __name__ == "__main__":
    main()
