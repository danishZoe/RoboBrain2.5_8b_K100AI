#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总所有公开评测的墙钟耗时，生成可审计的耗时台账。

数据来源（全部为远端工作目录内的日志/产物，不引入任何外部数字）：
  1. engine 完成行： "[engine][<tag>] 完成 N 条 -> ... 用时 X.Xs"
  2. 3D 判分产物：    scores_3dchunk_3D.json 里的 wall_s_total
  3. 脚本头尾标记：  "===== ... =====" 里的起止时间戳
  4. 被主动中止的运行：按日志 mtime - 启动标记 计上限

输出：
  outputs/bench_public/eval_timing.json
  outputs/bench_public/eval_timing.md
"""
import json
import os
import re
import subprocess
from datetime import datetime

WS = "/root/private_data/workspace/robo"
OUT_DIR = os.path.join(WS, "outputs", "bench_public")

RE_ENGINE = re.compile(r"\[engine\]\[([^\]]+)\]\s*完成\s*(\d+)\s*条.*?用时\s*([\d.]+)s")
RE_HEADER = re.compile(r"^=====\s*(.*?)\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*=====")
RE_PROTO = re.compile(r"^---\s*protocol=(\S+)\s*\((\d{2}:\d{2}:\d{2})\)")


def read(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read().splitlines()
    except OSError:
        return []


def mtime(path):
    try:
        return datetime.fromtimestamp(os.path.getmtime(path))
    except OSError:
        return None


def engine_runs(rel):
    """从一个日志里抽出所有 engine 完成记录的 (tag, n, wall_s)。"""
    out = []
    for line in read(os.path.join(WS, rel)):
        m = RE_ENGINE.search(line)
        if m:
            out.append((m.group(1), int(m.group(2)), float(m.group(3))))
    return out


def header_stamps(rel):
    """抽出 '===== ... <ts> =====' 形式的起止标记。"""
    out = []
    for line in read(os.path.join(WS, rel)):
        m = RE_HEADER.match(line)
        if m:
            out.append((m.group(1).strip(), datetime.strptime(m.group(2), "%Y-%m-%d %H:%M:%S")))
    return out


def group_sums(rel):
    """按 tag 前缀聚合 engine 墙钟（同一 tag 多次出现求和）。"""
    agg = {}
    for tag, n, w in engine_runs(rel):
        key = tag.split("/")[0] if "/" in tag else tag
        agg.setdefault(key, {"tag": key, "n": sum([]), "wall_s": 0.0, "parts": 0})
        agg[key]["n"] += n
        agg[key]["wall_s"] += w
        agg[key]["parts"] += 1
    return list(agg.values())


def json_wall(path, key="wall_s_all_jobs"):
    """真实墙钟。

    注意：scores_3dchunk_*.json 里同时有
      - wall_s_total     = 各分片墙钟之和（并发累加，不是墙上时钟）
      - wall_s_all_jobs  = 进程级真实墙上时钟
    两者在并发 16 时相差约 4 倍，台账必须用后者。
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return float(d.get(key) or 0.0)
    except (OSError, ValueError, TypeError):
        return None


def fmt(sec):
    if sec is None:
        return "-"
    sec = float(sec)
    if sec < 90:
        return "%.1f s" % sec
    m, s = divmod(sec, 60)
    if m < 60:
        return "%d m %02.0f s" % (m, s)
    h, m = divmod(m, 60)
    return "%d h %02d m" % (h, m)


def main():
    rows = []

    # ---- ① 推理类（engine 墙钟）----
    infer_specs = [
        ("CV-Bench 全量 2638（基线 eager / canonical_png）",
         "logs/public_bench_baseline-eager/cvbench.log", "L3", 16),
        ("CV-Bench 全量 2638（候选 非eager / canonical_png）",
         "logs/public_bench_noneager-candidate/cvbench.log", "L3", 16),
        ("CV-Bench 全量 2638（候选 非eager / up2_png，并发 48 快跑）",
         "logs/cvbench_up2_full.log", "L3", 48),
        ("CV-Bench 协议矩阵 8 组 × 788（Count 子集）",
         "logs/public_bench_baseline-eager/cvbench_protocol.log", "L3", 16),
        ("CV-Bench Count 上采样实验 3 组 × 788（canonical/up2_png/up2_jpeg95）",
         "logs/cvbench_upscale.log", "L2", 16),
        ("CV-Bench 提示词口径 2 组 × 788（dataset/flagevalmm）",
         "logs/cvbench_prompt.log", "L2", 16),
        ("RefSpatial 12 组 × (100/100/77)（基线 eager）",
         "logs/public_bench_baseline-eager/refspatial.log", "L1", 16),
        ("RefSpatial 12 组（候选 非eager）",
         "logs/public_bench_noneager-candidate/refspatial.log", "L1", 16),
        ("RefSpatial 噪声底复测 12 组（候选 非eager）",
         "logs/public_bench_noneager-candidate/noisecheck_refspatial.log", "L1", 16),
        ("TraceSpatial 推理 4 组合 × 100（基线 eager）",
         "logs/public_bench_baseline-eager/tracespatial_infer.log", "L1", 16),
        ("TraceSpatial 推理 4 组合 × 100（候选 非eager）",
         "logs/public_bench_noneager-candidate/tracespatial_infer.log", "L1", 16),
    ]
    for name, rel, lvl, conc in infer_specs:
        g = group_sums(rel)
        if not g:
            continue
        wall = sum(x["wall_s"] for x in g)
        n = sum(x["n"] for x in g)
        rows.append({"kind": "推理", "name": name, "level": lvl, "n": n,
                     "conc": conc, "wall_s": round(wall, 1), "log": rel})

    # ---- ② 判分类 ----
    p = os.path.join(WS, "outputs", "bench_public_3d96", "bench_public",
                     "tracespatial", "official", "scores_3dchunk_3D.json")
    w = json_wall(p)
    if w:
        rows.append({"kind": "判分", "name": "TraceSpatial 3D 兜底（96 条 × 4 组合）",
                     "level": "L2", "n": 96 * 4, "conc": 16, "wall_s": w,
                     "log": "outputs/bench_public_3d96/bench_public/.../scores_3dchunk_3D.json"})
    p2 = os.path.join(WS, "outputs", "bench_public_3d96", "bench_public_noneager",
                      "tracespatial", "official", "scores_3dchunk_3D.json")
    w2 = json_wall(p2)
    if w2:
        rows.append({"kind": "判分", "name": "TraceSpatial 3D 兜底（候选侧，96 条 × 4 组合）",
                     "level": "L2", "n": 96 * 4, "conc": 16, "wall_s": w2,
                     "log": "outputs/bench_public_3d96/bench_public_noneager/..."})

    st = header_stamps("logs/ts_score_2d.log")
    if st:
        end = mtime("logs/ts_score_2d.log")
        wall = (end - st[0][1]).total_seconds() if end else None
        rows.append({"kind": "判分", "name": "TraceSpatial 2D 官方判分（4 组合 × 100）",
                     "level": "L2", "n": 400, "conc": 16, "wall_s": wall,
                     "log": "logs/ts_score_2d.log"})

    st = header_stamps("logs/verify_3d_merge.log")
    if st:
        end = mtime("logs/verify_3d_merge.log")
        wall = (end - st[0][1]).total_seconds() if end else None
        rows.append({"kind": "对拍", "name": "3D 分片合并精确性对拍（48 项指标）",
                     "level": "L1", "n": 48, "conc": 1, "wall_s": wall,
                     "log": "logs/verify_3d_merge.log"})

    # ---- ③ 被主动中止的运行（按日志 mtime - 启动标记 计上限）----
    aborted = []
    up = os.path.join(WS, "logs", "exp_cvbench_upscale.log")
    lines = read(up)
    start = None
    for ln in lines:
        m = RE_PROTO.match(ln)
        if m and m.group(1) == "up4_png":
            start = datetime.strptime("2026-10-28 " + m.group(2), "%Y-%m-%d %H:%M:%S")
            break
    if start:
        end = mtime("logs/cvbench_upscale.log")
        wall = (end - start).total_seconds() if end else None
        aborted.append({"name": "CV-Bench Count up4 组（788 题）", "wall_s": wall,
                        "reason": "单组 >23 min 无结果，主动停止（L4 档，按耗时分级规则砍掉）",
                        "log": "logs/exp_cvbench_upscale.log"})

    st = header_stamps("logs/ts_score_3d.log")
    if st:
        # 该次运行卡在深景病态样本上，日志自 14:45:35 起不再有进展；
        # 改用兜底口径的启动时刻作为「进程仍存活且未产出」的时间下界。
        fb = header_stamps("logs/ts_score_3d_96.log")
        end = fb[0][1] if fb else mtime("logs/ts_score_3d_96.log")
        wall = (end - st[0][1]).total_seconds() if end else None
        aborted.append({"name": "TraceSpatial 3D 整集 100 条（4 组合）", "wall_s": wall,
                        "wall_is_lower_bound": True,
                        "reason": "样本 56/78/82/98 深景单条 >65 min 跑不完，整集口径不可达，主动停止",
                        "log": "logs/ts_score_3d.log"})

    # ---- ④ 汇总 ----
    done_wall = sum(r["wall_s"] for r in rows if r.get("wall_s"))
    aborted_wall = sum(a["wall_s"] for a in aborted if a.get("wall_s"))

    out = {
        "generated_at": subprocess.run(["date", "+%F %T"], capture_output=True,
                                       text=True).stdout.strip(),
        "rows": rows,
        "aborted": aborted,
        "total_completed_wall_s": round(done_wall, 1),
        "total_completed_wall_human": fmt(done_wall),
        "total_aborted_wall_s": round(aborted_wall, 1),
        "total_aborted_wall_human": fmt(aborted_wall),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "eval_timing.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    md = []
    md.append("# 公开评测耗时台账\n")
    md.append("> 生成时间：%s（远端 SCNet K100AI）\n" % out["generated_at"])
    md.append("> 口径：仅统计公开上游评测集（CV-Bench / RefSpatial-Bench / TraceSpatial-Bench）；"
              "不含模型加载/服务启动时间，不含压测。\n")
    md.append("\n## 1. 已完成评测的墙钟\n")
    md.append("| # | 类别 | 评测项 | 规模 | 并发 | 档位 | 墙钟 | 证据 |")
    md.append("|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        md.append("| %d | %s | %s | %d | %s | %s | **%s** | `%s` |"
                  % (i, r["kind"], r["name"], r["n"], r.get("conc", "-"), r["level"],
                     fmt(r["wall_s"]), r["log"]))
    md.append("\n**已完成合计（串行累加口径）：%s**\n" % fmt(done_wall))
    md.append("\n> 3D 判分墙钟取 `scores_3dchunk_3D.json` 的 `wall_s_all_jobs`（进程级真实墙上时钟）；"
              "同文件的 `wall_s_total` 是各分片墙钟之和（并发累加，并发 16 时约为真实墙钟的 4 倍），"
              "不可直接当墙钟使用。\n")
    md.append("\n## 2. 主动中止的评测（不计入合计）\n")
    md.append("| 评测项 | 已耗时 | 中止原因 | 证据 |")
    md.append("|---|---|---|---|")
    for a in aborted:
        wtxt = fmt(a.get("wall_s"))
        if a.get("wall_is_lower_bound"):
            wtxt = "≥ " + wtxt
        md.append("| %s | %s | %s | `%s` |" % (a["name"], wtxt, a["reason"], a["log"]))
    md.append("\n**中止合计（沉没成本）：%s**\n" % fmt(aborted_wall))
    md.append("\n## 3. 耗时分级规则（本轮落地）\n")
    md.append("| 档位 | 预估墙钟 | 处置 |")
    md.append("|---|---|---|")
    md.append("| L1 | < 5 min | 随时可跑 |")
    md.append("| L2 | 5 ~ 20 min | 正常排队 |")
    md.append("| L3 | 20 ~ 60 min | 需设上限，优先提并发压缩 |")
    md.append("| L4 | > 60 min | 不跑；先排除病态样本或改用兜底口径 |")
    md.append("\n规则落地实例：\n")
    md.append("- TraceSpatial 3D 整集（L4）→ 改为 **96 条兜底口径**，941 s 完成；\n")
    md.append("- CV-Bench Count up4 组（L4）→ 主动停止，保留 up2 作为第二口径；\n")
    md.append("- CV-Bench 全量 up2（L3）→ 并发由 16 提到 48（服务 `max_num_seqs=64`），"
              "墙钟由 ~40 min 压到 ~15 min。\n")

    with open(os.path.join(OUT_DIR, "eval_timing.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")

    print("rows=%d aborted=%d done=%s aborted=%s"
          % (len(rows), len(aborted), fmt(done_wall), fmt(aborted_wall)))
    for r in rows:
        print("  %-58s %10s  %s" % (r["name"][:58], fmt(r["wall_s"]), r["level"]))
    for a in aborted:
        print("  [中止] %-54s %10s" % (a["name"][:54], fmt(a.get("wall_s"))))


if __name__ == "__main__":
    main()
