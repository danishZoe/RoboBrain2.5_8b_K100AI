#!/usr/bin/env bash
# 修正 TraceSpatial 预测记录里的 terminal_state 记账
#
# 背景：官方变体的 2D 任务输出是 2 元组 ``[(x, y), ...]``，而我们原先给 2D 任务
# 也用了只认 3 元组的 trajectory 解析器，导致 100/100 被误标为 parse_failed。
# 官方判分脚本 ``text2pts_normalize`` 两种格式都接受，**判分内容完全不受影响**，
# 影响的只是我们自己的状态计数（以及 resume 时是否会重跑）。
#
# 本脚本离线重算 terminal_state 并回写 predictions/*.jsonl 与 summary.json，
# **不改动 raw_output / model_prediction**，因此不改变任何已产生的判分结果。
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONPATH="$WS/envs/pylibs_core:$WS/submission/bench_public:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

python3 - <<'PY'
import json, os, sys
from collections import Counter
sys.path.insert(0, "/root/private_data/workspace/robo/submission/bench_public")
import common as C

ROOTS = [("/root/private_data/workspace/robo/outputs/bench_public", "基线 eager"),
         ("/root/private_data/workspace/robo/outputs/bench_public_noneager", "候选 非eager")]


def recompute_state(rec, parse_task):
    """复刻 engine._terminal_state 的判定，只把 parse 换成修好的解析器。"""
    raw = rec.get("raw_output") or ""
    if rec.get("finish_reason") == "length":
        return "truncated", None
    if not raw.strip():
        return "empty_output", None
    try:
        parsed = C.parse_points_1000(raw, parse_task)
    except Exception as exc:
        return "parse_failed", {"error": repr(exc)}
    if not parsed:
        return "parse_failed", parsed
    return "ok", parsed


for root, label in ROOTS:
    ts = os.path.join(root, "tracespatial")
    if not os.path.isdir(ts):
        continue
    sp = os.path.join(ts, "summary.json")
    summary = json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else None
    for variant in sorted(os.listdir(ts)):
        pdir = os.path.join(ts, variant, "predictions")
        if not os.path.isdir(pdir):
            continue
        for task in ("2D", "3D"):
            pp = os.path.join(pdir, "%s.jsonl" % task)
            if not os.path.exists(pp):
                continue
            parse_task = "trajectory_2d" if task == "2D" else "trajectory"
            recs = [json.loads(l) for l in open(pp, encoding="utf-8") if l.strip()]
            before = Counter(r.get("terminal_state") for r in recs)
            changed = 0
            for r in recs:
                st, parsed = recompute_state(r, parse_task)
                if r.get("terminal_state") != st:
                    changed += 1
                r["terminal_state"] = st
                r["parsed"] = parsed
            with open(pp, "w", encoding="utf-8") as fh:
                for r in recs:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            after = Counter(r.get("terminal_state") for r in recs)
            print("%s %-8s %-3s: before=%s -> after=%s (改判 %d 条)" % (
                label, variant, task, dict(before), dict(after), changed))
            if summary:
                info = summary.get("variants", {}).get(variant, {}).get("tasks", {}).get(task)
                if info is not None:
                    info["terminal_states"] = dict(after)
    if summary:
        with open(sp, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        print("  -> 已回写 %s" % sp)
PY
echo "===== 修正完成 $(date '+%F %T') ====="
