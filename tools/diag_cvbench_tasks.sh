#!/usr/bin/env bash
# 诊断：CV-Bench 按 task / 选项数 的准确率分解 + 错误样例
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONPATH="$WS/envs/pylibs_core:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore

python3 - <<'PY'
import json, os, collections, statistics

BASE = "/root/private_data/workspace/robo/outputs/bench_public/cvbench"
anns = json.load(open(os.path.join(BASE, "annotations.json"), encoding="utf-8"))
preds = json.load(open(os.path.join(BASE, "predictions_official.json"), encoding="utf-8"))

by_task = collections.defaultdict(lambda: [0, 0, 0])   # task -> [correct, total, nopt_sum]
by_nopt = collections.defaultdict(lambda: [0, 0])
by_task_type = collections.defaultdict(lambda: [0, 0])
wrong_samples = collections.defaultdict(list)

for p in preds:
    qid = str(p["question_id"])
    g = anns.get(qid)
    if g is None:
        continue
    pa = (p.get("answer") or "")[:1].upper()
    ga = (g.get("answer") or "")[:1].upper()
    ok = (ga == pa)
    t = g.get("task") or "?"
    nopt = len(g.get("options") or [])
    by_task[t][0] += ok
    by_task[t][1] += 1
    by_task[t][2] += nopt
    by_nopt[nopt][0] += ok
    by_nopt[nopt][1] += 1
    by_task_type[(g.get("source"), t)][0] += ok
    by_task_type[(g.get("source"), t)][1] += 1
    if not ok and len(wrong_samples[t]) < 3:
        wrong_samples[t].append({
            "qid": qid, "question": (g.get("question") or "")[:150],
            "options": g.get("options"), "gt": ga, "pred": pa,
            "filename": g.get("filename")})

print("--- 按 task ---")
tot_r = tot_t = 0
for t in sorted(by_task, key=lambda k: -by_task[k][1]):
    r, n, _ = by_task[t]
    tot_r += r; tot_t += n
    print("  %-22s %4d/%4d = %6.2f%%  (平均选项数 %.1f)" % (t, r, n, 100.0*r/max(n,1), by_task[t][2]/n))
print("  %-22s %4d/%4d = %6.2f%%" % ("【合计】", tot_r, tot_t, 100.0*tot_r/max(tot_t,1)))

print("\n--- 按选项数 ---")
for n in sorted(by_nopt):
    r, t = by_nopt[n]
    print("  %d 选项: %4d/%4d = %6.2f%%" % (n, r, t, 100.0*r/max(t,1)))

print("\n--- 按 (source, task) ---")
for k in sorted(by_task_type, key=lambda k: (k[0], by_task_type[k][1])):
    r, n = by_task_type[k]
    print("  %-10s %-20s %4d/%4d = %6.2f%%" % (k[0], k[1], r, n, 100.0*r/max(n,1)))

print("\n--- 各 task 的错误样例 ---")
for t in sorted(wrong_samples):
    print(" [%s]" % t)
    for s in wrong_samples[t]:
        print("   ", json.dumps(s, ensure_ascii=False)[:330])
PY
