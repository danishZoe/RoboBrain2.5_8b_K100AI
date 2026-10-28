#!/usr/bin/env bash
# 诊断：CV-Bench 失分层级分析（按选项数 / 按 source / 视觉 token 预算）
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONPATH="$WS/envs/pylibs_core:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore

python3 - <<'PY'
import json, os, collections, statistics

BASE = "/root/private_data/workspace/robo/outputs/bench_public/cvbench"
ann_p = os.path.join(BASE, "annotations.json")
prd_p = os.path.join(BASE, "predictions_official.json")
per_p = os.path.join(BASE, "per_sample.json")

anns = json.load(open(ann_p, encoding="utf-8"))
preds = json.load(open(prd_p, encoding="utf-8"))
print("annotations=%d predictions=%d" % (len(anns), len(preds)))

# 官方判分器已在 predictions 里写入 correct / answer / label
by_nopt = collections.defaultdict(lambda: [0, 0])
by_src = collections.defaultdict(lambda: [0, 0])
pred_letter = collections.Counter()
gt_letter = collections.Counter()
confusion = collections.Counter()
for p in preds:
    qid = str(p["question_id"])
    g = anns.get(qid)
    if g is None:
        continue
    nopt = len(g.get("options") or [])
    ok = bool(p.get("correct"))
    by_nopt[nopt][0] += ok
    by_nopt[nopt][1] += 1
    by_src[g["source"]][0] += ok
    by_src[g["source"]][1] += 1
    pa = (p.get("answer") or "")[:1]
    ga = (g.get("answer") or "")[:1]
    pred_letter[pa] += 1
    gt_letter[ga] += 1
    confusion[(ga, pa)] += 1

print("\n--- 按选项数分层的准确率 ---")
for n in sorted(by_nopt):
    r, t = by_nopt[n]
    print("  %d 选项: %4d/%4d = %6.2f%%" % (n, r, t, 100.0 * r / max(t, 1)))
print("\n--- 按 source ---")
for s in sorted(by_src):
    r, t = by_src[s]
    print("  %-8s %4d/%4d = %6.2f%%" % (s, r, t, 100.0 * r / max(t, 1)))

print("\n--- 预测字母分布 vs 真值字母分布 ---")
print("  pred:", dict(pred_letter.most_common()))
print("  gt  :", dict(gt_letter.most_common()))

print("\n--- 真值->预测 混淆 Top15 ---")
for (g, p), c in confusion.most_common(15):
    mark = "  <=命中" if g == p else ""
    print("   gt=%s pred=%s -> %4d%s" % (g, p, c, mark))

# 视觉 token 预算：usage.prompt_tokens
if os.path.exists(per_p):
    per = json.load(open(per_p, encoding="utf-8"))
    items = per if isinstance(per, list) else per.get("samples", [])
    pt = [ (r.get("usage") or {}).get("prompt_tokens") for r in items ]
    pt = [x for x in pt if isinstance(x, (int, float))]
    if pt:
        print("\n--- 服务端 usage.prompt_tokens（含视觉 token）---")
        print("  n=%d  min=%d  p50=%d  p90=%d  max=%d  mean=%.0f" % (
            len(pt), min(pt), int(statistics.median(pt)),
            int(sorted(pt)[int(len(pt) * 0.9)]), max(pt), statistics.mean(pt)))
        print("  分位: p10=%d p25=%d p75=%d" % (
            int(sorted(pt)[int(len(pt) * 0.10)]), int(sorted(pt)[int(len(pt) * 0.25)]),
            int(sorted(pt)[int(len(pt) * 0.75)])))
    # 图像尺寸分布
    geos = [r.get("image_geometry") for r in items if r.get("image_geometry")]
    if geos:
        ws = [g.get("width") for g in geos if g.get("width")]
        hs = [g.get("height") for g in geos if g.get("height")]
        px = [w * h for w, h in zip(ws, hs)]
        print("\n--- 送入图像尺寸 ---")
        print("  width  min=%d max=%d ; height min=%d max=%d" % (min(ws), max(ws), min(hs), max(hs)))
        print("  像素数 p50=%d p90=%d max=%d ; >501760 的占比 %.1f%%" % (
            int(statistics.median(px)), int(sorted(px)[int(len(px)*0.9)]), max(px),
            100.0 * sum(1 for v in px if v > 501760) / len(px)))
PY
