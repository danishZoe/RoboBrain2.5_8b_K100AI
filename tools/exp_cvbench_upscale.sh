#!/usr/bin/env bash
# CV-Bench Count：上采样协议实验（up2 / up4）+ 同配置下的 canonical_png 对照
#
# 动机：模型原生 preprocessor 的 shortest_edge=65536，256x256 的图只有 64 个视觉
# token；Count 子任务里的小目标可能因 token 太少被整块吞掉。上采样不增加信息，
# 但会改变 ViT 的 patch 网格密度。up2/up4 总像素上限 3,000,000 px。
#
# 三个协议在**同一服务实例、同一 session** 上跑，保证可比。
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONPATH="$WS/envs/pylibs_core:$WS/third_party/upstream_eval/FlagEvalMM:$WS/scripts:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore
export HF_HOME="$WS/cache/hf"
export XDG_CACHE_HOME="$WS/cache/xdg"
export TMPDIR="$WS/tmp"
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

URL=http://127.0.0.1:30000/v1/chat/completions
OUT="$WS/outputs/bench_public/cvbench_upscale"
LOG="$WS/logs/cvbench_upscale.log"
mkdir -p "$OUT"

PROTOCOLS="canonical_png up2_png up2_jpeg95 up4_png up4_jpeg95"

echo "===== 服务健康检查 ====="
curl -s -o /dev/null -w "health=%{http_code}\n" http://127.0.0.1:30000/health || exit 1

echo "===== [1] Count 子任务 × ${PROTOCOLS} ====="
for P in $PROTOCOLS; do
  D="$OUT/$P"
  mkdir -p "$D"
  echo "--- protocol=$P  ($(date '+%T')) ---"
  python3 "$WS/submission/bench_public/run_cv_bench.py" \
    --url "$URL" --run-id "upscale-cvbench-count-$P" \
    --out-dir "$D" --task-filter Count \
    --image-protocol "$P" --no-resume \
    2>&1 | tee -a "$LOG" | grep -E '题目数|accuracy_|图像输入协议|失败' || true
done

echo
echo "===== [2] 汇总 ====="
python3 - <<'PY'
import json, os, collections
BASE = "/root/private_data/workspace/robo/outputs/bench_public/cvbench_upscale"
REF = "/root/private_data/workspace/robo/outputs/bench_public/cvbench_protocol"
rows = []
for p in sorted(os.listdir(BASE)):
    sp = os.path.join(BASE, p, "summary.json")
    per = os.path.join(BASE, p, "per_sample.json")
    if not os.path.exists(sp):
        continue
    d = json.load(open(sp, encoding="utf-8"))
    res = d.get("scores", {}).get("official_result", {})
    anns = json.load(open(os.path.join(BASE, p, "annotations.json"), encoding="utf-8"))
    st = collections.defaultdict(lambda: [0, 0])
    if os.path.exists(per):
        for s in json.load(open(per, encoding="utf-8")):
            g = anns.get(s["question_id"], {})
            k = "%s/%s" % (g.get("source"), g.get("task"))
            st[k][0] += ((s.get("extracted_answer") or "")[:1].upper() ==
                         (s.get("gt_answer") or "")[:1].upper())
            st[k][1] += 1
    allc = sum(v[0] for v in st.values()); alln = sum(v[1] for v in st.values())
    rows.append({"protocol": p, "ade_count": st["ADE20K/Count"][0],
                 "ade_count_n": st["ADE20K/Count"][1],
                 "coco_count": st["COCO/Count"][0], "coco_count_n": st["COCO/Count"][1],
                 "all_correct": allc, "all_total": alln,
                 "all_acc": round(100.0 * allc / max(alln, 1), 2),
                 "official": {k: res.get(k) for k in
                              ("accuracy_2d_ade", "accuracy_2d_coco",
                               "accuracy_2d", "accuracy_3d", "accuracy")}})
print("%-18s %-14s %-14s %-14s" % ("protocol", "ADE20K/Count", "COCO/Count", "Count 全部"))
for r in rows:
    print("%-18s %-14s %-14s %-14s" % (
        r["protocol"],
        "%d/%d=%.2f%%" % (r["ade_count"], r["ade_count_n"],
                          100.0 * r["ade_count"] / max(r["ade_count_n"], 1)),
        "%d/%d=%.2f%%" % (r["coco_count"], r["coco_count_n"],
                          100.0 * r["coco_count"] / max(r["coco_count_n"], 1)),
        "%d/%d=%.2f%%" % (r["all_correct"], r["all_total"], r["all_acc"])))
# 与既有 8 协议矩阵的 canonical_png 对照
ref = os.path.join(REF, "canonical_png", "per_sample.json")
if os.path.exists(ref):
    for r in rows:
        if r["protocol"] != "canonical_png":
            continue
        anns = json.load(open(os.path.join(BASE, "canonical_png", "annotations.json"),
                              encoding="utf-8"))
        cur = {s["question_id"]: (s.get("extracted_answer") or "")[:1].upper()
               for s in json.load(open(os.path.join(BASE, "canonical_png",
                                                    "per_sample.json"), encoding="utf-8"))}
        old = {s["question_id"]: (s.get("extracted_answer") or "")[:1].upper()
               for s in json.load(open(ref, encoding="utf-8"))}
        same = sum(1 for k in cur if k in old and cur[k] == old[k])
        diff = sum(1 for k in cur if k in old and cur[k] != old[k])
        print("\n  同协议(本 session) vs 既有 8 协议矩阵的 canonical_png：一致 %d，不一致 %d" % (same, diff))
json.dump(rows, open(os.path.join(BASE, "upscale_compare.json"), "w"),
          ensure_ascii=False, indent=2)
print("\n-> %s" % os.path.join(BASE, "upscale_compare.json"))
PY
echo "===== 上采样实验结束 $(date '+%F %T') ====="
