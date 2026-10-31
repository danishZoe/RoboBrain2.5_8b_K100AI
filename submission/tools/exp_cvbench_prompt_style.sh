#!/usr/bin/env bash
# CV-Bench 提示词口径因子实验（只跑 Count 子任务，788 题）
#
# 因子：提示词口径
#   flagevalmm = FlagEvalMM VqaBaseDataset.build_prompt + PromptTemplate + RoboBrain 适配器
#                （我们当前主口径；选项标记 "A." + "Answer with the option's letter..."）
#   dataset    = CV-Bench 数据集自带的 prompt 字段（Cambrian-1 原始口径；
#                选项标记 "(A)" + "Select from the following choices."）
#
# 图像协议固定 canonical_png（同主口径），唯一变量 = 提示词。
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
OUT="$WS/outputs/bench_public/cvbench_prompt"
LOG="$WS/logs/cvbench_prompt.log"
mkdir -p "$OUT"

echo "===== 服务健康检查 ====="
curl -s -o /dev/null -w "health=%{http_code}\n" http://127.0.0.1:30000/health || exit 1

for STYLE in dataset flagevalmm; do
  D="$OUT/$STYLE"
  mkdir -p "$D"
  echo "--- prompt-style=$STYLE  ($(date '+%T')) ---"
  python3 "$WS/submission/bench_public/run_cv_bench.py" \
    --url "$URL" --run-id "cvbench-count-prompt-$STYLE" \
    --out-dir "$D" --task-filter Count \
    --image-protocol canonical_png --prompt-style "$STYLE" --no-resume \
    2>&1 | tee -a "$LOG" | grep -E '题目数|accuracy_|图像输入协议|失败' || true
done

echo
echo "===== 汇总 ====="
python3 - <<'PY'
import json, os, collections
BASE = "/root/private_data/workspace/robo/outputs/bench_public/cvbench_prompt"
rows = []
for p in sorted(os.listdir(BASE)):
    sp = os.path.join(BASE, p, "summary.json")
    if not os.path.exists(sp):
        continue
    d = json.load(open(sp, encoding="utf-8"))
    res = d.get("scores", {}).get("official_result", {})
    anns = json.load(open(os.path.join(BASE, p, "annotations.json"), encoding="utf-8"))
    st = collections.defaultdict(lambda: [0, 0])
    per = os.path.join(BASE, p, "per_sample.json")
    if os.path.exists(per):
        for s in json.load(open(per, encoding="utf-8")):
            g = anns.get(s["question_id"], {})
            k = "%s/%s" % (g.get("source"), g.get("task"))
            st[k][0] += ((s.get("extracted_answer") or "")[:1].upper() ==
                         (s.get("gt_answer") or "")[:1].upper())
            st[k][1] += 1
    allc = sum(v[0] for v in st.values()); alln = sum(v[1] for v in st.values())
    rows.append({"style": p, "ade": st["ADE20K/Count"], "coco": st["COCO/Count"],
                 "all": (allc, alln), "official": res})
print("%-12s %-16s %-16s %-16s" % ("prompt-style", "ADE20K/Count", "COCO/Count", "Count 全部"))
for r in rows:
    print("%-12s %-16s %-16s %-16s" % (
        r["style"],
        "%d/%d=%.2f%%" % (r["ade"][0], r["ade"][1], 100.0 * r["ade"][0] / max(r["ade"][1], 1)),
        "%d/%d=%.2f%%" % (r["coco"][0], r["coco"][1], 100.0 * r["coco"][0] / max(r["coco"][1], 1)),
        "%d/%d=%.2f%%" % (r["all"][0], r["all"][1], 100.0 * r["all"][0] / max(r["all"][1], 1))))
    print("             官方 get_result: %s" % json.dumps(r["official"], ensure_ascii=False))
json.dump(rows, open(os.path.join(BASE, "prompt_compare.json"), "w"),
          ensure_ascii=False, indent=2)
print("\n-> %s" % os.path.join(BASE, "prompt_compare.json"))
PY
echo "===== 提示词口径实验结束 $(date '+%F %T') ====="
