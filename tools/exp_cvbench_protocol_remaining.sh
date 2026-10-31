#!/usr/bin/env bash
# CV-Bench 图像输入协议因子实验 —— 只补齐尚未完成的协议组（已完成的跳过）
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
OUT="$WS/outputs/bench_public/cvbench_protocol"
LOG="$WS/logs/public_bench_baseline-eager/cvbench_protocol.log"
mkdir -p "$OUT" "$WS/logs/public_bench_baseline-eager"

PROTOCOLS="canonical_png canonical_jpeg95 raw_png raw_jpeg95 long1500_png long1500_jpeg95 maxpix501760_png maxpix501760_jpeg95"

echo "===== [0] 服务健康检查 ====="
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$URL" || echo 000)
curl -s -o /dev/null -w 'health=%{http_code}\n' --max-time 5 http://127.0.0.1:30000/health
if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:30000/health)" != "200" ]; then
  echo "服务未就绪，退出"; exit 1
fi

echo
echo "===== [1] CV-Bench Count 子任务 × ${PROTOCOLS}（跳过已完成）====="
for P in $PROTOCOLS; do
  D="$OUT/$P"
  if [ -f "$D/summary.json" ]; then
    echo "--- protocol=$P 已完成，跳过 ($(date '+%T')) ---"
    continue
  fi
  mkdir -p "$D"
  echo "--- protocol=$P  ($(date '+%T')) ---"
  python3 "$WS/submission/bench_public/run_cv_bench.py" \
    --url "$URL" --run-id "baseline-eager-cvbench-count-$P" \
    --out-dir "$D" --task-filter Count \
    --image-protocol "$P" --no-resume \
    2>&1 | tee -a "$LOG" | grep -E '题目数|accuracy_|图像输入协议' || true
done

echo
echo "===== [2] 汇总：各协议的 Count 准确率 ====="
python3 - <<'PY'
import json, os
BASE = "/root/private_data/workspace/robo/outputs/bench_public/cvbench_protocol"
rows = []
for p in sorted(os.listdir(BASE)):
    sp = os.path.join(BASE, p, "summary.json")
    if not os.path.exists(sp):
        continue
    d = json.load(open(sp, encoding="utf-8"))
    res = d.get("scores", {}).get("official_result", {})
    per = os.path.join(BASE, p, "per_sample.json")
    n_ok = n_tot = 0
    if os.path.exists(per):
        for s in json.load(open(per, encoding="utf-8")):
            ok = (s.get("extracted_answer") or "")[:1].upper() == (s.get("gt_answer") or "")[:1].upper()
            n_ok += bool(ok); n_tot += 1
    rows.append((p, res.get("accuracy_2d_ade"), res.get("accuracy_2d_coco"),
                 res.get("accuracy_2d"), res.get("accuracy"), n_ok, n_tot))
print("%-20s %8s %8s %8s %8s   %s" % ("protocol", "ADE20K", "COCO", "Acc_2D", "CV", "Count 正确/总数"))
for r in rows:
    print("%-20s %8s %8s %8s %8s   %d/%d = %.2f%%" % (
        r[0], r[1], r[2], r[3], r[4], r[5], r[6], 100.0*r[5]/max(r[6],1)))
json.dump(rows, open(os.path.join(BASE, "protocol_compare.json"), "w"), indent=2)
print("\n-> %s" % os.path.join(BASE, "protocol_compare.json"))
PY
echo "===== 协议实验结束 $(date '+%F %T') ====="
