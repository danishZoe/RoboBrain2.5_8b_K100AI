#!/usr/bin/env bash
# 精度流水线状态速查
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
date
echo
echo "===== 流水线阶段 ====="
grep -E '^########## ' logs/run_accuracy_pipeline.log | tail -4
echo
echo "===== 候选 CV-Bench ====="
python3 - <<'PY'
import json, os
for tag, root in (("基线(eager)", "outputs/bench_public"), ("候选(非eager)", "outputs/bench_public_noneager")):
    p = os.path.join(root, "cvbench", "summary.json")
    if not os.path.exists(p):
        print("  %-16s 还没有 summary.json" % tag); continue
    d = json.load(open(p, encoding="utf-8"))
    r = d.get("scores", {}).get("official_result", {})
    print("  %-16s n=%s ret=%s states=%s" % (
        tag, d.get("n_planned"), d.get("n_returned"), d.get("terminal_states")))
    print("      %s" % {k: r.get(k) for k in
          ("accuracy_2d_ade","accuracy_2d_coco","accuracy_3d_omni",
           "accuracy_2d","accuracy_3d","accuracy")})
PY
echo
echo "===== 候选 RefSpatial 默认范围均值 ====="
python3 - <<'PY'
import json, os
for tag, root in (("基线(eager)", "outputs/bench_public"), ("候选(非eager)", "outputs/bench_public_noneager")):
    p = os.path.join(root, "refspatial", "summary.json")
    if not os.path.exists(p):
        print("  %-16s 还没有 summary.json" % tag); continue
    d = json.load(open(p, encoding="utf-8"))
    for v, vin in d.get("variants", {}).items():
        s = vin.get("splits", {})
        vals = [s[k]["scores"]["official_cli"]["official_accuracy"]
                for k in ("Location", "Placement") if k in s]
        if vals:
            print("  %-16s %-18s 默认范围均值=%.4f" % (tag, v, sum(vals)/len(vals)))
PY
echo
echo "===== TraceSpatial 判分进度 ====="
tail -n 3 logs/public_bench_baseline-eager/tracespatial_score.log 2>/dev/null | cut -c1-120
echo "--- 候选 infer ---"
tail -n 2 logs/public_bench_noneager-candidate/tracespatial_infer.log 2>/dev/null | cut -c1-120
echo "--- 3D 判分进程数 ---"
ps -eo pid,etime,cmd | grep 'task_name 3D' | grep -v grep | wc -l
echo
echo "===== 协议实验进度 ====="
PB="$WS/outputs/bench_public/cvbench_protocol"
for d in "$PB"/*/; do
  [ -d "$d" ] || continue
  n=$(wc -l < "$d/predictions/cvbench.jsonl" 2>/dev/null || echo 0)
  echo "  $(basename "$d"): $n/788"
done
if [ -f "$PB/protocol_compare.json" ]; then
  echo "  --- 各协议结果（Count 子任务）---"
  python3 - <<'PY'
import json
p = "/root/private_data/workspace/robo/outputs/bench_public/cvbench_protocol/protocol_compare.json"
for r in json.load(open(p, encoding="utf-8")):
    print("  %-20s ADE20K=%-7s COCO=%-7s Acc_2D=%-7s | Count %d/%d = %.2f%%" %
          (r[0], r[1], r[2], r[3], r[5], r[6], 100.0 * r[5] / max(r[6], 1)))
PY
fi
echo "===== 3D 判分进程数 ====="
ps -eo cmd | grep 'task_name 3D' | grep -v grep | wc -l
