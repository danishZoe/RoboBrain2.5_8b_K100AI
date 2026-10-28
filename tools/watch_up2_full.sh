#!/usr/bin/env bash
# 全量 CV-Bench（up2_png，并发 48）看门狗
#
# 用户要求「慢评测要设上限」：本次 L3 档运行硬上限 = 启动后 59 min。
#  - 在上限内跑完 -> run_cv_bench.py 自带官方判分，直接产出 summary.json
#  - 超过上限     -> 主动停止，标记为「已排除/未完成」，不影响已有结论
#
# 不向服务发任何额外请求（只读文件 + 观察进程）。
set -uo pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
LOG="$WS/logs/watch_up2_full.log"
DEADLINE_EPOCH=$1   # 绝对截止时间（epoch 秒）

echo "===== up2 全量看门狗启动 $(date '+%F %T') 截止 $(date -d "@$DEADLINE_EPOCH" '+%F %T') =====" >> "$LOG"

while true; do
  if ! pgrep -f 'run_cv_bench.py .*up2-cvbench-full' >/dev/null 2>&1; then
    echo "[watch] run_cv_bench 进程已退出 $(date '+%F %T')" >> "$LOG"
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE_EPOCH" ]; then
    echo "[watch] 超过硬上限，主动停止 $(date '+%F %T')" >> "$LOG"
    pkill -f 'run_cv_bench.py .*up2-cvbench-full' || true
    sleep 5
    pkill -9 -f 'run_cv_bench.py .*up2-cvbench-full' 2>/dev/null || true
    echo "[watch] 已停止；本次不计入完成台账" >> "$LOG"
    python3 tools/build_eval_timing.py >> "$LOG" 2>&1
    echo "===== watch 结束（超时停止）$(date '+%F %T') =====" >> "$LOG"
    exit 0
  fi
  sleep 15
done

# 跑完了：等 summary.json 落盘，然后取官方判分结果
for i in $(seq 1 40); do
  [ -f "$WS/outputs/bench_public/cvbench_up2_full/summary.json" ] && break
  sleep 15
done

if [ -f "$WS/outputs/bench_public/cvbench_up2_full/summary.json" ]; then
  echo "[watch] 产出 summary.json，开始提取官方判分结果" >> "$LOG"
  python3 - <<'PY' >> "$LOG" 2>&1
import json
p = "/root/private_data/workspace/robo/outputs/bench_public/cvbench_up2_full/summary.json"
d = json.load(open(p, encoding="utf-8"))
sc = d.get("scores", {}).get("official_result", {})
res = {k: sc.get(k) for k in
       ("accuracy_2d_ade", "accuracy_2d_coco", "accuracy_3d_omni",
        "accuracy_2d", "accuracy_3d", "accuracy")}
res["n_planned"] = d.get("n_planned")
res["n_returned"] = d.get("n_returned")
res["n_fallback_extraction"] = d.get("n_fallback_extraction")
res["terminal_states"] = d.get("terminal_states")
res["image_protocol"] = d.get("protocol", {}).get("image_protocol")
res["concurrency"] = d.get("protocol", {}).get("concurrency")
out = "/root/private_data/workspace/robo/outputs/bench_public/cvbench_up2_full_result.json"
json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("RESULT", json.dumps(res, ensure_ascii=False))
PY
else
  echo "[watch] 进程退出但无 summary.json（异常）" >> "$LOG"
fi

python3 tools/build_eval_timing.py >> "$LOG" 2>&1
echo "===== watch 结束（正常完成）$(date '+%F %T') =====" >> "$LOG"
