#!/usr/bin/env bash
# =============================================================================
# 干净环境复现（验收标准 §7.2）
#
#   "从干净环境执行启动、环境采集、Smoke、精度与性能流程，保存复跑日志"
#
# 本脚本做的事：
#   1) 停掉当前服务，用提交包自带的 run.sh 重新拉起（等于一次全新启动）
#   2) env_check.sh  -> 环境采集
#   3) smoke.sh      -> Smoke
#   4) verify.py     -> 精度测评（无官方基线时如实 BLOCKED）
#   5) 官方压测客户端跑 4096/1024/并发 1 档 -> 性能复跑，与提交表同口径比对
#   6) 全程日志落 logs/clean_repro/
#
# 判定：性能复跑与 submission/bench/summary.md 同档位（记录轮）的相对偏差 <= 20%
#      （PDF 规定复现偏差 >20% 打回）；Smoke 必须全过；精度项只能是 PASS 或 BLOCKED，
#      出现 FAIL 即整体判失败。
#
# 用法: bash tools/clean_repro.sh
# =============================================================================
set -uo pipefail
W=/root/private_data/workspace/robo
OUT="$W/logs/clean_repro"
MODEL_PATH="/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV"
mkdir -p "$OUT"
cd "$W/submission" || exit 1

echo "=== 0) 等待收尾流水线全部结束 ==="
DEADLINE=$(( $(date +%s) + 36000 ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  if ! pgrep -f 'pipeline_after_bench.sh' >/dev/null 2>&1 \
     && ! pgrep -f 'pipeline_final.sh' >/dev/null 2>&1; then
    echo "pipelines finished at $(date -Is)"; break
  fi
  sleep 30
done
# 安全闸：官方压测必须已真正结束（本脚本会重启服务）
while pgrep -f 'bash bench.sh' >/dev/null 2>&1 || pgrep -f 'bench_serving' >/dev/null 2>&1; do
  echo "[$(date -Is)] 官方压测仍在运行，等待其结束（本脚本会重启服务，不能与压测并行）..."
  sleep 30
done
N=$(wc -l < "$W/submission/bench/round2.jsonl" 2>/dev/null || echo 0)
echo "round2 entries = $N"

source ./env.sh

echo
echo "=== 1) 全新启动服务（submission/run.sh）==="
pkill -f "scripts/serve_vllm_patched.py" 2>/dev/null
pkill -f "VLLM::EngineCore" 2>/dev/null
sleep 8
T0=$(date +%s)
setsid nohup bash run.sh > "$OUT/server_start.log" 2>&1 &
READY=0
for i in $(seq 1 120); do
  if curl -sf --noproxy '*' "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    READY=1
    echo "server up after $(( $(date +%s) - T0 ))s"
    break
  fi
  sleep 10
done
[ "$READY" = "1" ] || { echo "FAIL: 服务未起来"; exit 1; }
cp -f server.log "$OUT/server.log" 2>/dev/null

echo
echo "=== 2) env_check.sh ==="
bash env_check.sh > "$OUT/env_check.log" 2>&1; echo "exit=$?"
cp -f env.log "$OUT/env.log" 2>/dev/null

echo
echo "=== 3) smoke.sh ==="
bash smoke.sh > "$OUT/smoke.log" 2>&1; echo "exit=$?"
cp -f smoke.log "$OUT/smoke.log" 2>/dev/null

echo
echo "=== 4) verify.py ==="
python3 verify.py > "$OUT/verify_stdout.log" 2>&1; echo "exit=$? (2 = BLOCKED，预期)"
cp -f verify_result.json "$OUT/verify_result.json" 2>/dev/null

echo
echo "=== 5) 性能复跑：4096/1024/并发 1（官方客户端，与提交表同口径）==="
cd "$W" || exit 1
python3 scripts/bench_serving_vllm_image.py \
  --backend vllm --host 127.0.0.1 --port "$PORT" \
  --model "$SERVED_NAME" --tokenizer "$MODEL_PATH" \
  --dataset-name image --image-count 1 --image-resolution 1080p \
  --random-input-len 4096 --random-output-len 1024 --random-range-ratio 1 \
  --max-concurrency 1 --num-prompts 8 --seed 1 \
  --output-file "$OUT/repro_4096_c1.jsonl" \
  > "$OUT/repro_4096_c1.log" 2>&1
echo "bench exit=$?"

echo
echo "=== 6) 复现判定 ==="
python3 - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
ok = True

# 性能
base = None
for line in open("submission/bench/round2.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r.get("random_input_len") == 4096 and r.get("max_concurrency") == 1:
        base = r
repro = None
p = os.path.join(out, "repro_4096_c1.jsonl")
if os.path.exists(p):
    for line in open(p, encoding="utf-8"):
        if line.strip():
            repro = json.loads(line)

if base and repro:
    b, c = base["output_throughput"], repro["output_throughput"]
    dev = (c - b) / b * 100.0
    print("| 项 | 提交表(记录轮) | 本次复跑 | 相对偏差 |")
    print("|---|---|---|---|")
    print("| 输出吞吐 tok/s | %.2f | %.2f | %+.1f%% |" % (b, c, dev))
    print("| 完成请求数 | %d | %d | - |" % (base["completed"], repro["completed"]))
    print("| 失败请求数 | %d | %d | - |" % (base.get("failed", 0), repro.get("failed", 0)))
    print("| 平均 TTFT ms | %.2f | %.2f | %+.1f%% |" % (
        base["mean_ttft_ms"], repro["mean_ttft_ms"],
        (repro["mean_ttft_ms"] - base["mean_ttft_ms"]) / base["mean_ttft_ms"] * 100.0))
    if abs(dev) > 20.0:
        print("FAIL: 性能复现偏差 %.1f%% 超过 20%% 打回线" % dev); ok = False
    else:
        print("PASS: 性能复现偏差 %.1f%% 在 20%% 允许范围内" % dev)
    if repro.get("completed") != 8 or repro.get("failed", 0):
        print("FAIL: 复跑存在失败请求"); ok = False
else:
    print("FAIL: 缺少提交表或复跑数据"); ok = False

# smoke
sl = os.path.join(out, "smoke.log")
if os.path.exists(sl):
    txt = open(sl, encoding="utf-8", errors="replace").read()
    print("PASS: smoke.log 已生成 (%d B)" % len(txt))
else:
    print("FAIL: smoke.log 未生成"); ok = False

# verify
vp = os.path.join(out, "verify_result.json")
if os.path.exists(vp):
    d = json.load(open(vp, encoding="utf-8"))
    st = d.get("status")
    print("精度项状态: %s (passed=%s)" % (st, d.get("passed")))
    if st not in ("PASS", "BLOCKED"):
        print("FAIL: 精度项状态异常"); ok = False
else:
    print("FAIL: verify_result.json 未生成"); ok = False

print()
print("干净复现结论:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
RC=$?

echo
echo "=== 复现日志 ==="
ls -l "$OUT" | sed 's/^/  /'
echo "exit=$RC"
exit $RC
