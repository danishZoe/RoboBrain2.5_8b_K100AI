#!/usr/bin/env bash
set -uo pipefail
W=/root/private_data/workspace/robo
RUN_ID=${RUN_ID:-clean_repro_full_$(date +%Y%m%d_%H%M%S)}
OUT="$W/logs/clean_repro_full/$RUN_ID"
BENCH_OUT="$OUT/bench"
mkdir -p "$BENCH_OUT"
cd "$W" || exit 1

if pgrep -f 'bench_serving|bash bench.sh' >/dev/null 2>&1; then
  echo "另一个压测仍在运行，拒绝启动" >&2
  exit 3
fi
if pgrep -f "$W/submission/scripts/serve_vllm_patched.py" >/dev/null 2>&1; then
  echo "提交包服务已在运行，请先停止该服务" >&2
  exit 4
fi

printf 'run_id=%s\nstarted_at=%s\n' "$RUN_ID" "$(date -Is)" | tee "$OUT/run_config.txt"
cd "$W/submission" || exit 1
source ./env.sh
printf 'framework=%s\ncard=%s\nprecision=%s\ntp=%s\nmax_len=%s\nmem_frac=%s\nextra_args=%s\n' \
  "$FRAMEWORK" "$CARD" "$PRECISION" "$TP" "$MAX_LEN" "$MEM_FRAC" "$EXTRA_ARGS" \
  | tee -a "$OUT/run_config.txt"

T0=$(date +%s)
if ! bash run.sh > "$OUT/server_start.log" 2>&1; then
  echo "服务启动失败，见 $OUT/server_start.log" >&2
  exit 1
fi
echo "server_ready_seconds=$(( $(date +%s) - T0 ))" | tee -a "$OUT/run_config.txt"

if [ "${SKIP_CHECKS:-0}" != 1 ]; then
  if ! bash env_check.sh > "$OUT/env_check.log" 2>&1; then
    echo "环境检查失败" >&2
    exit 1
  fi
  cp -f env.log "$OUT/env.log"
  if ! bash smoke.sh > "$OUT/smoke.log" 2>&1; then
    echo "Smoke 失败，见 $OUT/smoke.log" >&2
    exit 1
  fi
  python3 verify.py > "$OUT/verify_stdout.log" 2>&1
  VRC=$?
  cp -f verify_result.json "$OUT/verify_result.json"
  if [ "$VRC" != 0 ] && [ "$VRC" != 2 ]; then
    echo "verify.py 异常退出：$VRC" >&2
    exit 1
  fi
fi

cd "$W" || exit 1
HF_SNAP="$HF_HOME/hub/models--${SERVED_NAME}/snapshots/main"
mkdir -p "$(dirname "$HF_SNAP")"
if [ ! -e "$HF_SNAP" ]; then
  ln -sfn "$MODEL_PATH" "$HF_SNAP"
fi

BENCH_RC=0
for ILEN in 4096 16384; do
  for C in 1 8 64; do
    if [ -n "${BENCH_TIERS:-}" ]; then
      case ",$BENCH_TIERS," in
        *",$ILEN:$C,"*) ;;
        *) continue ;;
      esac
    fi
    N=$(( C * 2 > 8 ? C * 2 : 8 ))
    for R in 1 2; do
      TAG="round${R}_${ILEN}_1024_c${C}"
      echo "[$(date -Is)] $TAG"
      python3 "$W/submission/scripts/bench_serving_vllm_image.py" \
        --backend vllm --host 127.0.0.1 --port "$PORT" \
        --model "$SERVED_NAME" --tokenizer "$MODEL_PATH" \
        --dataset-name image --image-count 1 --image-resolution 1080p \
        --random-input-len "$ILEN" --random-output-len 1024 --random-range-ratio 1 \
        --max-concurrency "$C" --num-prompts "$N" --seed 1 \
        --output-file "$BENCH_OUT/round${R}.jsonl" \
        > "$BENCH_OUT/${TAG}.log" 2>&1
      RC=$?
      echo "[$(date -Is)] $TAG exit=$RC"
      if [ "$RC" != 0 ]; then BENCH_RC=1; fi
    done
  done
done

python3 - "$W" "$BENCH_OUT" "$OUT" <<'PY'
import json, os, sys
w, bench, out = sys.argv[1:]
rounds = {}
for rnd in (1, 2):
    path = os.path.join(bench, f"round{rnd}.jsonl")
    rows = []
    if os.path.exists(path):
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    rounds[rnd] = rows
base_rows = [json.loads(line) for line in open(
    os.path.join(w, "submission/bench/round2.jsonl"), encoding="utf-8") if line.strip()]
base = {(r["random_input_len"], r["max_concurrency"]): r for r in base_rows}
repro = {(r["random_input_len"], r["max_concurrency"]): r for r in rounds[2]}
order = [(i, c) for i in (4096, 16384) for c in (1, 8, 64)]
result = {"round1_entries": len(rounds[1]), "round2_entries": len(rounds[2]), "rows": []}
ok = len(rounds[1]) == 6 and len(rounds[2]) == 6
lines = [
    "| 输入 | 并发 | 完成/计划 | 基线 tok/s | 复现 tok/s | 吞吐偏差 | TTFT偏差 | TPOT偏差 |",
    "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
]
for key in order:
    ilen, conc = key
    expected = max(8, conc * 2)
    b, r = base.get(key), repro.get(key)
    row = {"input": ilen, "concurrency": conc, "expected": expected}
    if not b or not r:
        row["status"] = "MISSING"
        ok = False
        lines.append(f"| {ilen} | {conc} | MISSING | — | — | — | — | — |")
        result["rows"].append(row)
        continue
    throughput_dev = (r["output_throughput"] / b["output_throughput"] - 1) * 100
    ttft_dev = (r["mean_ttft_ms"] / b["mean_ttft_ms"] - 1) * 100
    tpot_dev = (r["mean_tpot_ms"] / b["mean_tpot_ms"] - 1) * 100
    failed = r.get("failed", 0)
    row.update({
        "completed": r.get("completed"), "failed": failed,
        "baseline_output_throughput": b["output_throughput"],
        "repro_output_throughput": r["output_throughput"],
        "throughput_deviation_percent": throughput_dev,
        "ttft_deviation_percent": ttft_dev, "tpot_deviation_percent": tpot_dev,
        "status": "PASS" if r.get("completed") == expected and not failed and abs(throughput_dev) <= 20 else "FAIL",
    })
    if row["status"] != "PASS":
        ok = False
    lines.append(
        f"| {ilen} | {conc} | {r.get('completed')}/{expected}，失败 {failed} | "
        f"{b['output_throughput']:.2f} | {r['output_throughput']:.2f} | "
        f"{throughput_dev:+.2f}% | {ttft_dev:+.2f}% | {tpot_dev:+.2f}% |")
    result["rows"].append(row)
result["passed"] = ok
result["criterion"] = "六档两轮齐全；第二轮完成数符合计划、失败为0、输出吞吐相对偏差绝对值不超过20%"
with open(os.path.join(out, "comparison.json"), "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n\n干净复现结论：" + ("PASS" if ok else "FAIL") + "\n")
print("\n".join(lines))
print("干净复现结论:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
SUMMARY_RC=$?
cp -f "$W/submission/server.log" "$OUT/server.log"
printf 'finished_at=%s\n' "$(date -Is)" >> "$OUT/run_config.txt"

pkill -TERM -f "$W/submission/scripts/serve_vllm_patched.py" 2>/dev/null || true
sleep 8
if [ "$BENCH_RC" != 0 ] || [ "$SUMMARY_RC" != 0 ]; then
  exit 1
fi
exit 0
