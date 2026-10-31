#!/usr/bin/env bash
set -euo pipefail

WS=/root/private_data/workspace/robo
MODEL=/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV
RUN_ID=${RUN_ID:?"set RUN_ID"}
STAGE=${1:?"usage: RUN_ID=id TIERS=4096:8 ROUNDS=2 bash scripts/run_prompt_retune.sh baseline|trace|candidate|quality"}
TIERS=${TIERS:-4096:8}
ROUNDS=${ROUNDS:-2}
PORT=30000
ROOT="$WS/outputs/prompt_retune/$RUN_ID"
OUT="$ROOT/$STAGE"
mkdir -p "$OUT"
cd "$WS"

record_time() {
  local phase=$1 started=$2 ended
  ended=$(date +%s)
  printf '%s\t%s\t%s\t%s\n' "$phase" "$started" "$ended" "$((ended-started))" >> "$ROOT/timings.tsv"
}

stop_service() {
  local f pid args
  for f in "$WS"/outputs/.bench_server_graph.pid "$WS"/outputs/.bench_server_trace.pid "$WS"/outputs/.bench_server_tunedlib.pid; do
    [ -f "$f" ] || continue
    pid=$(cat "$f")
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    args=$(ps -p "$pid" -o args= 2>/dev/null || true)
    if [[ "$args" == *"serve_vllm_patched.py"* ]]; then
      kill -TERM -- "-$pid" 2>/dev/null || true
      for _ in $(seq 1 15); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 2
      done
      kill -KILL -- "-$pid" 2>/dev/null || true
    fi
    rm -f "$f"
  done
}

start_service() {
  local started output log
  started=$(date +%s)
  stop_service
  case "$STAGE" in
    baseline)
      output=$(env -u ROCBLAS_TENSILE_LIBPATH -u VLLM_ATTENTION_BACKEND -u VLLM_ROCM_USE_AITER \
        bash tools/start_bench_service.sh graph "$PORT" 2>&1) ;;
    trace)
      output=$(env -u ROCBLAS_TENSILE_LIBPATH -u VLLM_ATTENTION_BACKEND -u VLLM_ROCM_USE_AITER \
        bash scripts/start_trace_service.sh 2>&1) ;;
    candidate*)
      output=$(bash scripts/start_candidate_service.sh 2>&1) ;;
    *) return 0 ;;
  esac
  printf '%s\n' "$output" | tee "$OUT/service_start.log"
  log=$(printf '%s\n' "$output" | sed -n 's/.*log=\([^ ]*\).*/\1/p' | tail -1)
  [ -n "$log" ] && printf '%s\n' "$log" > "$OUT/service_log_path.txt"
  record_time "${STAGE}_startup" "$started"
}

run_bench() {
  local started tier ilen conc n r tag
  started=$(date +%s)
  set +u
  source ./env.sh >/dev/null 2>&1
  set -u
  IFS=',' read -ra tier_array <<< "$TIERS"
  for tier in "${tier_array[@]}"; do
    ilen=${tier%%:*}
    conc=${tier##*:}
    n=$((conc * 2 > 8 ? conc * 2 : 8))
    for r in $(seq "${ROUND_START:-1}" "$ROUNDS"); do
      tag="round${r}_${ilen}_1024_c${conc}"
      echo "[bench] $STAGE $tag"
      python3 scripts/bench_serving_vllm_image.py \
        --backend vllm --host 127.0.0.1 --port "$PORT" \
        --model robobrain2.5-8b-nv --tokenizer "$MODEL" \
        --dataset-name image --image-count 1 --image-resolution 1080p \
        --random-input-len "$ilen" --random-output-len 1024 --random-range-ratio 1 \
        --max-concurrency "$conc" --num-prompts "$n" --seed 1 \
        --output-file "$OUT/round${r}.jsonl" > "$OUT/${tag}.log" 2>&1
      grep -E 'Successful requests|Failed requests|Benchmark duration|Total generated tokens|Output token throughput|Mean TTFT|Mean TPOT|Mean E2E' "$OUT/${tag}.log" || true
    done
  done
  record_time "${STAGE}_bench" "$started"
}

if [ ! -f "$ROOT/run.meta" ]; then
  {
    echo "run_id=$RUN_ID"
    echo "created_epoch=$(date +%s)"
    echo "created_at=$(date --iso-8601=seconds)"
    echo "tiers=$TIERS rounds=$ROUNDS"
    echo "protocol=round1_warmup_round2_record; image=1x1080p; output=1024; seed=1; prefix_cache=off"
  } > "$ROOT/run.meta"
  sha256sum samples/tasks_official.json scripts/bench_serving_vllm_image.py \
    scripts/probe_http_request.py submission/scripts/serve_vllm_patched.py > "$ROOT/input_hashes.sha256"
fi

case "$STAGE" in
  baseline|trace|candidate*|rmsnorm*|linear*|hipprof*)
    if [ "${REUSE_SERVICE:-0}" = 1 ]; then
      curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/health" >/dev/null
    else
      start_service
    fi
    run_bench
    if [ "$STAGE" = trace ]; then
      stop_service
      log=$(cat "$OUT/service_log_path.txt")
      bash scripts/extract_fc_size.sh "$log" "$OUT/fc_size.log"
      cp "$log" "$OUT/service_trace.log"
    fi
    ;;
  quality*)
    started=$(date +%s)
    python3 scripts/probe_http_request.py --backend vllm --port "$PORT" \
      --tasks samples/tasks_official.json --image-dir samples/images \
      --repeat 1 --out-dir "$OUT/probe" > "$OUT/probe.log" 2>&1
    python3 scripts/compare_backends.py \
      --baseline outputs/phase3_baseline/baseline_raw.json \
      --candidate "$OUT/probe/probe_result.json" --out "$OUT/agreement.md" > "$OUT/compare.log" 2>&1
    record_time "$STAGE" "$started"
    cat "$OUT/agreement.md"
    ;;
esac
