#!/usr/bin/env bash
# =============================================================================
# SGLang aiter 兼容垫片验证
#
#   bash tools/try_sglang_aiter.sh --stage0    只做导入期验证（不需要 GPU，可随时跑）
#   bash tools/try_sglang_aiter.sh --full      起服务 + 发一次真实图文请求（需要空闲 GPU）
#
# 纪律：不改 site-packages、不改权重/采样；垫片只补导出与模块别名。
# 必须在官方压测与收尾流水线结束之后跑 --full，避免污染性能测量。
# =============================================================================
set -o pipefail

W=/root/private_data/workspace/robo
cd "$W" || exit 1
source ./env.sh >/dev/null 2>&1

MODE="${1:---stage0}"
OUTDIR="$W/outputs/sglang_aiter"
mkdir -p "$OUTDIR" "$W/logs"

SHIM_DIR="$W/envs/sglang_shim"
LOG="$W/logs/sglang_aiter_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================"
echo "SGLang aiter 垫片验证  mode=$MODE  $(date -Is)"
echo "shim: $SHIM_DIR/sitecustomize.py"
echo "log : $LOG"
echo "============================================================"

# ---------------------------------------------------------------- stage 0
echo ""
echo "=== stage0：导入期符号补齐验证（不启服务） ==="
PYTHONPATH="$SHIM_DIR:${PYTHONPATH:-}" ROBO_AITER_SHIM=1 python3 - <<'PY' 2>&1 | tee -a "$LOG"
import importlib, sys

want_top = ["flash_attn_varlen_func", "get_mla_metadata_info_v1", "get_mla_metadata_v1",
            "get_ps_metadata_info_v1", "get_ps_metadata_v1", "mha_batch_prefill_func",
            "mla_prefill_ps_asm_fwd", "mla_reduce_v1", "paged_attention_ragged"]

aiter = importlib.import_module("aiter")
print("aiter:", aiter.__file__)

missing = [n for n in want_top if not hasattr(aiter, n)]
for n in want_top:
    obj = getattr(aiter, n, None)
    kind = "MISSING"
    if obj is not None:
        kind = "placeholder" if getattr(obj, "__doc__", "") and "RoboBrain compat placeholder" in (obj.__doc__ or "") else "real"
    print("  aiter.%-26s %s" % (n, kind))

try:
    from aiter.mla import mla_decode_fwd, mla_prefill_fwd
    print("  aiter.mla.mla_decode_fwd / mla_prefill_fwd : OK")
except Exception as e:
    print("  aiter.mla import FAILED:", repr(e))

try:
    from aiter.ops.triton.attention.unified_attention import unified_attention
    print("  aiter.ops.triton.attention.unified_attention : OK ->", unified_attention)
except Exception as e:
    print("  aiter.ops.triton.attention.unified_attention import FAILED:", repr(e))

print()
print(">>> 结论：stage0 %s" % ("PASS" if not missing else "FAIL(missing=%s)" % missing))
sys.exit(0 if not missing else 1)
PY
RC0=${PIPESTATUS[0]}
echo "stage0 退出码: $RC0"

if [ "$MODE" = "--stage0" ]; then
  exit $RC0
fi

# ---------------------------------------------------------------- 占用检查
echo ""
echo "=== 占用检查：压测/流水线在跑则拒绝启动 ==="
BUSY=0
for pat in bench_serving bench.sh pipeline_after_bench dryrun; do
  if pgrep -f "$pat" >/dev/null 2>&1; then echo "  仍��在运行: $pat"; BUSY=1; fi
done
if [ "$BUSY" = "1" ]; then
  echo "  REFUSE：当前有压测/流水线在跑，GPU 数据会被污染。稍后重试。"
  exit 3
fi
echo "  GPU 空闲，可以跑"

# ---------------------------------------------------------------- 起服务
PORT=30011
SRV_LOG="$W/logs/sglang_aiter_server_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "=== 启动 SGLang（--attention-backend aiter + 垫片） ==="
echo "  server log: $SRV_LOG"

PYTHONPATH="$SHIM_DIR:${PYTHONPATH:-}" ROBO_AITER_SHIM=1 \
setsid sglang serve \
  --model-path "${MODEL_PATH}" \
  --served-model-name robobrain2.5-8b-nv \
  --host 127.0.0.1 --port "${PORT}" \
  --tp-size 1 --dtype bfloat16 \
  --mem-fraction-static 0.85 --context-length 8192 \
  --max-running-requests 16 \
  --limit-mm-data-per-request '{"image": 8}' \
  --trust-remote-code \
  --attention-backend aiter > "${SRV_LOG}" 2>&1 &
SRV_PID=$!
echo "  pid=$SRV_PID"

READY=0
for i in $(seq 1 90); do
  if ! kill -0 "${SRV_PID}" 2>/dev/null; then
    echo "  服务进程已退出"; break
  fi
  if curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1; echo "  服务就绪（约 $((i*10))s）"; break
  fi
  sleep 10
done

if [ "${READY}" != "1" ]; then
  echo "  RESULT: NOT READY —— 决定性日志行："
  grep -E "Invalid address|Error code|Fatal Python|NameError|ModuleNotFound|ImportError|scheduler died|HSA_STATUS|VMFault|assert" "${SRV_LOG}" 2>/dev/null | head -10 | sed 's/^/    /'
  kill -TERM -- "-${SRV_PID}" 2>/dev/null; sleep 8; kill -KILL -- "-${SRV_PID}" 2>/dev/null
  exit 1
fi

# ---------------------------------------------------------------- 发请求
echo ""
echo "=== 发一次真实图文请求 ==="
python3 scripts/probe_one_request.py --port "${PORT}" \
  --image "${WORK_DIR}/samples/images/official_affordance.jpg" \
  --prompt "What is shown in this image?" \
  --out "${OUTDIR}/aiter_answer.json" 2>&1 | tee -a "$LOG"
RC1=${PIPESTATUS[0]}

echo ""
echo "=== 与 transformers 直跑基线比对 ==="
python3 - "$OUTDIR/aiter_answer.json" <<'PY' 2>&1 | tee -a "$LOG"
import json, os, sys
p = sys.argv[1]
if not os.path.exists(p):
    print("  没有产出答案文件"); raise SystemExit
a = json.load(open(p, encoding="utf-8"))
ans = (a.get("answer") or "").strip()
print("  sglang/aiter 答案:", ans[:220])

bp = "/root/private_data/workspace/robo/outputs/phase3_baseline/baseline_raw.json"
if os.path.exists(bp):
    d = json.load(open(bp, encoding="utf-8"))
    recs = d.get("records") if isinstance(d, dict) else d
    for r in (recs or []):
        if r.get("task") == "general" and not r.get("warmup"):
            print("  transformers 基线:", (r.get("answer") or "")[:220])
            break
print("  （人工核对语义是否一致；不是官方精度指标）")
PY

kill -TERM -- "-${SRV_PID}" 2>/dev/null; sleep 8; kill -KILL -- "-${SRV_PID}" 2>/dev/null
echo ""
echo "RESULT: request_rc=$RC1   server_log=$SRV_LOG"
