#!/usr/bin/env bash
set -o pipefail
W=/root/private_data/workspace/robo
echo "===== 依赖 ====="
python3 - <<'PY'
for m in ("numpy", "PIL", "tqdm", "scipy", "pyarrow", "requests", "datasets"):
    try:
        mod = __import__(m)
        print("  %-10s %s" % (m, getattr(mod, "__version__", "ok")))
    except Exception as e:
        print("  %-10s 缺失: %s" % (m, type(e).__name__))
PY

echo
echo "===== 服务状态 ====="
if curl -sf --noproxy '*' http://127.0.0.1:30000/health >/dev/null; then
  echo "  vLLM 服务 OK (30000)"
else
  echo "  vLLM 服务 DOWN"
fi

echo
echo "===== 官方压测是否在跑（评测前必须为空） ====="
ps -eo pid,etime,cmd | grep bench_serving_vllm_image.py | grep -v grep || echo "  (无官方压测在跑)"

echo
echo "===== clean_repro 进度 ====="
ls -lt "$W/logs/clean_repro/" 2>/dev/null | head -8
tail -n 6 "$W/logs/pipeline_phase6.log" 2>/dev/null
