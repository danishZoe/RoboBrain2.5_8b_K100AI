#!/usr/bin/env bash
# probe_vllm_metrics.sh - 静态检查 vLLM 的 /metrics 暴露与指标名（不向运行中的服务发请求）
set -uo pipefail
SP=/usr/local/lib/python3.10/dist-packages/vllm
echo "=== vllm version ==="
python3 -c 'import vllm,os;print(vllm.__version__);print(os.path.dirname(vllm.__file__))' 2>&1 | tail -3
echo
echo "=== api_server metrics route ==="
grep -rn 'metrics' "$SP/entrypoints/openai/api_server.py" 2>/dev/null | head -20
echo
echo "=== num_gpu_blocks / kv cache metric names ==="
grep -rn 'num_gpu_blocks\|gpu_cache_usage' "$SP/v1/metrics/loggers.py" 2>/dev/null | head -20
echo
echo "=== KV cache size log line ==="
grep -rn 'GPU KV cache size\|Available KV cache memory' "$SP" 2>/dev/null | head -10
echo
echo "=== serving_chat / etc routes on live server (no inference) ==="
curl -s --noproxy '*' -o /dev/null -w 'metrics_http=%{http_code}\n' http://127.0.0.1:8000/metrics
curl -s --noproxy '*' http://127.0.0.1:8000/metrics 2>/dev/null | grep -E 'num_gpu_blocks|gpu_cache_usage|cache_config_info' | head -10
