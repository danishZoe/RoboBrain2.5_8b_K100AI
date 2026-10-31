#!/usr/bin/env bash
# 独立安全网（脱离会话）：
#   监视 cgroup 内存用量；若超过阈值（默认 85 GiB，远低于 115 GiB 容器上限），
#   先杀掉 vLLM 服务与整条流水线，**在 kernel OOM killer 动手之前**停下，
#   避免容器被杀 → Pod 重建 → 节点不可达数小时 → 后续步骤与数据全丢。
#   已完成的压测数据同时存在于本地镜像，故此网只保护机器，不保护数据。
set -uo pipefail
LIMIT_GIB=${1:-85}
LIMIT_BYTES=$((LIMIT_GIB * 1024 * 1024 * 1024))
USAGE=/sys/fs/cgroup/memory/memory.usage_in_bytes
LOG=/root/private_data/workspace/robo/logs/guard_cgroup.log

{
  echo "[guard] started $(date -Is), threshold=${LIMIT_GIB} GiB"
  while true; do
    U=$(cat "$USAGE" 2>/dev/null || echo 0)
    if [ "$U" -gt "$LIMIT_BYTES" ]; then
      echo "[guard] $(date -Is) cgroup usage ${U} > ${LIMIT_BYTES} -> killing pipeline + vLLM"
      pkill -f 'pipeline_after_bench.sh' 2>/dev/null
      pkill -f 'run_non_eager_ab.sh' 2>/dev/null
      pkill -f 'run_batched_tokens_ab.sh' 2>/dev/null
      pkill -f 'collect_rocblas_trace.sh' 2>/dev/null
      pkill -f 'scripts/serve_vllm_patched.py' 2>/dev/null
      pkill -f 'VLLM::EngineCore' 2>/dev/null
      sleep 5
      pkill -9 -f 'scripts/serve_vllm_patched.py' 2>/dev/null
      pkill -9 -f 'VLLM::EngineCore' 2>/dev/null
      echo "[guard] killed; exiting"
      break
    fi
    sleep 3
  done
} >> "$LOG" 2>&1
