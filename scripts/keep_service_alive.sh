#!/usr/bin/env bash
# =============================================================================
# 常驻守护：确保「最优版」服务在节点上持续可用（不随 agent / SSH 会话退出而消失）
#
# 为什么需要（2026-11-01）：
#   SCNet 上的 agent 会话随时可能中断，而 vLLM 服务是用 setsid 脱离会话启动的，
#   正常情况下应当存活；但一旦服务因异常退出（例如 OOM、脚本 rc 非 0 后的残留清理、
#   平台回收进程），30000 端口就会空掉，交付包里的 run.sh 语义要求"服务可一键拉起"。
#   本守护脚本负责：定期探测 → 掉了就按**最优版配置**原样拉起 → 写日志 → 不抢卡。
#
# 行为约束（重要）：
#   * 不修改 /opt/dtk、公共环境、模型权重
#   * 不启动压测客户端，不与压测抢卡
#   * 若检测到有压测在跑（bench_serving / run_full_bench），则**只观察不拉起**，
#     避免与压测链互相干扰（压测自己会管服务启停）
#   * 拉起前先确认端口空闲、进程无残留
#
# 用法（节点上 detached 启动）：
#   setsid nohup bash scripts/keep_service_alive.sh > logs/keep_service_alive.log 2>&1 &
# =============================================================================
set -uo pipefail

WS="${WS:-/root/private_data/workspace/robo}"
PORT="${PORT:-30000}"
INTERVAL="${INTERVAL:-120}"          # 探测间隔（秒）
LOG="$WS/logs/keep_service_alive.log"
cd "$WS" || exit 1
mkdir -p logs outputs tmp

log() { echo "[$(date -Is)] $*" >> "$LOG"; }

log "=== 守护启动（pid=$$ port=$PORT interval=${INTERVAL}s） ==="

while true; do
  sleep "$INTERVAL"

  # --- 1) 有压测在跑 → 只观察，不干预 -----------------------------------------
  if pgrep -f "bench_serving" >/dev/null 2>&1 \
     || pgrep -f "run_full_bench" >/dev/null 2>&1 \
     || pgrep -f "run_same_card_bench" >/dev/null 2>&1 \
     || pgrep -f "chain_16k64" >/dev/null 2>&1 \
     || pgrep -f "chain_c64_fill" >/dev/null 2>&1; then
    log "压测在跑，跳过本轮（不抢卡）"
    continue
  fi

  # --- 2) 服务健康 → 无需动作 -------------------------------------------------
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 \
         "http://127.0.0.1:${PORT}/health" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then
    continue
  fi

  # --- 3) 服务不在 → 尝试按最优版配置拉起 -------------------------------------
  log "服务不健康（health=$code），尝试拉起最优版服务"

  # 清残留（上一实例可能只是僵死）
  pkill -f "$WS/submission/scripts/serve_vllm_patched.py" 2>/dev/null
  pkill -f "VLLM::EngineCore" 2>/dev/null
  sleep 5
  pkill -9 -f "$WS/submission/scripts/serve_vllm_patched.py" 2>/dev/null
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  sleep 3

  # 端口占用检查：被别人占用则不拉起，避免冲突
  if pgrep -f "serve_vllm_patched" >/dev/null 2>&1; then
    log "仍有 serve 进程，跳过"
    continue
  fi

  # start_aiterfa_service.sh 自己会组装/校验调优库并等待 READY
  if bash "$WS/scripts/start_aiterfa_service.sh" "$PORT" >> "$LOG" 2>&1; then
    log "服务已拉起（最优版配置：graph + GEMM 调优库 + ROCM_AITER_FA）"
  else
    log "拉起失败，见上方日志"
  fi
done
