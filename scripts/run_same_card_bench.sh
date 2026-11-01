#!/usr/bin/env bash
# =============================================================================
# 同一张卡上依次跑三套配置：首提基线 eager / graph 基线 / 最优版
#
# 背景（2026-11-01）：SCNet 容器被平台回收重建后**物理卡会更换**，跨卡的绝对值
#   不可比（实测同配置跨卡可差约 30%）。要给出可信的「提升」，必须在
#   **同一张卡**上同时测出 eager（赛事首次提交口径）与最优版（提交配置）两套结果。
#
# 三套配置：
#   eager     = --enforce-eager + stock rocBLAS 库 + 默认 attention（赛事首提口径）
#   baseline  = 非 eager + stock 库 + 默认 attention（只开 graph，用于收益归因）
#   candidate = 非 eager + GEMM 调优库 + ROCM_AITER_FA（当前提交配置）
#
# 用法：
#   bash scripts/run_same_card_bench.sh \
#       --tiers "4096:1:2,4096:8:2,16384:1:2,16384:8:2,4096:64:2,16384:64:1" \
#       --baseline-tiers "4096:1:2,4096:8:2,16384:1:2,16384:8:2" \
#       --port 30000
#
#   --skip-eager / --skip-baseline / --skip-candidate   跳过对应配置
#   --smoke 0    关闭每套配置后的功能冒烟（默认开）
#
# 退出码：0 = 三套（或被跳过的部分）都跑完
# =============================================================================
set -uo pipefail

WS="${WS:-/root/private_data/workspace/robo}"
PORT="${PORT:-30000}"
TIERS="${TIERS:-4096:1:2,4096:8:2,16384:1:2,16384:8:2,4096:64:2,16384:64:1}"
BASELINE_TIERS="${BASELINE_TIERS:-4096:1:2,4096:8:2,16384:1:2,16384:8:2}"
RUN_EAGER=1
RUN_BASELINE=1
RUN_CANDIDATE=1
SKIP_SMOKE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --tiers) TIERS="$2"; shift 2 ;;
    --baseline-tiers) BASELINE_TIERS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --skip-eager) RUN_EAGER=0; shift ;;
    --skip-baseline) RUN_BASELINE=0; shift ;;
    --skip-candidate) RUN_CANDIDATE=0; shift ;;
    --smoke) SKIP_SMOKE="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

cd "$WS" || exit 1

# --- 卡指纹（决定这批数据能不能和别的卡相减）---------------------------------
# shellcheck disable=SC1091
source "$WS/scripts/card_identity.sh" 2>/dev/null || true
echo "================================================================"
echo "同一张卡三套压测"
echo "================================================================"
echo "card_id=${CARD_ID:-unknown}"
echo "card_uid=${CARD_UID:-}"
echo "card_name=${CARD_NAME:-}"
echo "card_gfx=${CARD_GFX:-}"
echo "node_host=${CARD_HOST:-}"
echo "ws=$WS"
echo "port=$PORT"
echo "tiers=$TIERS"
echo "baseline_tiers=$BASELINE_TIERS"
echo "started_at=$(date -Is)"
echo

# --- 调优库先组装好（candidate 依赖；缺失/哈希不符会直接失败）-----------------
echo "[same-card] 组装 GEMM 调优库"
if ! bash "$WS/submission/scripts/make_tuned_rocblas_lib.sh"; then
  echo "[same-card] ERROR: 调优库组装/校验失败，先修好再跑 candidate" >&2
  exit 3
fi
echo

LOGDIR="$WS/logs/same_card_bench_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

FAILED=""
run_one() {
  cfg="$1"; tiers="$2"; label="$3"
  echo
  echo "================================================================"
  echo "[same-card] 开始：$label（config=$cfg, tiers=$tiers）  $(date -Is)"
  echo "================================================================"
  SKIP_SMOKE_ARG=""
  [ "$SKIP_SMOKE" = "1" ] && SKIP_SMOKE_ARG="--skip-smoke"
  bash "$WS/scripts/run_full_bench.sh" --config "$cfg" --tiers "$tiers" \
       --port "$PORT" $SKIP_SMOKE_ARG 2>&1 | tee "$LOGDIR/${cfg}.log"
  rc=${PIPESTATUS[0]}
  echo "[same-card] 结束：$label  rc=$rc  $(date -Is)"
  if [ "$rc" != "0" ]; then
    FAILED="$FAILED $label(rc=$rc)"
  fi
  # 把本次运行的目录记下来，方便后面核对
  ls -dt "$WS/outputs/fullbench_${cfg}_"* 2>/dev/null | head -1 > "$LOGDIR/${cfg}.rundir"
  echo "[same-card] run_dir=$(cat "$LOGDIR/${cfg}.rundir" 2>/dev/null)"
}

# 顺序：eager（首提基线）→ baseline（graph 归因）→ candidate（最优版）
[ "$RUN_EAGER" = "1" ] && run_one eager "$TIERS" "首提基线 eager（赛事首提口径）"
[ "$RUN_BASELINE" = "1" ] && run_one baseline "$BASELINE_TIERS" "graph 基线（只开 graph，归因用）"
[ "$RUN_CANDIDATE" = "1" ] && run_one candidate "$TIERS" "最优版（提交配置）"

echo
echo "================================================================"
echo "完成 $(date -Is)"
echo "日志目录：$LOGDIR"
for c in eager baseline candidate; do
  [ -f "$LOGDIR/${c}.rundir" ] && echo "  $c → $(cat "$LOGDIR/${c}.rundir")"
done
if [ -n "$FAILED" ]; then
  echo "以下配置未成功：$FAILED"
  exit 4
fi
echo "全部成功。"
