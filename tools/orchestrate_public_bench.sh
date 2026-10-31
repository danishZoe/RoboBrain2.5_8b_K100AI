#!/usr/bin/env bash
# 公开基准评测编排：等依赖安装完 → 拆前缀 → 自检 → 等服务 → manifest → 三个基准 → 离线重判分
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1

URL="${1:-http://127.0.0.1:30000/v1/chat/completions}"
RUN_ID="${2:-baseline-eager}"

echo "===== [1/5] 等 pip 安装结束 ====="
for i in $(seq 1 120); do
  n=$(ps -ef | grep -c '[p]ip install' || true)
  [ "$n" -eq 0 ] && break
  echo "  pip 仍在运行 ($n)，等待 20s ..."
  sleep 20
done
echo "  pip 结束"

echo
echo "===== [2/5] 拆分隔离前缀（open3d + numpy2 -> envs/pylibs_core_o3d）====="
bash tools/prepare_core_prefix.sh 2>&1 | tail -n 40

echo
echo "===== [3/5] 代码自检 ====="
bash tools/selfcheck_bench_public.sh 2>&1 | grep -v 'NumPy 1.x\|NumPy 2.2.6\|versions of NumPy\|rebuild instead\|If you are a user\|downgrade to\|We expect that\|Traceback (most recent\|File \"\|    ' | tail -n 60

echo
echo "===== [4/5] 等服务就绪 ====="
BASE="${URL%/v1/chat/completions}"
for i in $(seq 1 90); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$BASE/health" || echo 000)
  if [ "$code" = "200" ]; then echo "  服务就绪 ($BASE)"; break; fi
  echo "  等待服务 ... ($code)"
  sleep 10
done
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$BASE/health" || echo 000)
[ "$code" = "200" ] || { echo "  服务未就绪，退出"; exit 1; }

echo
echo "===== [5/5] 跑全量评测 ====="
bash tools/run_public_bench.sh "$URL" "$RUN_ID" all
echo "[orchestrate] 退出码 $?"
