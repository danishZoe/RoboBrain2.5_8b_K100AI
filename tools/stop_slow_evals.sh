#!/usr/bin/env bash
set -uo pipefail
WS=/root/private_data/workspace/robo
cd "$WS"

echo "=== ① 停掉卡住的 TraceSpatial 3D 生产判分（4 个深景病态样本，已跑 1.5h）==="
PID=$(pgrep -f 'score_3d_chunked.py --job' | head -1)
if [ -n "${PID:-}" ]; then
  echo "  父进程 $PID；先杀子进程（官方判分）再杀父进程"
  pkill -TERM -P "$PID" 2>/dev/null || true
  kill -TERM "$PID" 2>/dev/null || true
  sleep 5
  pkill -KILL -P "$PID" 2>/dev/null || true
  kill -KILL "$PID" 2>/dev/null || true
  echo "  已停止（产物未写出，96 题兜底口径为准）"
else
  echo "  生产判分已不在运行"
fi

echo
echo "=== ② 停掉 up4 上采样协议（23 min 未出结果，收益边际）==="
UP=$(pgrep -f 'exp_cvbench_upscale.sh' | head -1)
if [ -n "${UP:-}" ]; then
  echo "  停 $UP"
  pkill -TERM -P "$UP" 2>/dev/null || true
  kill -TERM "$UP" 2>/dev/null || true
  sleep 3
  pkill -KILL -P "$UP" 2>/dev/null || true
  kill -KILL "$UP" 2>/dev/null || true
  # 连带停掉 run_cv_bench.py
  pkill -f 'run_cv_bench.py.*cvbench_upscale' 2>/dev/null || true
  echo "  已停止"
else
  echo "  上采样实验已不在运行"
fi

echo
echo "=== ③ 剩余进程 ==="
echo "  官方判分子进程: $(ps -eo args | grep -c '[s]ummarize_acc.py')"
echo "  vLLM 服务:      $(pgrep -fc 'serve_vllm_patched.py' || echo 0)"
echo "  链式实验:       $(pgrep -fc 'chain_cvbench.sh' || echo 0)"
echo
echo "=== ④ 对拍结果（已落盘）==="
python3 - <<'PY'
import json, os
WS = "/root/private_data/workspace/robo"
p = WS + "/outputs/bench_public/verify_3d_merge.json"
d = json.load(open(p, encoding="utf-8"))
print("  all_match =", d.get("all_match"), " n_samples =", d.get("n_samples"),
      " offset =", d.get("offset"))
bad = [r for r in d.get("rows", []) if not r.get("match")]
print("  不一致项 =", len(bad))
for r in d.get("rows", []):
    print("   %-26s %-30s 整集A=%-14s 分片B=%-14s %s" % (
        r["case"], r["metric"], r.get("single_full"), r.get("chunk1_merged"),
        "MATCH" if r.get("match") else "*** MISMATCH ***"))
PY
