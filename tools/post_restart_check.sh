#!/usr/bin/env bash
# 实例重启后的环境自检：GPU、CPU、内存、/usr/bin/time、探针有效性
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1

echo "===== 1. 时间/运行时长 ====="
date
uptime

echo
echo "===== 2. CPU 核数 ====="
nproc
grep -c ^processor /proc/cpuinfo

echo
echo "===== 3. 内存 (cgroup limit / current) ====="
cat /sys/fs/cgroup/memory.max 2>/dev/null
cat /sys/fs/cgroup/memory.current 2>/dev/null
cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo "(no memory.peak)"
free -g | head -2

echo
echo "===== 4. /usr/bin/time 是否存在（探针失效根因） ====="
ls -la /usr/bin/time 2>/dev/null || echo "MISSING /usr/bin/time  -> 探针的 /usr/bin/time -v 整条命令失败，判分从未执行"
command -v time || true
type time 2>/dev/null || true

echo
echo "===== 5. 探针各分块实际输出 ====="
for d in "$WS/tmp/probe3d"/*/; do
  [ -d "$d" ] || continue
  t=$(basename "$d")
  echo "--- $t ---"
  echo "  out.txt: $(wc -c < "$d/out.txt" 2>/dev/null) bytes"
  head -c 400 "$d/out.txt" 2>/dev/null
  echo
  echo "  err.txt: $(wc -c < "$d/err.txt" 2>/dev/null) bytes"
  head -c 400 "$d/err.txt" 2>/dev/null
  echo
done

echo
echo "===== 6. torch GPU 探测 ====="
export PYTHONPATH="$WS/envs/pylibs_core:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u
python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("device_count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, p.total_memory // 2**20, "MiB")
try:
    x = torch.randn(1024, 1024, device="cuda:0", dtype=torch.bfloat16)
    y = (x @ x).float().abs().mean().item()
    print("matmul ok, mean|x@x| =", y)
except Exception as e:
    print("matmul FAILED:", type(e).__name__, e)
PY

echo
echo "===== 7. 关键产物是否还在 ====="
for p in outputs/bench_public/tracespatial outputs/bench_public/cvbench outputs/bench_public/refspatial \
         outputs/bench_public_noneager/tracespatial outputs/bench_public_noneager/cvbench \
         outputs/bench_public/cvbench_protocol outputs/bench_public/stability; do
  n=$(ls -1 "$WS/$p" 2>/dev/null | wc -l)
  echo "  $p : $n entries"
done

echo
echo "===== 8. 端口 30000 ====="
(curl -s -m 3 -o /dev/null -w "health=%{http_code}\n" http://127.0.0.1:30000/health) 2>/dev/null || echo "health=000"
