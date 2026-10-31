#!/usr/bin/env bash
# 修复：隔离前缀里残留的 numpy 元数据（只有 dist-info、没有包本体）
#
# 症状：importlib.metadata.version("numpy") 报 2.2.6，但 import numpy 拿到系统 1.25
#       -> accelerate 的 np_core = np._core if is_numpy_available("2.0.0") else np.core
#          判定走 numpy>=2 分支 -> numpy 1.25 没有 _core -> AttributeError
# 处置：把孤立的 dist-info 移到 _quarantine/（可回退），不删除
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1

for P in envs/pylibs_core envs/pylibs; do
  [ -d "$P" ] || continue
  echo "--- $P ---"
  # 有 numpy 包本体吗？
  if [ -d "$P/numpy" ]; then
    echo "  有 numpy 包本体 -> 元数据与实现一致，无需处理"
    continue
  fi
  for D in "$P"/numpy-*.dist-info; do
    [ -e "$D" ] || continue
    mkdir -p "$P/_quarantine"
    echo "  隔离孤立元数据: $D"
    mv "$D" "$P/_quarantine/$(basename "$D").$(date +%s)"
  done
done

echo
echo "===== 验证 ====="
PYTHONPATH="$WS/envs/pylibs_core" python3 - <<'PY'
import numpy
from importlib.metadata import version
print("  numpy.__version__ =", numpy.__version__, numpy.__file__)
print("  dist numpy version =", version("numpy"))
import accelerate.utils.other  # 之前在这里炸
print("  accelerate import OK")
PY

echo
echo "===== 启动 graph 服务（E1 候选配置） ====="
bash "$WS/tools/start_bench_service.sh" graph 30000 2>&1 | tail -n 6
