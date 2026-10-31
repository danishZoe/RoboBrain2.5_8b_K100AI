#!/usr/bin/env python3
"""给 submission/run.sh 打补丁：加入「最优版调优库组装」与「graph 启动期内存看门狗」。

只做两处定点插入，不改动其它内容；插入点缺失即报错退出。
"""
import shutil
import sys
from pathlib import Path

TARGET = Path(sys.argv[1] if len(sys.argv) > 1 else "submission/run.sh")
BAK = TARGET.with_suffix(".sh.pre_tuned.bak")

STEP_LIB = '''
# --- 1.4 最优版：组装 GEMM 算子调优库 ---------------------------------------
# 候选 rocBLAS 选优表只写到工作目录：把 stock 库复制到 $ROCBLAS_TENSILE_LIBPATH
# 再覆盖仓库内 6 个差异文件（4 改写 + 2 新增）。公共 /opt/dtk 库保持只读。
# 组装脚本幂等（已存在且哈希一致则跳过），哈希不符则启动失败——不允许带病启动。
if [ "${USE_TUNED:-1}" = "1" ] && [ -n "${ROCBLAS_TENSILE_LIBPATH:-}" ]; then
  echo "[run] assembling tuned rocBLAS library -> $ROCBLAS_TENSILE_LIBPATH"
  if ! bash "$REPO_DIR/scripts/make_tuned_rocblas_lib.sh"; then
    echo "[run] 调优库组装/校验失败，拒绝启动" >&2
    exit 1
  fi
fi
'''

STEP_WATCHDOG = '''
# --- 2.1 graph 启动期内存看门狗 ----------------------------------------------
# 2026-10-27 首次非 eager 启动在 inductor 编译期触顶容器 cgroup 上限（115 GiB），
# 被内核 OOM killer 杀掉并导致 Pod 重建。graph 模式启动期会做编译与图捕获，
# 这里轮询 cgroup 用量，超阈值（默认 70 GiB）就 SIGKILL 整个进程组，
# 让失败发生在我们自己手里而不是内核 OOM。
MEM_WATCHDOG_GIB="${MEM_WATCHDOG_GIB:-70}"
CGROUP_V1=/sys/fs/cgroup/memory/memory.usage_in_bytes
CGROUP_V2=/sys/fs/cgroup/memory.current
if [ -r "$CGROUP_V1" ]; then
  MEM_USAGE_CMD="cat $CGROUP_V1"
  MEM_LIMIT_BYTES=$((MEM_WATCHDOG_GIB * 1024 * 1024 * 1024))
elif [ -r "$CGROUP_V2" ]; then
  MEM_USAGE_CMD="cat $CGROUP_V2"
  MEM_LIMIT_BYTES=$((MEM_WATCHDOG_GIB * 1024 * 1024 * 1024))
else
  MEM_USAGE_CMD=""
  MEM_LIMIT_BYTES=0
fi
if [ -n "$MEM_USAGE_CMD" ]; then
  SERVER_PGID=$(ps -o pgid= -p "$SERVER_PID" 2>/dev/null | tr -d ' ')
  (
    PEAK=0
    while kill -0 "$SERVER_PID" 2>/dev/null; do
      U=$($MEM_USAGE_CMD 2>/dev/null || echo 0)
      if [ "$U" -gt "$PEAK" ]; then PEAK=$U; fi
      if [ "$MEM_LIMIT_BYTES" -gt 0 ] && [ "$U" -gt "$MEM_LIMIT_BYTES" ]; then
        echo "[watchdog] cgroup usage ${U} > ${MEM_LIMIT_BYTES} bytes; SIGKILL process group ${SERVER_PGID}" >> server.log
        kill -9 -"${SERVER_PGID}" 2>/dev/null
        break
      fi
      sleep 3
    done
    echo "[watchdog] peak_cgroup_bytes=${PEAK}" >> server.log
  ) &
  WATCHDOG_PID=$!
fi
'''

ANCHOR_LIB = "# --- 2) 拼启动命令（报告「4. 服务启动脚本」）----------------------------------"
ANCHOR_WD = '"${CMD[@]}" $EXTRA_ARGS >> server.log 2>&1 &\nSERVER_PID=$!\n'

text = TARGET.read_text(encoding="utf-8")
if "make_tuned_rocblas_lib.sh" in text:
    print("already patched, skip")
    sys.exit(0)

if ANCHOR_LIB not in text:
    sys.exit(f"ERROR: anchor not found: {ANCHOR_LIB!r}")
if ANCHOR_WD not in text:
    sys.exit(f"ERROR: anchor not found: {ANCHOR_WD!r}")

shutil.copy2(TARGET, BAK)
text = text.replace(ANCHOR_LIB, STEP_LIB.lstrip("\n") + "\n" + ANCHOR_LIB, 1)
text = text.replace(ANCHOR_WD, ANCHOR_WD + STEP_WATCHDOG, 1)
TARGET.write_text(text, encoding="utf-8")
print(f"patched: {TARGET} (backup: {BAK})")
