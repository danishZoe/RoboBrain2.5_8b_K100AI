#!/usr/bin/env bash
# 安装额外依赖、准备适配改动，再拉起 vLLM serving 并等待就绪。
#
# 与官方 run.sh 的两处必要差异（报告第 3.3 / 4 节逐条说明）：
#   1) 启动入口用 `python3 scripts/serve_vllm_patched.py` 而不是 `vllm serve`：
#      vLLM 用 spawn 起 EngineCore，父进程的 monkey-patch 不会被继承，
#      因此适配改动以 sitecustomize 垫片注入（每个解释器启动时自动 import）。
#   2) patch 不写进 site-packages：本容器的 site-packages 是公共只读环境。
#      脚本仍按官方流程尝试重放 patches/*.patch（先 dry-run 判重），
#      打不上时不报错退出，而是记录并依赖 sitecustomize 垫片 —— 二者等价。
set -euo pipefail

source ./env.sh
cd "$REPO_DIR"

# --- 1) 适配改动（报告「3. 适配优化方案」）------------------------------------

# 1.1 新增依赖装进工作目录内的隔离前缀，不污染公共 site-packages
if [ -f requirements-extra.txt ]; then
  if python -c "import qwen_vl_utils" 2>/dev/null; then
    echo "[run] skip requirements-extra.txt (already installed)"
  else
    echo "[run] installing requirements-extra.txt into $WORK_DIR/envs/pylibs"
    pip install -q --no-deps --target "$WORK_DIR/envs/pylibs" \
      -r requirements-extra.txt
  fi
fi

# 1.2 patch：**本节点不写入公共 site-packages**。
#     公共 Python 环境（/opt/dtk、/opt/hyhal、site-packages）按任务红线一律只读，
#     因此这里只做只读的判重检查（反向 dry-run），不执行 `patch` 落盘。
#     等价的改动由 envs/vllm_shim/sitecustomize.py 在运行时注入（见 1.3）；
#     patches/ 下的 diff 保留用于提交上游仓库。
for p in patches/*.patch; do
  [ -e "$p" ] || continue
  if patch -p1 -R -f -s --dry-run -d "$PATCH_ROOT" < "$p" > /dev/null 2>&1; then
    echo "[run] note: $p is already applied in $PATCH_ROOT (no write performed)"
  else
    echo "[run] note: $p not applied in $PATCH_ROOT; the public environment is"
    echo "        left untouched on purpose. The equivalent change is injected"
    echo "        at runtime by envs/vllm_shim/sitecustomize.py."
  fi
done

# 1.3 确认垫片可用（必须用**全新解释器**验证：
#     sitecustomize 在解释器启动时已被 site 自动 import，
#     在同一进程里再 import 一次是空操作）
ROBO_VLLM_TIE_SHIM=1 ROBO_MODEL_PATH="$MODEL_PATH" \
PYTHONPATH="$REPO_DIR/envs/vllm_shim:${PYTHONPATH:-}" \
python3 -c "
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig as C
print('[run] shim check: Qwen3VLTextConfig.tie_word_embeddings =', C.tie_word_embeddings)
"

# --- 2) 拼启动命令（报告「4. 服务启动脚本」）----------------------------------
if [ "$FRAMEWORK" = "sglang" ]; then
  CMD=(python -m sglang.launch_server --model-path "$MODEL_PATH"
       --served-model-name "$SERVED_NAME"
       --tp-size "$TP" --ep-size "$EP" --pp-size "$PP"
       --context-length "$MAX_LEN" --mem-fraction-static "$MEM_FRAC"
       --host 127.0.0.1 --port "$PORT" --trust-remote-code)
else
  CMD=(python3 "$REPO_DIR/scripts/serve_vllm_patched.py"
       --model "$MODEL_PATH" --served-model-name "$SERVED_NAME"
       --host 127.0.0.1 --port "$PORT"
       --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP"
       --max-model-len "$MAX_LEN" --gpu-memory-utilization "$MEM_FRAC"
       --dtype bfloat16 --trust-remote-code)
fi
if [ "$NNODES" -gt 1 ]; then
  CMD+=(--nnodes "$NNODES" --node-rank "$NODE_RANK" --dist-init-addr "$DIST_ADDR")
fi

echo "[run] ${CMD[*]} $EXTRA_ARGS" | tee server.log
"${CMD[@]}" $EXTRA_ARGS >> server.log 2>&1 &
SERVER_PID=$!

# --- 3) 轮询 /health，就绪即返回（不要用固定 sleep）---------------------------
for ((t = 0; t < LOAD_TIMEOUT; t += 5)); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[run] server exited, see server.log" >&2
    exit 1
  fi
  if curl -sf --noproxy "*" "http://127.0.0.1:$PORT/health" > /dev/null; then
    echo "[run] server up after ${t}s"
    exit 0
  fi
  sleep 5
done
echo "[run] server not ready in ${LOAD_TIMEOUT}s, see server.log" >&2
kill "$SERVER_PID"; exit 1
