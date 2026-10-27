#!/usr/bin/env bash
# 采集验证环境写入 env.log；报告「2. 验证环境」的版本表直接取自这里。
# 在 run.sh 之后执行，记录的是升级 / 打 patch 之后实际生效的版本。
#
# 注意：不能在 `set -u` 下 source ./env.sh（/opt/dtk/env.sh 引用未定义变量）。
source ./env.sh

{
  echo "image: $IMAGE"
  echo "dtk: $(cat /opt/dtk/.info/version 2>/dev/null || readlink -f /opt/dtk)"
  echo "model_id: $MODEL_ID"
  echo "model_path: $MODEL_PATH"
  echo "served_name: $SERVED_NAME"
  echo "framework: $FRAMEWORK  card: $CARD  precision: $PRECISION"
  echo "tp: $TP  max_len: $MAX_LEN  mem_frac: $MEM_FRAC  port: $PORT"
  echo "--- hy-smi ---"
  hy-smi
  echo "--- runtime ---"
  python -c "import sys, torch; print('python:', sys.version.split()[0]); print('torch:', torch.__version__); print('HIP:', torch.version.hip); print('Device:', torch.cuda.get_device_name(0), 'x', torch.cuda.device_count())"
  echo "--- packages ---"
  pip list 2>/dev/null | grep -iE "^(torch|triton|transformers|tokenizers|sglang|vllm|flash|fla|lmslim|ray|qwen)|das"
  echo "--- extra deps (isolated, see requirements-extra.txt) ---"
  ls "$WORK_DIR/envs/pylibs" 2>/dev/null | grep -iE "dist-info" || true
  echo "--- model config facts ---"
  python - <<'PY'
import json, os
p = os.path.join(os.environ["MODEL_PATH"], "config.json")
c = json.load(open(p, encoding="utf-8"))
print("architectures:", c.get("architectures"))
print("tie_word_embeddings:", c.get("tie_word_embeddings"))
print("vision depth:", (c.get("vision_config") or {}).get("depth"))
idx = os.path.join(os.environ["MODEL_PATH"], "model.safetensors.index.json")
wm = json.load(open(idx, encoding="utf-8")).get("weight_map", {})
print("n tensors:", len(wm))
print("shards:", sorted(set(wm.values())))
print("lm_head present:", any("lm_head" in k for k in wm))
PY
  echo "--- container limits ---"
  echo "cgroup memory limit: $(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null) bytes"
  echo "cgroup memory usage: $(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null) bytes"
} 2>&1 | tee env.log
