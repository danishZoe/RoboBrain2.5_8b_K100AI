#!/usr/bin/env bash
# Phase 1: environment and model inspection. Read-only collection.
set -uo pipefail
cd "$(dirname "$0")/.."
source ./env.sh

LOG_DIR="${WORK_DIR}/logs"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/env_check_${TS}.log"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG}") 2>&1

echo "########## RoboBrain2.5-8B-NV env check ${TS} ##########"

echo; echo "===== 1. host / container ====="
echo "hostname: $(hostname)"
echo "user    : $(whoami)"
echo "pwd     : $(pwd)"
grep -m1 PRETTY_NAME /etc/os-release 2>/dev/null
if [ -f /.dockerenv ]; then echo "container: yes (/.dockerenv)"; else echo "container: no"; fi
echo "cgroup  : $(head -1 /proc/1/cgroup 2>/dev/null | cut -c1-120)"
echo "docker  : $(command -v docker || echo 'not available (in-container)')"

echo; echo "===== 2. resource allocation / disk ====="
echo "nproc   : $(nproc)"
free -g | head -2
df -h "${WORK_DIR}" "${MODEL_PATH}" 2>/dev/null
python3 -c "
import os
p = os.path.join(os.environ['WORK_DIR'], 'tmp', '.wtest')
open(p, 'w').write('x')
os.unlink(p)
print('workdir write test: OK')
"

echo; echo "===== 3. accelerator / driver / runtime ====="
hy-smi 2>&1 | sed -n '1,12p'
echo "-- DTK version --"
/opt/dtk/bin/hipconfig --version 2>&1 | head -5
echo "-- device nodes --"
ls -la /dev/kfd /dev/dri/ 2>/dev/null

echo; echo "===== 4. python component versions ====="
python3 --version
python3 -m pip list 2>/dev/null | grep -iE "^(torch|torchvision|torchaudio|triton|transformers|vllm|sglang|sglang-kernel|accelerate|flash-attn|flash-mla|decord|av|qwen|pillow|numpy) "

echo; echo "===== 5. torch device discovery + minimal tensor compute ====="
python3 - <<'PYEOF'
import time
import torch
print("torch            :", torch.__version__)
print("device_count     :", torch.cuda.device_count())
print("device_name      :", torch.cuda.get_device_name(0))
print("capability       :", torch.cuda.get_device_capability(0))
print("bf16 supported   :", torch.cuda.is_bf16_supported())
free, total = torch.cuda.mem_get_info()
print("vram free/total  : %.2f GiB / %.2f GiB" % (free/1024**3, total/1024**3))
a = torch.randn(2048, 2048, device="cuda:0", dtype=torch.bfloat16)
b = torch.randn(2048, 2048, device="cuda:0", dtype=torch.bfloat16)
torch.cuda.synchronize()
t0 = time.time()
c = a @ b
torch.cuda.synchronize()
dt = time.time() - t0
print("matmul 2048^3 bf16: %.4f s -> %.1f TFLOPS" % (dt, 2*2048**3/dt/1e12))
print("result finite     :", bool(torch.isfinite(c).all()))
PYEOF

echo; echo "===== 6. model weight integrity ====="
echo "-- shards --"
ls -l "${MODEL_PATH}"/*.safetensors
echo "-- index vs actual files --"
python3 - "$MODEL_PATH" <<'PYEOF'
import json, os, sys
mp = sys.argv[1]
idx = json.load(open(os.path.join(mp, "model.safetensors.index.json")))
need = sorted(set(idx["weight_map"].values()))
have = sorted(f for f in os.listdir(mp) if f.endswith(".safetensors"))
print("index expects:", need)
print("files present:", have)
print("MISSING:", [f for f in need if f not in have] or "none")
print("total_size GB: %.2f" % (idx.get("metadata", {}).get("total_size", 0)/1024**3))
PYEOF
echo "-- config --"
python3 -c "import json;c=json.load(open('${MODEL_PATH}/config.json'));print('arch:',c['architectures']);print('model_type:',c['model_type']);print('dtype:',c['text_config']['dtype']);print('max_pos:',c['text_config']['max_position_embeddings']);print('vision depth:',c['vision_config']['depth'],'hidden:',c['vision_config']['hidden_size'])"
echo "-- component files --"
ls "${MODEL_PATH}" | grep -E "tokenizer|preprocessor|chat_template|configuration.json"

echo; echo "===== 7. backend support matrix ====="
python3 - <<'PYEOF'
try:
    from vllm.model_executor.models.registry import _MULTIMODAL_MODELS
    print("vllm Qwen3VL registered:", [k for k in _MULTIMODAL_MODELS if "Qwen3VL" in k])
except Exception as e:
    print("vllm check failed:", type(e).__name__, e)
try:
    import os, sglang
    p = os.path.join(os.path.dirname(sglang.__file__), "srt", "models")
    print("sglang qwen3_vl.py present:", os.path.isfile(os.path.join(p, "qwen3_vl.py")))
except Exception as e:
    print("sglang check failed:", type(e).__name__, e)
try:
    import transformers, os
    print("transformers", transformers.__version__,
          "qwen3_vl:", os.path.isdir(os.path.join(os.path.dirname(transformers.__file__), "models", "qwen3_vl")))
except Exception as e:
    print("transformers check failed:", type(e).__name__, e)
PYEOF

echo; echo "===== 8. network reachability ====="
for u in https://github.com https://huggingface.co https://pypi.org/simple/; do
  code=$(timeout 10 curl -s -o /dev/null -w "%{http_code}" "$u" 2>/dev/null); rc=$?
  echo "$u -> rc=$rc http=${code:-none}"
done

echo; echo "########## check finished, log: ${LOG} ##########"