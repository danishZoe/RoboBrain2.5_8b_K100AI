#!/usr/bin/env bash
# 诊断：chat template 是否支持 enable_thinking / 推理开关
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
MODEL=/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV

echo "########## 1. tokenizer_config.json 里的 chat_template ##########"
python3 - <<'PY'
import json, re
p = "/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV/tokenizer_config.json"
d = json.load(open(p, encoding="utf-8"))
ct = d.get("chat_template")
if isinstance(ct, list):
    for i, t in enumerate(ct):
        print("--- template[%d] len=%d ---" % (i, len(t)))
        print(t[:4000])
        print("...")
elif isinstance(ct, str):
    print(ct[:6000])
else:
    print("无 chat_template 字段；键:", [k for k in d.keys()][:40])
PY

echo
echo "########## 2. generation_config.json ##########"
cat "$MODEL/generation_config.json" 2>/dev/null || echo "(无)"

echo
echo "########## 3. config.json 关键字段 ##########"
python3 - <<'PY'
import json
p = "/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV/config.json"
d = json.load(open(p, encoding="utf-8"))
for k in ("model_type", "architectures", "torch_dtype", "max_position_embeddings",
          "vision_config", "text_config"):
    v = d.get(k)
    if k in ("vision_config", "text_config") and isinstance(v, dict):
        print(" ", k, "->", {kk: v[kk] for kk in list(v)[:6]})
    else:
        print(" ", k, "=", v)
PY

echo
echo "########## 4. 上游 RoboBrain2.5 README/demo 是否提到 thinking ##########"
grep -rni 'think\|reasoning\|enable_thinking' third_party/RoboBrain2.5/*.py third_party/RoboBrain2.5/*.md 2>/dev/null | head -20
