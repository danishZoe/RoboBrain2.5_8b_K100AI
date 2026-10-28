#!/usr/bin/env bash
set -uo pipefail
WS=/root/private_data/workspace/robo
M=/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV

echo "########## 1. 模型目录文件 ##########"
ls -la "$M" | head -40

echo
echo "########## 2. chat_template 是否存在 + thinking 开关 ##########"
python3 - <<'PY'
import json, os, re
M = "/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV"
tc = os.path.join(M, "tokenizer_config.json")
d = json.load(open(tc, encoding="utf-8"))
ct = d.get("chat_template")
print("tokenizer_config 里含 chat_template:", bool(ct))
if ct is None:
    p = os.path.join(M, "chat_template.json")
    if os.path.exists(p):
        ct = json.load(open(p, encoding="utf-8")).get("chat_template")
        print("chat_template.json 里含 chat_template:", bool(ct))
if ct:
    s = ct if isinstance(ct, str) else json.dumps(ct, ensure_ascii=False)
    print("chat_template 长度:", len(s))
    for kw in ("enable_thinking", "think", "<think>", "</think>", "answer"):
        n = s.count(kw)
        print("   关键词 %-18s 出现 %d 次" % (kw, n))
    # 打印含 think 的片段
    for m in re.finditer(r"think", s):
        a, b = max(0, m.start()-260), min(len(s), m.end()+260)
        print("   ...", s[a:b].replace("\n", "\\n"), "...")
        break
else:
    print("!! 没有 chat_template")
PY

echo
echo "########## 3. generation_config / config.json 摘要 ##########"
for f in generation_config.json config.json; do
  echo "--- $f"
  python3 -c "
import json,sys
try:
    d=json.load(open('$M/$f',encoding='utf-8'))
    for k in ('temperature','top_p','top_k','do_sample','max_new_tokens','eos_token_id','think','enable_thinking','chat_template'):
        if k in d: print('   ',k,'=',str(d[k])[:200])
except Exception as e:
    print('   ERR',e)
"
done

echo
echo "########## 4. vLLM 日志里的图像缩放/截断告警 ##########"
for f in "$WS"/logs/*vllm*.log "$WS"/logs/*serve*.log "$WS"/logs/*service*.log; do
  [ -f "$f" ] || continue
  echo "--- $f"
  grep -inE "resiz|truncat|reduc|too (large|many)|max_model_len|mm_processor|reduce.*image|exceed" "$f" 2>/dev/null | head -15
done
ls "$WS"/logs/ 2>/dev/null | head -40
