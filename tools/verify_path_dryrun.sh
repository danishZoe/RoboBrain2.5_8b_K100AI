#!/usr/bin/env bash
# =============================================================================
# 官方材料到位路径的「机制演练」（dry run）—— 三阶段版
#
# 目的：在官方 baselines/data/judge 真正下发之前，先证明 verify.py 的
#       「基线存在 → 解析材料 → 逐样本请求 → 判分 → 写 verify_evidence.json」
#       这条路径是通的、可复现的。
#
# 阶段：
#   A 负路径：沙箱内**不放**基线 → verify.py 必须 BLOCKED(退出码 2)，且不得碰服务。
#   B 正路径（默认，**不碰线上服务**）：沙箱内放临时基线 + 起一个 mock OpenAI
#     服务 → 整条链路必须走通并写出逐样本证据。
#     ★ 用 mock 而不是真服务，是为了不污染正在运行的官方压测。
#   C 真服务路径（仅 --live）：基线存在 + 真实调用 127.0.0.1:30000 的 vLLM 服务。
#     ★ 只在压测不在跑的时候用。
#
# 纪律（重要）：
#   1. **绝不**把演练用的假基线写进 submission/baselines/（那是官方材料目录）。
#      演练在 $W/tmp/dryrun_verify/ 独立沙箱里进行。
#   2. **绝不**覆盖 submission/verify_result.json —— 官方材料到位前它必须保持
#      status=BLOCKED。
#   3. 演练的参考输出来自我们自己的 transformers 直跑结果 / 同配置复读结果，
#      属于**内部自洽性**，不是官方精度，报告里必须如此标注。
# =============================================================================
set -o pipefail

W=/root/private_data/workspace/robo
SB="$W/tmp/dryrun_verify"
OUT="$W/logs/verify_path_dryrun.log"
MOCK_PORT="${MOCK_PORT:-39999}"
LIVE=0
[ "${1:-}" = "--live" ] && LIVE=1

mkdir -p "$SB" "$W/logs"
: > "$OUT"

log() { echo "$@" | tee -a "$OUT"; }

log "############################################################"
log "# verify.py 官方材料路径 机制演练   $(date -Is)   live=$LIVE"
log "############################################################"

# ---------------------------------------------------------------- 沙箱搭建
rm -rf "$SB/baselines" "$SB/data" "$SB/scripts" "$SB/verify.py" \
       "$SB"/verify_result*.json "$SB"/verify_evidence*.json
mkdir -p "$SB/baselines" "$SB/data"
cp "$W/submission/verify.py" "$SB/verify.py"
cp -r "$W/submission/scripts" "$SB/scripts"

# --------------------------------------------------------------- 组装子集
log "--- 组装内部子集 ---"
SB="$SB" python3 - <<'PY' 2>&1 | tee -a "$OUT"
import json, os

SB = os.environ["SB"]
W = "/root/private_data/workspace/robo"

tasks = json.load(open(f"{W}/samples/tasks_official.json", encoding="utf-8"))

# 1) 找 transformers 直跑参考输出
ref_map = {}
for p in [f"{W}/outputs/tasks_official/task_results.json",
          f"{W}/outputs/phase3_baseline/baseline_raw.json"]:
    if not os.path.exists(p):
        continue
    d = json.load(open(p, encoding="utf-8"))
    recs = d.get("results") if isinstance(d, dict) else d
    if isinstance(d, dict) and "records" in d:
        recs = d["records"]
    if not isinstance(recs, list):
        continue
    for r in recs:
        if not isinstance(r, dict):
            continue
        sid = r.get("id") or r.get("sample_id")
        ans = r.get("answer")
        if r.get("baseline") and isinstance(r["baseline"], dict):
            ans = r["baseline"].get("text")
        if sid and ans and r.get("warmup") in (None, False):
            ref_map.setdefault(sid, {"answer": ans, "source": p})

# 2) 组装子集记录
records = []
for t in tasks:
    sid = t["id"]
    img = os.path.join(W, "samples/images", t["image"])
    rec = {
        "sample_id": sid,
        "task": t["task"],
        "prompt": t["prompt"],
        "image": img,
        "max_tokens": t.get("max_new_tokens", 128),
    }
    if sid in ref_map:
        rec["reference"] = ref_map[sid]["answer"]
        rec["_ref_source"] = ref_map[sid]["source"]
    records.append(rec)

out = f"{SB}/data/internal_demo.jsonl"
with open(out, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("subset 记录数:", len(records), "->", out)
for r in records:
    print("  ", r["sample_id"], "| task=", r["task"], "| ref=", "有" if "reference" in r else "无")

# 3) 组装「假」基线文件（只在沙箱里）
baseline = {
    "model": "BAAI/RoboBrain2.5-8B-NV",
    "extra_body": {},
    "_WARNING": "DRY-RUN ONLY —— 这是机制演练用的临时基线，不是官方基线，禁止进入 submission/",
    "_metrics_source": "同卡 transformers 直跑 / 同配置复读自洽（内部指标）",
    "metrics": {
        "internal_demo_agreement": {
            "subset": "data/internal_demo.jsonl",
            "n": len(records),
            "max_tokens": 256,
            "ref": 100.0,
            "scale": 100,
            "task": "mixed",
            "judge_type": "agreement",
            "tolerance_px": 10,
            "tolerance": 10,
        }
    },
}
bp = f"{SB}/baselines/BAAI__RoboBrain2.5-8B-NV.json"
json.dump(baseline, open(bp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("假基线 ->", bp)
PY

# ------------------------------------------------------- 摘要打印（键名对齐）
summarize() {
python3 - "$1" "$2" <<'PY'
import json, os, sys
result_p, ev_p = sys.argv[1], sys.argv[2]
if not os.path.exists(result_p):
    print("!! 没有产出", result_p)
    raise SystemExit
d = json.load(open(result_p, encoding="utf-8"))
print("status  :", d.get("status"))
print("passed  :", d.get("passed"))
for k, v in (d.get("detail") or {}).items():
    if isinstance(v, dict):
        print(f"  [{k}] status={v.get('status')} got={v.get('got')} "
              f"official_ref={v.get('official_ref')} tol={v.get('tolerance_points')} "
              f"scored={v.get('scored')} errors={v.get('errors')} "
              f"without_reference={v.get('without_reference')}")
        if v.get("reason"):
            print("      reason:", v["reason"])
        if v.get("tried_paths"):
            print("      tried_paths:", v["tried_paths"])
if os.path.exists(ev_p):
    e = json.load(open(ev_p, encoding="utf-8"))
    for metric, det in (e or {}).items():
        det = det or []
        print(f"逐样本证据 {len(det)} 条  [{metric}]")
        for x in det[:12]:
            print("   ", {k: x.get(k) for k in ("index", "task", "correct", "error") if k in x})
else:
    print("逐样本证据: (无 evidence 文件)")
PY
}

# ============================================================== 阶段 A：负路径
log ""
log "=== 阶段 A：沙箱内无基线 → 必须 BLOCKED 且不碰服务 ==="
mv "$SB/baselines/BAAI__RoboBrain2.5-8B-NV.json" "$SB/_fake_baseline.json"
( cd "$SB" && python3 verify.py --out "$SB/verify_result_A.json" \
    --evidence "$SB/verify_evidence_A.json" ) 2>&1 | tee -a "$OUT"
log "阶段 A 退出码: ${PIPESTATUS[0]}"
summarize "$SB/verify_result_A.json" "$SB/verify_evidence_A.json" | tee -a "$OUT"

# ============================================================== 阶段 B：正路径
log ""
log "=== 阶段 B：临时基线 + mock 服务 → 整条链路必须走通（不碰线上服务） ==="
mv "$SB/_fake_baseline.json" "$SB/baselines/BAAI__RoboBrain2.5-8B-NV.json"
python3 "$W/tools/mock_openai_server.py" "$SB/data/internal_demo.jsonl" "$MOCK_PORT" \
  > "$W/logs/mock_server.log" 2>&1 &
MOCK_PID=$!
for i in $(seq 1 40); do
  curl -sf --noproxy "*" "http://127.0.0.1:$MOCK_PORT/health" >/dev/null 2>&1 && break
  sleep 0.25
done
if curl -sf --noproxy "*" "http://127.0.0.1:$MOCK_PORT/health" >/dev/null 2>&1; then
  log "mock 服务就绪 (pid=$MOCK_PID, port=$MOCK_PORT)"
  ( cd "$SB" && PORT="$MOCK_PORT" python3 verify.py \
      --out "$SB/verify_result.json" --evidence "$SB/verify_evidence.json" ) 2>&1 | tee -a "$OUT"
  log "阶段 B 退出码: ${PIPESTATUS[0]}"
else
  log "!! mock 服务未就绪"
fi
kill "$MOCK_PID" 2>/dev/null
wait "$MOCK_PID" 2>/dev/null
summarize "$SB/verify_result.json" "$SB/verify_evidence.json" | tee -a "$OUT"

# ============================================================== 阶段 C：真服务
if [ "$LIVE" = "1" ]; then
  log ""
  log "=== 阶段 C：临时基线 + 真实 vLLM 服务 (127.0.0.1:30000) ==="
  ( cd "$SB" && source "$W/submission/env.sh" >/dev/null 2>&1; \
    python3 verify.py --out "$SB/verify_result_C.json" \
      --evidence "$SB/verify_evidence_C.json" ) 2>&1 | tee -a "$OUT"
  log "阶段 C 退出码: ${PIPESTATUS[0]}"
  summarize "$SB/verify_result_C.json" "$SB/verify_evidence_C.json" | tee -a "$OUT"
else
  log ""
  log "=== 阶段 C：真服务路径已跳过（加 --live 开启；压测运行期间禁止开启） ==="
fi

# ------------------------------------------------------------ 纪律复核
log ""
log "--- 纪律复核：submission/ 目录未被污染 ---"
if ls "$W/submission/baselines/" | grep -q 'BAAI__RoboBrain2.5-8B-NV.json'; then
  log "!! 警告：submission/baselines/ 里出现了基线文件，需人工核查"
else
  log "OK：submission/baselines/ 仍无官方基线文件"
fi
python3 - "$W" <<'PY' 2>&1 | tee -a "$OUT"
import json, sys
p = sys.argv[1] + "/submission/verify_result.json"
d = json.load(open(p, encoding="utf-8"))
print("submission/verify_result.json status =", d.get("status"), "（应保持 BLOCKED）")
PY

log ""
log "DRY-RUN DONE  $(date -Is)"
log "沙箱产物: $SB  (verify_result*.json / verify_evidence*.json)"
log "日志    : $OUT"
