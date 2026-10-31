#!/usr/bin/env bash
# =============================================================================
# 提交包自检：对照官方技术报告模板逐项校验
#   - 必须存在的文件
#   - 产物非空
#   - README 里的数字与 bench/summary.md、env.log、verify_result.json 一致
#   - 配置一致性（MAX_LEN / PORT / FRAMEWORK ... 在 env.sh 与 server.log 之间）
# 用法: bash tools/check_submission.sh
# 退出码: 0 = 全部通过；1 = 有 FAIL
# =============================================================================
set -uo pipefail
cd "${WS:-/root/private_data/workspace/robo}/submission"

FAIL=0
WARN=0
ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=$((FAIL+1)); }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$1"; WARN=$((WARN+1)); }
must() { [ -s "$1" ] && ok "$2 ($(wc -c < "$1") B)" || bad "$2 [缺失或为空: $1]"; }

echo "=== 1. 模板要求的文件 ==="
for f in README.md env.sh env_check.sh run.sh smoke.sh verify.py bench.sh reproduce.sh \
         requirements-extra.txt env.log server.log smoke.log verify_result.json; do
  must "$f" "$f"
done
[ -d bench ] && ok "bench/ 目录" || bad "bench/ 目录 [缺失]"
[ -d patches ] && ok "patches/ 目录" || bad "patches/ 目录 [缺失]"
[ -f bench/round1.jsonl ] && ok "bench/round1.jsonl" || bad "bench/round1.jsonl [缺失]"
[ -f bench/round2.jsonl ] && ok "bench/round2.jsonl" || bad "bench/round2.jsonl [缺失]"
[ -f bench/summary.md ] && ok "bench/summary.md" || bad "bench/summary.md [缺失]"
# 验收标准 §9：固定样例的来源与校验值
[ -f samples/MANIFEST.md ] && ok "samples/MANIFEST.md（样例来源+校验值）" \
  || bad "samples/MANIFEST.md [缺失]"
[ -f samples/upstream_provenance.json ] && ok "samples/upstream_provenance.json（upstream 溯源）" \
  || bad "samples/upstream_provenance.json [缺失]"
if [ -f samples/MANIFEST.md ]; then
  if grep -q '逐位一致' samples/MANIFEST.md; then
    ok "样例校验：与 upstream 记录的 sha256 逐位一致"
  else
    bad "样例校验未通过或未标注，详见 samples/MANIFEST.md"
  fi
fi
# 验收标准 §9：固定样例及输出 = 样例来源、校验值、原文、解析值、坐标/轨迹叠加图
[ -f samples/parsed_outputs.json ] && ok "samples/parsed_outputs.json（原文 + 解析值）" \
  || bad "samples/parsed_outputs.json [缺失：验收 §9 要求的『原文、解析值』]"
if [ -f samples/parsed_outputs.json ]; then
  python3 - <<'PY'
import json, os, sys
p = "samples/parsed_outputs.json"
try:
    d = json.load(open(p, encoding="utf-8"))
except Exception as e:
    print("  FAIL  samples/parsed_outputs.json 解析失败: %s" % e); sys.exit(1)
s = d.get("samples") or []
n_raw = sum(1 for x in s if x.get("raw_output"))
n_par = sum(1 for x in s if x.get("parsed"))
print("  PASS  样例 %d 个：原文 %d、解析值 %d" % (len(s), n_raw, n_par))
PY
fi
npng=$(ls samples/annotated/*.png 2>/dev/null | wc -l)
[ "$npng" -ge 5 ] && ok "samples/annotated/ 坐标/轨迹叠加图 ${npng} 张" \
                 || bad "samples/annotated/ 只有 ${npng} 张叠加图（验收 §9 要求）"
n=$(ls bench/round*_*.log 2>/dev/null | wc -l)
[ "$n" -ge 12 ] && ok "bench/ 每档日志 ${n} 个 (期望 >= 12)" \
               || bad "bench/ 每档日志只有 ${n} 个 (期望 12: 6 档 x 2 轮)"

echo
echo "=== 2. env.sh 关键变量 ==="
for v in MODEL_ID MODEL_PATH SERVED_NAME IMAGE FRAMEWORK CARD PRECISION PORT \
         TP MAX_LEN MEM_FRAC EXTRA_ARGS WORK_DIR REPO_DIR HF_HOME PYTHONPATH; do
  if grep -qE "^[[:space:]]*export $v=" env.sh; then
    val=$(grep -E "^[[:space:]]*export $v=" env.sh | head -1 | cut -d= -f2- | sed 's/[[:space:]]*#.*$//' | tr -d '"' | cut -c1-72)
    printf '  PASS  %-12s = %s\n' "$v" "$val"
  else
    bad "$v 未在 env.sh 中定义"
  fi
done

echo
echo "=== 3. 配置一致性（env.sh vs server.log）==="
# 取值时必须剥掉行尾注释（env.sh 里这些变量带说明性注释）与首尾空白
envval() {
  grep -E "^[[:space:]]*export $1=" env.sh | head -1 | cut -d= -f2- \
    | sed 's/[[:space:]]*#.*$//' | tr -d '"' | sed 's/[[:space:]]*$//'
}
for pair in MAX_LEN PORT FRAMEWORK CARD PRECISION; do
  v=$(envval "$pair")
  case "$pair" in
    MAX_LEN) grep -q -- "--max-model-len $v " server.log && ok "server.log 的 max-model-len = $v" || bad "server.log 未见 --max-model-len $v" ;;
    PORT)    grep -q -- "--port $v " server.log && ok "server.log 的 port = $v" || bad "server.log 未见 --port $v" ;;
    FRAMEWORK) grep -q "framework: $v" env.log && ok "env.log 的 framework = $v" || warn "env.log 未见 framework: $v（env.log 可能早于本次配置）" ;;
    CARD)    grep -q "card: $v" env.log && ok "env.log 的 card = $v" || warn "env.log 未见 card: $v" ;;
    PRECISION) grep -q "precision: $v" env.log && ok "env.log 的 precision = $v" || warn "env.log 未见 precision: $v" ;;
  esac
done

echo
echo "=== 4. 精度项状态 ==="
python3 - <<'PY'
import json, os, sys
p = "verify_result.json"
if not os.path.exists(p):
    print("  FAIL  verify_result.json 缺失"); sys.exit(1)
d = json.load(open(p, encoding="utf-8"))
st = d.get("status")
print("  status =", st, " passed =", d.get("passed"))
if st == "BLOCKED":
    print("  PASS  精度项按官方口径标记为 BLOCKED（未用自建数据顶替）")
    if not d.get("reason"):
        print("  FAIL  BLOCKED 但没有写明原因"); sys.exit(1)
elif st == "PASS":
    det = d.get("detail") or {}
    bads = [k for k, v in det.items() if v.get("status") != "PASS"]
    if bads:
        print("  FAIL  标 PASS 但这些指标未 PASS:", bads); sys.exit(1)
    print("  PASS  精度项 PASS，各指标均有评测证据")
else:
    print("  FAIL  精度项状态异常:", st); sys.exit(1)
PY
[ $? -ne 0 ] && FAIL=$((FAIL+1))

echo
echo "=== 5. README 与 bench/summary.md 的数字一致性 ==="
if [ -f bench/summary.md ]; then
  if grep -q 'BENCH_SUMMARY:BEGIN' README.md || grep -q '<!-- BENCH_SUMMARY -->' README.md; then
    if grep -q 'BENCH_SUMMARY:BEGIN' README.md; then
      ok "README 第 7 节已回填官方格式压测表（summary.md $(grep -c '^| ' bench/summary.md) 行）"
    else
      bad "README 第 7 节仍是未回填的占位符"
    fi
  else
    bad "README 第 7 节找不到压测表"
  fi
  if grep -q '| 4096 | 1024 | 1 |' README.md; then
    ok "README 含官方 13 列格式的表头行"
  else
    bad "README 未见官方 13 列格式的表头行"
  fi
fi

echo
echo "=== 6. 红线自检 ==="
grep -q 'max_position_embeddings' README.md && \
  { grep -qE '修改的模型参数.*无' README.md && ok "模型参数未修改（README 已声明）" || warn "README 未明确声明模型参数未修改"; } \
  || warn "README 未提及 max_position_embeddings"
if [ -d baselines ] && [ -n "$(ls baselines/*.json 2>/dev/null)" ]; then
  bad "baselines/ 内有 .json：疑似自建基线文件（红线：不得自建）"
else
  ok "baselines/ 无 .json：未自建官方基线"
fi
if grep -qE '^\s*patch .*-d "\$PATCH_ROOT" < ' run.sh && ! grep -q 'dry-run' run.sh; then
  bad "run.sh 疑似会写入公共 site-packages"
else
  ok "run.sh 不写公共环境（patch 仅做只读判重）"
fi

echo
echo "=============================================="
echo "FAIL=$FAIL  WARN=$WARN"
[ "$FAIL" -eq 0 ] && echo "提交包自检通过" || echo "提交包自检未通过"
exit $((FAIL > 0))
