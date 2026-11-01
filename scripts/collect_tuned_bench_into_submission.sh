#!/usr/bin/env bash
# =============================================================================
# 归集「同一张卡」三套压测的原始产物到提交包，并重建 MANIFEST
#
# 目的：提交包内同时保留三套原始数据——
#   submission/bench/round{1,2}.jsonl        = 首提基线（eager，官方口径基准，10-27 节点）
#   submission/bench/tuned/                  = 最优版（graph + 调优库 + aiter FA）
#   submission/bench/same_card_eager/       = 同卡 eager 首提基线（换卡后重测）
#   submission/bench/same_card_baseline/    = 同卡 graph 基线（归因用）
# 基线材料不删不改，只新增子目录。
#
# 「同一张卡」的判定由 scripts/bench_set.py 负责（run_config.txt 里的 card_id
# 与 outputs/CARD_ID.txt 一致）；凑不齐时回退到历史同节点会话并标注。
#
# 用法：bash scripts/collect_tuned_bench_into_submission.sh [candidate_run_dir]
# =============================================================================
set -euo pipefail

WS="${WS:-/root/private_data/workspace/robo}"
cd "$WS"
# 让 scripts/bench_set.py 按当前工作目录解析（默认写死的是节点路径）
export ROBO_WS="$WS"

# --- 解析三套目录 -------------------------------------------------------------
SESSION_LABEL="未知"
CAND_DIR=""
BASE_DIR=""
EAGER_DIR=""
if command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1; then
  PY_BIN="$(command -v python3 || command -v python)"
  eval "$("$PY_BIN" - <<'PY' 2>/dev/null || true
import shlex
import sys
sys.path.insert(0, "scripts")
import bench_set

label_c, cand = bench_set.pick_single("candidate")
label_b, base, _ = bench_set.pick_pair("baseline", "candidate")
label_e, eager, _ = bench_set.pick_pair("eager", "candidate")
print("SESSION_LABEL=" + shlex.quote(str(label_c)))
print("CAND_DIR=" + shlex.quote(",".join(str(d) for d in cand)))
print("BASE_DIR=" + shlex.quote(",".join(str(d) for d in base)))
print("EAGER_DIR=" + shlex.quote(",".join(str(d) for d in eager)))
PY
)"
fi

# 纯 shell 兜底（没有 python 时按目录名里的时间戳取全部，不看 mtime）
# 注意：`set -o pipefail` 下，glob 无匹配时 `ls` 退出码为 2，会让整条管道非零，
# 再叠加 `set -e` 就会**静默退出 2**。这里统一加 `|| true` 兜住。
if [ -z "$CAND_DIR" ]; then
  CAND_DIR="$(ls -d outputs/fullbench_candidate_* 2>/dev/null | sort | tr '\n' ',' | sed 's/,$//' || true)"
  SESSION_LABEL="未知（未做同卡判定）"
fi
if [ -z "$BASE_DIR" ]; then
  BASE_DIR="$(ls -d outputs/fullbench_baseline_* 2>/dev/null | sort | tr '\n' ',' | sed 's/,$//' || true)"
fi
if [ -z "$EAGER_DIR" ]; then
  EAGER_DIR="$(ls -d outputs/fullbench_eager_* 2>/dev/null | sort | tr '\n' ',' | sed 's/,$//' || true)"
fi

if [ -n "${1:-}" ]; then
  CAND_DIR="$1"
fi

# --- 目录清单校验（逗号分隔，可能有多个服务实例目录）---------------------------
CAND_DIRS=""
IFS=',' read -ra _cd <<< "$CAND_DIR"
for d in "${_cd[@]}"; do
  [ -n "$d" ] || continue
  if [ ! -d "$d" ]; then
    echo "ERROR: 找不到最优版压测目录（$d）" >&2
    exit 1
  fi
  CAND_DIRS="${CAND_DIRS:+$CAND_DIRS,}$d"
done
CAND_DIR="$CAND_DIRS"
if [ -z "$CAND_DIR" ]; then
  echo "ERROR: 找不到最优版压测目录（CAND_DIR 为空）" >&2
  exit 1
fi

echo "[collect] 会话：$SESSION_LABEL"
echo "[collect] 最优版：$CAND_DIR"
echo "[collect] graph 基线：${BASE_DIR:-（无）}"
echo "[collect] eager 基线：${EAGER_DIR:-（无）}"

collect_one() {
  src="$1"; dest="$2"; title="$3"
  [ -n "$src" ] || return 0
  [ -d "$src" ] || return 0
  mkdir -p "$dest"
  tag="$(basename "$src")"
  # 同一配置可能被拆成多次服务实例跑，多个目录里可能有同名文件：
  # 内容一致就跳过；内容不同则用「目录名__原名」保留，绝不静默覆盖。
  for f in "$src"/round*.jsonl "$src"/round*.log "$src"/run_config.txt \
           "$src"/service_start.log "$src"/worker_env.txt \
           "$src"/summary.md "$src"/agreement.md; do
    [ -f "$f" ] || continue
    b="$(basename "$f")"
    if [ -f "$dest/$b" ] && ! cmp -s "$f" "$dest/$b"; then
      b="${tag}__${b}"
    fi
    cp -f "$f" "$dest/$b" 2>/dev/null || true
  done
  echo "[collect]   $title ← $src （$dest 现有 $(ls -1 "$dest" | wc -l) 个文件）"
}

collect_many() {
  dirs="$1"; dest="$2"; title="$3"
  # 目标目录先建出来：即使这一套没有源目录，后面写 README 也不会因目录不存在而失败
  mkdir -p "$dest"
  IFS=',' read -ra _dl <<< "$dirs"
  for d in "${_dl[@]}"; do
    [ -n "$d" ] || continue
    collect_one "$d" "$dest" "$title"
  done
}

collect_many "$CAND_DIR" "submission/bench/tuned" "最优版"
collect_many "$BASE_DIR" "submission/bench/same_card_baseline" "graph 基线"
collect_many "$EAGER_DIR" "submission/bench/same_card_eager" "首提基线 eager"

# --- 干净环境复现产物（验收标准 §7.2 / 报告 §4.5）---------------------------
# logs/ 被 .gitignore 排除，不归档就会随节点回收一起消失，故显式收进提交包。
CR_SRC="logs/clean_repro_full/clean_repro_full_20261031"
if [ -d "$CR_SRC" ]; then
  CR_DST="submission/bench/clean_repro"
  mkdir -p "$CR_DST/bench"
  for f in "$CR_SRC"/*.log "$CR_SRC"/run_config.txt "$CR_SRC"/verify_result.json; do
    [ -f "$f" ] && cp -f "$f" "$CR_DST/" 2>/dev/null || true
  done
  cp -f "$CR_SRC"/bench/*.jsonl "$CR_SRC"/bench/*.log "$CR_DST/bench/" 2>/dev/null || true
  echo "[collect] 干净复现产物 ← $CR_SRC （$CR_DST 现有 $(find "$CR_DST" -type f | wc -l) 个文件）"
else
  echo "[collect] 未找到干净复现目录 $CR_SRC（跳过，不影响其它产物）"
fi

CARD_ID_LINE=""
if [ -f outputs/CARD_ID.txt ]; then
  CARD_ID_LINE="$(grep -m1 '^card_id=' outputs/CARD_ID.txt 2>/dev/null || true)"
fi

# --- 各目录的说明 -------------------------------------------------------------
cat > submission/bench/tuned/README.md <<EOF
# 最优版压测原始数据（提交配置）

**配置**：graph(非 eager) + GEMM 算子调优库 + \`VLLM_ATTENTION_BACKEND=ROCM_AITER_FA\`
（即 \`USE_TUNED=1\` 的提交配置，见 \`../../env.sh\`）。

**会话**：$SESSION_LABEL
${CARD_ID_LINE:-（卡指纹未记录）}

**口径**：单卡 K100AI（K500SM_AI / gfx928 / 63.98 GiB）/ vLLM 0.15.1+das.opt1.alpha.dtk2604 /
BF16 / TP=1 / \`--gpu-memory-utilization 0.85 --max-model-len 32768\` /
\`--max-num-batched-tokens 4096 --max-num-seqs 64 --no-enable-prefix-caching\` /
\`--limit-mm-per-prompt '{"image":8}'\`；官方 bench_serving 客户端，image 数据集，
1×1080p 图/请求，输出 1024 token，seed 1，请求数 \`max(8, 2×并发)\`。

| 文件 | 内容 |
|---|---|
| \`round*_<输入>_1024_c<并发>.jsonl\` | 每档每轮的**客户端原始结果文件**（官方 13 列字段） |
| \`round*_<输入>_1024_c<并发>.log\` | 客户端完整日志（输入/输出 token 拆分、TTFT/TPOT/ITL 分布） |
| \`recovered_round*.jsonl\` | 节点不可达期间由**会话日志转录**恢复的结果（内容为压测客户端/汇总脚本的原样输出，**不是**客户端原始文件）；被新的客户端原始文件取代后保留作过程存档 |
| \`run_config.txt\` | 本次压测的配置指纹（卡指纹 \`card_id=\`、档位、模型、vllm 版本、stock/候选库哈希、git revision） |
| \`service_start.log\` | 服务启动输出（就绪耗时） |
| \`worker_env.txt\` | 推理 worker 进程的实际环境（调优开关是否真正进入进程） |
| \`summary.md\` | 汇总表（13 列 + 轮次波动） |
| \`agreement.md\` | 同口径探针与 transformers 基线的逐坐标比对 |

**首提基线（eager）的原始数据**：上一级目录 \`bench/round1.jsonl\` / \`bench/round2.jsonl\`
（10-27 节点的官方口径基准）；**同一张卡**上的 eager 与 graph 基线见
\`bench/same_card_eager/\` 与 \`bench/same_card_baseline/\`。

> \`recovered_round*.jsonl\` 是 2026-11-01 05:08 节点不可达期间，由压测客户端与汇总脚本的
> **原样输出转录**恢复的实测值（字段与官方口径一致，**不是**客户端原始文件），
> 文件名前缀用于与原始文件区分。被新的客户端原始文件取代后保留作过程存档。

> ⚠️ **跨节点的绝对值不可直接比较**。容器被平台回收重建后物理卡会更换，
> 收益结论一律用**同一张卡**上三套配置的 A/B（判定见 \`../../scripts/bench_set.py\`）。

> 同一配置若被拆到**多个服务实例**跑（例如 16384/并发64 这类单轮约 3.8 h 的重档
> 单独起一次服务），本目录会同时保留各次的 \`round*.jsonl\`；
> 同名但内容不同的文件以「运行目录名__原名」保存，**绝不静默覆盖**。
EOF

cat > submission/bench/same_card_eager/README.md <<EOF
# 同卡 · 首提基线 eager（赛事「首次提交」口径）

**配置**：\`--enforce-eager\` + stock rocBLAS 库 + 默认 attention。
**会话**：$SESSION_LABEL
${CARD_ID_LINE:-（卡指纹未记录）}

与 \`../tuned/\`（最优版）**必须来自同一张卡**才能相减；
逐档对比见 \`../../reports/tuning_comparison.md\` §2.1
与 \`../../reports/25_tuning_delivery_20261031.md\` §5.3。
EOF

cat > submission/bench/same_card_baseline/README.md <<EOF
# 同卡 · graph 基线（只开 graph，用于收益归因）

**配置**：非 eager + stock rocBLAS 库 + 默认 attention（不带任何调优开关）。
**会话**：$SESSION_LABEL
${CARD_ID_LINE:-（卡指纹未记录）}

与 \`../tuned/\`（最优版）**必须来自同一张卡**才能相减；
逐档对比见 \`../../reports/tuning_comparison.md\` §2.2
与 \`../../reports/25_tuning_delivery_20261031.md\` §5.2。
EOF

# --- 重建 MANIFEST.sha256（排除 __pycache__、MANIFEST 自身、pyc）---
cd submission || exit 1
find . -type f ! -path '*/__pycache__/*' ! -name 'MANIFEST.sha256' ! -name '*.pyc' \
  | sort | xargs sha256sum > MANIFEST.sha256
echo "[collect] MANIFEST 条目数：$(wc -l < MANIFEST.sha256)"
