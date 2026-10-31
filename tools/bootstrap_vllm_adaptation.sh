#!/usr/bin/env bash
# =============================================================================
# bootstrap_vllm_adaptation.sh
#   在**任意新实例**（尤其是换了官方 sglang 镜像的实例）里，一键重放本任务的
#   vLLM 适配成果，并把环境差异显式打印出来。
#
#   设计原则（与本任务红线一致）：
#     * 不写公共 site-packages：垫片走 sitecustomize + PYTHONPATH 注入；
#       新增依赖装进 $WORK_DIR/envs/pylibs（--no-deps --target，隔离前缀）。
#     * 不猜版本：先探测 torch / vllm / transformers / DTK，再决定垫片是否需要。
#     * 不改模型权重、不改驱动、不动公共环境。
#     * 只读模型目录：只 stat 权重分片，不写。
#
#   用法：
#     bash /root/private_data/workspace/robo/tools/bootstrap_vllm_adaptation.sh          # 只体检
#     bash .../bootstrap_vllm_adaptation.sh --install                                 # 体检 + 补齐依赖
#     bash .../bootstrap_vllm_adaptation.sh --install --start                          # 再拉起服务
#     bash .../bootstrap_vllm_adaptation.sh --probe                                   # 起服务并跑 6 样例探针
#
#   退出码： 0 成功 / 1 环境不满足（缺模型或缺 python 依赖）/ 2 服务未就绪
# =============================================================================
set -o pipefail

WORK_DIR="${WORK_DIR:-/root/private_data/workspace/robo}"
REPO_DIR="$WORK_DIR/submission"
MODEL_PATH="${MODEL_PATH:-/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV}"
DO_INSTALL=0
DO_START=0
DO_PROBE=0

for a in "$@"; do
  case "$a" in
    --install) DO_INSTALL=1 ;;
    --start)   DO_START=1 ;;
    --probe)   DO_PROBE=1; DO_START=1 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown arg: $a" >&2; exit 1 ;;
  esac
done

say()  { printf '\n=== %s ===\n' "$*"; }
info() { printf '  %s\n' "$*"; }
fail() { printf '\n[FATAL] %s\n' "$*" >&2; exit 1; }

# --- 0) DTK 运行时（/opt/dtk/env.sh 引用了未定义变量，不能在 set -u 下 source）------
if [ -f /opt/dtk/env.sh ]; then
  _had_u=0; case "$-" in *u*) _had_u=1 ;; esac
  set +u
  # shellcheck disable=SC1091
  source /opt/dtk/env.sh
  [ "$_had_u" = "1" ] && set -u
  unset _had_u
fi

# --- 1) 工作目录完整性 ---------------------------------------------------------
say "1) 工作目录"
if [ ! -d "$WORK_DIR" ]; then
  fail "工作目录不存在：$WORK_DIR（换实例后请确认 /root/private_data 持久化挂载仍在）"
fi
info "WORK_DIR = $WORK_DIR"
for p in "$REPO_DIR/run.sh" "$REPO_DIR/env.sh" "$REPO_DIR/scripts/serve_vllm_patched.py" \
         "$WORK_DIR/envs/vllm_shim/sitecustomize.py" \
         "$WORK_DIR/patches/0001-vllm-qwen3-tie-word-embeddings.patch"; do
  if [ -e "$p" ]; then info "OK   $p"; else info "MISS $p"; fi
done
if [ ! -e "$WORK_DIR/envs/vllm_shim/sitecustomize.py" ]; then
  fail "垫片缺失：$WORK_DIR/envs/vllm_shim/sitecustomize.py（请从本地 agent_work/adaptation/envs/ 同步）"
fi

# --- 2) 环境指纹 ---------------------------------------------------------------
say "2) 环境指纹"
python3 - <<'PY' 2>&1 | sed 's/^/  /'
import sys
print("python      :", sys.version.split()[0], "|", sys.executable)
for mod in ("torch", "transformers", "vllm", "sglang", "aiter"):
    try:
        m = __import__(mod)
        v = getattr(m, "__version__", "?")
        extra = ""
        if mod == "torch":
            extra = f" | hip {getattr(__import__('torch').version,'hip','?')}" \
                    f" | cuda_avail {__import__('torch').cuda.is_available()}"
        print(f"{mod:<12}: {v}{extra} | {getattr(m,'__file__','?')}")
    except Exception as e:
        print(f"{mod:<12}: <不可用> {type(e).__name__}: {str(e)[:120]}")
PY
info "DTK_VERSION = ${DTK_VERSION:-<unset>}  (/opt/dtk/dcc: $(cat /opt/dtk/.info 2>/dev/null | tr '\n' ' ' | cut -c1-60))"
info "cgroup      = $(head -1 /proc/1/cgroup | cut -c1-90)"

# --- 3) 模型可读性（只 stat，不写） ---------------------------------------------
say "3) 模型目录（只读校验）"
if [ ! -d "$MODEL_PATH" ]; then
  fail "模型目录不可访问：$MODEL_PATH"
fi
n_sh=$(ls -1 "$MODEL_PATH"/*.safetensors 2>/dev/null | wc -l)
n_idx=$(ls -1 "$MODEL_PATH"/*.safetensors.index.json 2>/dev/null | wc -l)
info "safetensors 分片: $n_sh 个 ; index: $n_idx 个"
info "config.json: $([ -e "$MODEL_PATH/config.json" ] && echo 存在 || echo 缺失)"
# lm_head 是否真的不在权重里（垫片的判定依据）
python3 - "$MODEL_PATH" <<'PY' 2>&1 | sed 's/^/  /'
import json, os, sys
mp = sys.argv[1]
try:
    cfg = json.load(open(os.path.join(mp, "config.json")))
    print("config.tie_word_embeddings =", cfg.get("tie_word_embeddings"))
    print("config.architectures      =", cfg.get("architectures"))
    idx = os.path.join(mp, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx)).get("weight_map", {})
        print("index 中 lm_head 张量       =", any("lm_head" in n for n in wm))
        print("权重总字节                  =", sum(v for v in wm.values() if False) or len(wm), "(条目数)")
except Exception as e:
    print("读取失败:", e)
PY

# --- 4) 垫片是否需要（版本自适应判定） ------------------------------------------
say "4) tie_word_embeddings 垫片判定"
NEED=$(python3 - <<'PY' 2>/dev/null || echo unknown
try:
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig as C
    print("needed" if not hasattr(C, "tie_word_embeddings") else "not-needed")
except Exception as e:
    print("unknown")
PY
)
case "$NEED" in
  needed)     info "当前栈**存在**该错位 → 垫片必须启用（ROBO_VLLM_TIE_SHIM=1 + PYTHONPATH 注入）" ;;
  not-needed) info "当前栈 transformers 已自带该属性 → 垫片自动成为空操作（仍保持启用，无副作用）" ;;
  *)          info "无法判定（transformers 结构可能变了）→ 保持垫片启用，启动后看 server.log" ;;
esac

# 垫片自检必须用**全新解释器**（sitecustomize 在解释器启动时已被 site 自动 import）
PYTHONPATH="$WORK_DIR/envs/vllm_shim:${PYTHONPATH:-}" \
ROBO_VLLM_TIE_SHIM=1 ROBO_MODEL_PATH="$MODEL_PATH" \
python3 -c "
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig as C
print('  shim check: Qwen3VLTextConfig.tie_word_embeddings =', getattr(C,'tie_word_embeddings','<缺失>'))
" 2>&1 | sed 's/^/  /'

# --- 5) 隔离依赖（--no-deps --target，不污染公共 site-packages） -----------------
if [ "$DO_INSTALL" = "1" ]; then
  say "5) 补齐隔离依赖 → $WORK_DIR/envs/pylibs"
  mkdir -p "$WORK_DIR/envs/pylibs"
  if python3 -c "import qwen_vl_utils" 2>/dev/null; then
    info "qwen_vl_utils 已可导入，跳过"
  else
    if [ -e "$REPO_DIR/requirements-extra.txt" ]; then
      PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
      pip install -q --no-deps --target "$WORK_DIR/envs/pylibs" -r "$REPO_DIR/requirements-extra.txt" \
        && info "安装完成" || info "安装失败（离线或索引不可达），qwen_vl_utils 可能已由镜像自带"
    else
      info "requirements-extra.txt 不存在，跳过"
    fi
  fi
else
  say "5) 隔离依赖（未安装，加 --install 执行）"
  python3 -c "import qwen_vl_utils; print('  qwen_vl_utils 已可用:', qwen_vl_utils.__file__)" 2>/dev/null \
    || info "  qwen_vl_utils 不可用 → 需要 --install"
fi

# --- 6) 启动服务 ---------------------------------------------------------------
if [ "$DO_START" = "1" ]; then
  say "6) 拉起 vLLM 服务"
  cd "$REPO_DIR"
  # env.sh 里 IMAGE 字段记录的是当前实例所用镜像；换镜像后请同步更新，保证报告与 env.log 一致
  bash ./run.sh
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "  服务未就绪，server.log 尾部：" >&2
    tail -n 40 "$REPO_DIR/server.log" >&2 2>/dev/null
    exit 2
  fi
  info "服务已就绪：http://127.0.0.1:${PORT:-30000}"
  if [ "$DO_PROBE" = "1" ]; then
    say "7) 6 样例探针（同卡逐坐标比对基线）"
    python3 "$REPO_DIR/scripts/robo_infer.py" --task all 2>&1 | tail -n 60 || true
  fi
else
  say "6) 服务（未启动，加 --start 执行）"
  info "启动命令： cd $REPO_DIR && bash ./run.sh"
  info "注意：run.sh 读取 env.sh 的 EXTRA_ARGS，其中 --enforce-eager 是**必须**的"
  info "      （非 eager 启动在本节点会撞 115 GiB 容器上限被 OOM killer 杀掉）"
fi

say "完成"
info "下一步：① 记录新 env.log（python/torch/vllm/transformers 版本）"
info "        ② 若 vllm 版本变了，重跑 submission/env_check.sh 并同步更新报告中的版本号"
info "        ③ 官方精度材料到位后：python submission/verify.py"
