#!/usr/bin/env bash
# =============================================================================
# record_revision.sh - 固化交付代码的 revision 与逐文件校验值
#
# 背景：验收标准 §8.1 要求「保留该版本的代码 revision」，§9 要求提交材料可核对。
#       本脚本在远程工作目录内建一个**只含交付物**的 git 仓库（不 push、不联网），
#       产出：
#         1) .revision            —— 交付代码 revision（commit hash + 时间 + 文件数）
#         2) submission/MANIFEST.sha256 —— 提交包每个文件的 sha256
#       两者都可被第三方独立复核，替代「口头声称没改过」。
#
# 两阶段（避免「revision 笔记本身又改变 revision」的自指问题）：
#   阶段 A（默认）  : git init -> 提交交付物 = **交付代码 revision** -> 生成校验清单
#   阶段 B(--finalize): 在 inject_revision.py 把 revision 写进报告/README 之后，
#                      再提交一次（只含 revision 笔记），记录为 `revision_note_commit`。
#
# 只纳入交付物目录；envs/pylibs（隔离依赖）、cache/、tmp/、outputs/、logs/ 不进仓库。
#
# 用法: bash tools/record_revision.sh [--finalize]
# =============================================================================
set -uo pipefail
W="${WS:-/root/private_data/workspace/robo}"
cd "$W" || exit 1

GITDIR="$W/.git"
DELIVERABLES="submission scripts tools configs samples patches reports platform"
FINALIZE=0
[ "${1:-}" = "--finalize" ] && FINALIZE=1

# .gitignore 的**唯一真源**在本脚本里：每次运行都写入完整版本。
# 早期版本只写最小集（envs/pylibs、cache、tmp、outputs、logs），
# 会把仓库里已有的「大体积 benchmark 镜像 / 密钥」排除规则整段覆盖掉，
# 导致 third_party/ 下的镜像库和 .env/*.token 有被误提交的风险。
# 因此这里写**全集**，任何一方都不得单独裁剪。
cat > "$W/.gitignore" <<'EOF'
# 隔离依赖与运行产物：不进交付仓库（体积大、可由脚本重新生成）
envs/
envs/*
cache/
tmp/
outputs/
logs/
*.pyc
__pycache__/

# 大体积 benchmark / 数据镜像：只保留清单与代码，数据另行拉取
third_party/benchmarks/
third_party/upstream_eval/
third_party/Robo-Dopamine/
third_party/RoboBrain2.5-upstream-HEAD/
third_party/RoboBrain2.5/

# 生成的快照与压缩包
*.tar.gz
*.tgz
*.bak_*
*.bak-*
submission/__pycache__/
submission/logs/

# 密钥与平台本地状态
.env
*.token
*.key

# 运行时组装的 rocBLAS 调优库（仓库内只提交 6 个差异文件，运行时装配目录不进仓库）
/rocblas_tuned/

# 进程崩溃转储（节点上出现过 3 GB 级 core 文件，绝不能进仓库）
core
core.*

# 厂商 GEMM 调优工具的工作目录（4 个实例合计 8 GB+，含 Tensile 二进制与旧库副本）。
# 可复核的关键证据（选优日志、shape 采集、算子级对比 CSV）已固化到
# reports/evidence/gemm_tuning_20261030/；
# 生效的候选库差异文件在 submission/rocblas_tuned/library_gpu4/。
tools/auto_select_tools*/
EOF

if [ ! -d "$GITDIR" ]; then
  git init -q "$W" || { echo "[rev] git init failed"; exit 1; }
  git -C "$W" config user.email "robo-adaptation@local"
  git -C "$W" config user.name "RoboBrain2.5-8B-NV adaptation"
fi

commit_all() {
  local msg="$1"
  git -C "$W" add -A -- $DELIVERABLES 2>/dev/null
  git -C "$W" add -f .gitignore 2>/dev/null
  if git -C "$W" diff --cached --quiet; then
    echo "[rev] nothing new to commit"
    return 0
  fi
  git -C "$W" commit -q -m "$msg" || { echo "[rev] commit failed"; return 1; }
  return 0
}

if [ "$FINALIZE" = "0" ]; then
  commit_all "${REV_MSG:-RoboBrain2.5-8B-NV on K100AI (K500SM_AI): vLLM 0.15.1 BF16 submission package}" || exit 1

  HASH=$(git -C "$W" rev-parse HEAD)
  SHORT=$(git -C "$W" rev-parse --short HEAD)
  NFILES=$(git -C "$W" ls-tree -r --name-only HEAD | wc -l)
  NOW=$(date -Is)

  MAN="$W/submission/MANIFEST.sha256"
  # 清单只覆盖**随仓库交付的文件**：__pycache__/*.pyc 与 logs/ 都被 .gitignore
  # 排除、不进仓库，若写进清单，评审 `sha256sum -c MANIFEST.sha256` 会报
  # 一堆 "No such file"。过滤规则必须与 .gitignore 的排除项保持一致。
  ( cd "$W/submission" && find . -type f ! -name 'MANIFEST.sha256' \
      ! -name '*.pyc' ! -path '*/__pycache__/*' ! -path './logs/*' \
      -print0 | sort -z | xargs -0 sha256sum ) > "$MAN"

  cat > "$W/.revision" <<EOF
# RoboBrain2.5-8B-NV / K100AI (K500SM_AI) 交付 revision
revision      = ${HASH}
revision_short= ${SHORT}
recorded_at   = ${NOW}
tracked_files = ${NFILES}
manifest      = submission/MANIFEST.sha256
repo_layout   = ${DELIVERABLES}
note          = 本仓库只含交付物；envs/pylibs、cache、tmp、outputs、logs 未纳入
                 （隔离依赖与运行产物，可由提交包脚本重新生成）。
                 托管平台链接与 revision 仍需由用户上传后回填
                 （PDF 要求代码与镜像分别提交到指定托管平台）。
EOF
  echo "[rev] revision      = ${HASH}"
  echo "[rev] tracked_files = ${NFILES}"
  echo "[rev] manifest      = submission/MANIFEST.sha256 ($(wc -l < "$MAN") files)"
  echo "[rev] .revision written (阶段 A 完成；跑 inject_revision.py 后再执行 --finalize)"
else
  # 阶段 B：revision 笔记已写进报告与 README，提交之
  #
  # 注意：阶段 A 生成的 MANIFEST.sha256 此时**已经过期** —— inject_revision.py
  # 在阶段 A 之后改写了 submission/README.md（把 revision 写进正文），
  # 逐文件哈希随之变化。所以阶段 B 必须**先重算 MANIFEST 再提交**，
  # 否则提交出去的清单与它所列的文件对不上（评审按 sha256 复核会失败）。
  MAN="$W/submission/MANIFEST.sha256"
  # 清单只覆盖**随仓库交付的文件**：__pycache__/*.pyc 与 logs/ 都被 .gitignore
  # 排除、不进仓库，若写进清单，评审 `sha256sum -c MANIFEST.sha256` 会报
  # 一堆 "No such file"。过滤规则必须与 .gitignore 的排除项保持一致。
  ( cd "$W/submission" && find . -type f ! -name 'MANIFEST.sha256' \
      ! -name '*.pyc' ! -path '*/__pycache__/*' ! -path './logs/*' \
      -print0 | sort -z | xargs -0 sha256sum ) > "$MAN"
  echo "[rev] manifest refreshed ($(wc -l < "$MAN") files)"

  commit_all "docs: record delivery revision $(git -C "$W" rev-parse --short HEAD 2>/dev/null || echo '')" || exit 1
  NOTE=$(git -C "$W" rev-parse HEAD)
  SHORTNOTE=$(git -C "$W" rev-parse --short HEAD)
  # 把笔记 commit 追加进 .revision
  if [ -f "$W/.revision" ]; then
    printf 'revision_note_commit= %s\nrevision_note_short = %s\n' "$NOTE" "$SHORTNOTE" >> "$W/.revision"
    # .revision 不在 DELIVERABLES 列表里，必须显式提交，
    # 否则 GitHub 上的 .revision 会一直停留在上一轮的旧值，
    # 与 README 里写的 revision 对不上。
    git -C "$W" add -f .revision
    git -C "$W" commit -q -m "docs: persist revision note (revision_note_commit=${SHORTNOTE})" || true
  fi
  echo "[rev] revision_note_commit = ${NOTE}"
  git -C "$W" log --oneline -3 | sed 's/^/[rev] /'
fi
