#!/usr/bin/env bash
# =============================================================================
# 提交包收尾：把压测结果回填进报告 → 归集原始数据 → 重建对比表 → 修正归因
#            → 更新 revision → 提交包自检 → 导出可发布 tarball
#
# 用法：bash scripts/finalize_submission.sh
# 退出码：0 = 全流程通过（自检 FAIL=0 且 tarball 已导出）
#
# 说明：
#   * 全部脚本均为幂等，可重复执行；
#   * 任何一步失败立即中止，不产出半成品 tarball；
#   * 不修改公共 /opt/dtk，不删除首提基线原始数据。
# =============================================================================
set -uo pipefail

WS="${WS:-/root/private_data/workspace/robo}"
cd "$WS" || exit 1

CAND_DIR="${CAND_DIR:-outputs/fullbench_candidate_20261031_231152}"
OUT_TARBALL="${OUT_TARBALL:-tmp/publish_final.tar.gz}"

step() { echo; echo "=================== $* ==================="; }

step "0/8 环境"
echo "ws=$WS"
echo "node_time=$(date -Is)"
ls -d "$CAND_DIR" 2>/dev/null || { echo "ERROR: 候选压测目录不存在：$CAND_DIR" >&2; exit 1; }

step "1/8 回填压测表（reports/25 §4/§5.2、submission/README §7）"
python3 scripts/fill_tuning_tables.py --run-dir "$CAND_DIR" || exit 1

step "2/8 限制表补充实测口径（reports/25 §9）"
python3 scripts/patch_report25_limits.py || exit 1

step "3/8 按原始数据修正结论数值与归因（reports/25 §1/§2、reports/05_final §4.1、README §3.5、env.sh 注释）"
python3 scripts/patch_report25_conclusion.py || exit 1

step "4/8 归集最优版原始数据到 submission/bench/tuned/ 并重建 MANIFEST"
bash scripts/collect_tuned_bench_into_submission.sh "$CAND_DIR" || exit 1

step "5/8 重建调优对比表（reports/tuning_comparison.md）"
python3 scripts/build_tuning_comparison.py --out reports/tuning_comparison.md >/dev/null || exit 1
echo "OK: reports/tuning_comparison.md"

step "6/8 清理一次性打补丁脚本（移到 gitignore 的 tmp/，可恢复）"
mkdir -p tmp/discarded_patchers
for f in submission/scripts/patch_run_sh.py tools/patch_check_submission.py \
         tools/patch_gitignore.py tools/patch_gitignore2.py; do
  if [ -f "$f" ]; then
    mv -f "$f" tmp/discarded_patchers/ && echo "  moved: $f"
  fi
done

step "7/8 更新 revision 并做提交包自检"
bash tools/record_revision.sh || exit 1
# MANIFEST 必须最后重建（覆盖上面所有对 submission/ 的改动）
( cd submission && find . -type f ! -path '*/__pycache__/*' ! -name 'MANIFEST.sha256' \
    ! -name '*.pyc' | sort | xargs sha256sum > MANIFEST.sha256 )
echo "MANIFEST 条目数：$(wc -l < submission/MANIFEST.sha256)"
bash tools/check_submission.sh
RC=$?
if [ "$RC" -ne 0 ]; then
  echo "ERROR: 提交包自检未通过（FAIL>0），中止导出" >&2
  exit 1
fi

step "8/8 导出可发布 tarball"
bash scripts/export_publish_tarball.sh "$OUT_TARBALL" || exit 1

echo
echo "=================== 完成 ==================="
echo "tarball : $WS/$OUT_TARBALL"
sha256sum "$OUT_TARBALL"
