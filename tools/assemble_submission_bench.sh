#!/usr/bin/env bash
# 把公开基准的**结果**（不含大体积逐题预测）装配进提交包，使其自包含。
# 只复制文本产物；逐题预测仍留在远程工作目录（体积大）。
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
DST="$WS/submission/bench_public/results"
mkdir -p "$DST"

copy() {  # copy <src> <dst-name>
  [ -f "$1" ] && cp -f "$1" "$DST/$2" && echo "  + $2" || echo "  - $2 (缺 $1)"
}

echo "===== 公开基准结果 -> submission/bench_public/results ====="
copy outputs/bench_public/summary_table.md          summary_table.md
copy outputs/bench_public/accuracy_regression.md    accuracy_regression.md
copy outputs/bench_public/accuracy_regression.json  accuracy_regression.json
copy outputs/bench_public/rescore_check.json        rescore_check.json
copy outputs/bench_public/manifest/data_manifest.md  data_manifest.md
copy outputs/bench_public/manifest/data_manifest.json data_manifest.json
copy outputs/bench_public/refspatial/summary.json   refspatial_summary.json
copy outputs/bench_public/cvbench/summary.json       cvbench_summary.json
copy outputs/bench_public/tracespatial/summary.json tracespatial_summary.json
copy outputs/bench_public/cvbench/task_breakdown.json cvbench_task_breakdown.json
copy outputs/bench_public/stability/stability.json  stability.json
copy outputs/bench_public/cvbench_protocol/protocol_compare.json cvbench_protocol_compare.json
copy outputs/bench_public_noneager/refspatial/summary.json   candidate_refspatial_summary.json
copy outputs/bench_public_noneager/cvbench/summary.json       candidate_cvbench_summary.json
copy outputs/bench_public_noneager/tracespatial/summary.json candidate_tracespatial_summary.json
copy outputs/bench_public_noneager/rescore_check.json        candidate_rescore_check.json
copy outputs/bench_public_noisecheck/noise_floor.json         refspatial_noise_floor.json

mkdir -p "$DST/logs"
for f in public_bench_baseline-eager/refspatial.log \
         public_bench_baseline-eager/cvbench.log \
         public_bench_baseline-eager/tracespatial.log \
         public_bench_baseline-eager/stability.log \
         public_bench_baseline-eager/rescore.log \
         public_bench_baseline-eager/cvbench_protocol.log \
         public_bench_noneager-candidate/refspatial.log \
         public_bench_noneager-candidate/cvbench.log \
         public_bench_noneager-candidate/tracespatial.log \
         public_bench_noneager-candidate/rescore.log ; do
  [ -f "logs/$f" ] && cp -f "logs/$f" "$DST/logs/$(echo "$f" | tr '/' '_')" && echo "  + logs/$(echo "$f" | tr '/' '_')"
done

cat > "$DST/README.md" <<'EOF'
# 公开基准评测结果（提交包内副本）

> ⚠️ **这三个评测集都不是赛事官方验收数据集。**
> 本目录只是**公开上游基准**上的测试有效性与任务质量证据，
> **不能替代赛事官方「与 GPU 参考输出一致率 ≥98%」门槛**。
> 赛事官方精度项见 `../verify_result.json`（当前 `BLOCKED`）。

| 文件 | 内容 |
|---|---|
| `summary_table.md` | 三基准汇总表（计划题数 / 有效返回 / 实际计分 / 失败数 / 参考值 / 本次值 / 差值 / 判定） |
| `refspatial_summary.json` | RefSpatial 4 变体 × 3 split 的官方 CLI 判分 + norm1k 交叉核对 |
| `cvbench_summary.json` | CV-Bench 官方 FlagEvalMM `get_result` 分组加权结果 |
| `cvbench_task_breakdown.json` | CV-Bench 按 task / 选项数 / source 的失分层级分解 |
| `tracespatial_summary.json` | TraceSpatial 官方 2D / 3D 几何指标 |
| `stability.json` | 每基准 12 样例 × 2 次重复推理的一致性检查（含逐题原文） |
| `rescore_check.json` | **离线重新判分**的逐指标复现比对 |
| `accuracy_regression.md` / `.json` | 参考（eager 基线）vs 候选（E1 非 eager）的逐指标差值与逐题变化率 |
| `candidate_*_summary.json` | 候选配置（非 eager）的对应结果 |
| `candidate_rescore_check.json` | 候选侧的离线重判分比对 |
| `cvbench_protocol_compare.json` | **CV-Bench 图像输入协议因子实验**（缩放 × 编码，8 组，Count 788 题） |
| `refspatial_noise_floor.json` | RefSpatial **同配置重跑**的噪声底（判定 1 pt 级差值能否归因于配置） |
| `data_manifest.md` / `.json` | 数据完整性清单（逐文件 SHA256 / 条目核对 / 全量解码检查） |
| `logs/` | 各阶段原始日志 |

逐题原始预测未随包提交（体积大），留在远程工作目录
`outputs/bench_public/` 与 `outputs/bench_public_noneager/`，
可用 `submission/bench_public/rescore.py --bench-root <dir>` 离线重判分复现。
EOF

echo
echo "===== 清单 ====="
ls -la "$DST" | sed 's/^/  /'
du -sh "$DST"
