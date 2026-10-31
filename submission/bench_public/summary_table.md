# 公开基准评测汇总表

> **这三个评测集都不是赛事官方验收数据集。** 本表只证明公开基准上的测试有效性与任务质量，**不能替代赛事官方 98% 一致率门槛**。

「允许下降」一栏全部为「未定义」：本次没有预先登记公开基准的质量回归阈值，按《精度测试验收要求》§7.3，精度回归结论为 `BLOCKED（阈值未定义）`，不得自拟通过线。

| 基准/指标 | 计划题数 | 有效返回数 | 实际计分数 | 请求/解析失败数 | 参考值 | 本次值 | 差值/单位 | 允许下降 | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| RefSpatial / Location / upstream_scaled（默认范围） | 100 | 100 | 100 | 0 | 60.50%（论文，聚合口径未公开） | 68.00% | 68.00%（官方 CLI） | 未定义 | PASS（公开基准） |
| RefSpatial / Placement / upstream_scaled（默认范围） | 100 | 100 | 100 | 0 | 60.50%（论文，聚合口径未公开） | 58.00% | 58.00%（官方 CLI） | 未定义 | PASS（公开基准） |
| RefSpatial / Unseen / upstream_scaled（重叠子集（单列，不计入默认总分）） | 77 | 77 | 77 | 0 | 60.50%（论文，聚合口径未公开） | 48.05% | 48.05%（官方 CLI） | 未定义 | PASS（公开基准） |
| RefSpatial / Location / upstream_raw（默认范围） | 100 | 100 | 100 | 0 | 60.50%（论文，聚合口径未公开） | 0.00% | 0.00%（官方 CLI） | 未定义 | PASS（公开基准） |
| RefSpatial / Placement / upstream_raw（默认范围） | 100 | 100 | 100 | 0 | 60.50%（论文，聚合口径未公开） | 1.00% | 1.00%（官方 CLI） | 未定义 | PASS（公开基准） |
| RefSpatial / Unseen / upstream_raw（重叠子集（单列，不计入默认总分）） | 77 | 77 | 77 | 0 | 60.50%（论文，聚合口径未公开） | 1.30% | 1.30%（官方 CLI） | 未定义 | PASS（公开基准） |
| RefSpatial / Location / official_scaled（默认范围） | 100 | 100 | 100 | 0 | 60.50%（论文，聚合口径未公开） | 66.00% | 66.00%（官方 CLI） | 未定义 | PASS（公开基准） |
| RefSpatial / Placement / official_scaled（默认范围） | 100 | 100 | 100 | 0 | 60.50%（论文，聚合口径未公开） | 58.00% | 58.00%（官方 CLI） | 未定义 | PASS（公开基准） |
| RefSpatial / Unseen / official_scaled（重叠子集（单列，不计入默认总分）） | 77 | 77 | 77 | 0 | 60.50%（论文，聚合口径未公开） | 45.45% | 45.45%（官方 CLI） | 未定义 | PASS（公开基准） |
| RefSpatial / Location / official_raw（默认范围） | 100 | 100 | 100 | 0 | 60.50%（论文，聚合口径未公开） | 0.00% | 0.00%（官方 CLI） | 未定义 | PASS（公开基准） |
| RefSpatial / Placement / official_raw（默认范围） | 100 | 100 | 100 | 0 | 60.50%（论文，聚合口径未公开） | 1.00% | 1.00%（官方 CLI） | 未定义 | PASS（公开基准） |
| RefSpatial / Unseen / official_raw（重叠子集（单列，不计入默认总分）） | 77 | 77 | 77 | 0 | 60.50%（论文，聚合口径未公开） | 1.30% | 1.30%（官方 CLI） | 未定义 | PASS（公开基准） |
| TraceSpatial / 2D / official / 3D Start | 100 | 100 | 100 | 0 | 83%（论文） | 0.00% | 0.00%；同集官方参照 RoboTracer_Intrinsics_Depth=75% | 未定义 | PASS（公开基准） |
| TraceSpatial / 2D / official / 3D End | 100 | 100 | 100 | 0 | 63%（论文） | 0.00% | 0.00%；同集官方参照 RoboTracer_Intrinsics_Depth=57% | 未定义 | PASS（公开基准） |
| TraceSpatial / 2D / official / Success | 100 | 100 | 100 | 0 | 44%（论文） | 0.00% | 0.00%；同集官方参照 RoboTracer_Intrinsics_Depth=52% | 未定义 | PASS（公开基准） |
| TraceSpatial / 2D / official / 2D Start-in-Mask | 100 | 100 | 100 | 0 | — | 73.00% | 73.00%；同集官方参照 RoboTracer_Intrinsics_Depth=58% | 未定义 | PASS（公开基准） |
| TraceSpatial / 2D / official / 2D End-in-BBox | 100 | 100 | 100 | 0 | — | 52.00% | 52.00%；同集官方参照 RoboTracer_Intrinsics_Depth=43% | 未定义 | PASS（公开基准） |
| TraceSpatial / 2D / official / 2D DFD | 100 | 100 | 100 | 0 | 越低越好；RoboTracer_Intrinsics_Depth 见报告 | 0.2055 | 0.2055 | 未定义 | PASS（公开基准） |
| TraceSpatial / 2D / official / 3D DFD | 100 | 100 | 100 | 0 | 越低越好；RoboTracer_Intrinsics_Depth 见报告 | 0.0000 | 0.0000 | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / official / 3D Start | 100 | 100 | 96 | 0 | 83%（论文） | 78.13% | 78.13%；同集官方参照 RoboTracer_Intrinsics_Depth=75% | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / official / 3D End | 100 | 100 | 96 | 0 | 63%（论文） | 63.54% | 63.54%；同集官方参照 RoboTracer_Intrinsics_Depth=57% | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / official / Success | 100 | 100 | 96 | 0 | 44%（论文） | 46.88% | 46.88%；同集官方参照 RoboTracer_Intrinsics_Depth=52% | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / official / 2D Start-in-Mask | 100 | 100 | 96 | 0 | — | 72.92% | 72.92%；同集官方参照 RoboTracer_Intrinsics_Depth=58% | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / official / 2D End-in-BBox | 100 | 100 | 96 | 0 | — | 51.04% | 51.04%；同集官方参照 RoboTracer_Intrinsics_Depth=43% | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / official / No-Collision Rate | 100 | 100 | 96 | 0 | — | 81.25% | 81.25% | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / official / 3D Start 距离(m) | 100 | 100 | 96 | 0 | 越低越好；RoboTracer_Intrinsics_Depth 见报告 | 0.1268 | 0.1268 | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / official / 3D End 距离(m) | 100 | 100 | 96 | 0 | 越低越好；RoboTracer_Intrinsics_Depth 见报告 | 0.1965 | 0.1965 | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / official / 2D DFD | 100 | 100 | 96 | 0 | 越低越好；RoboTracer_Intrinsics_Depth 见报告 | 0.2057 | 0.2057 | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / official / 3D DFD | 100 | 100 | 96 | 0 | 越低越好；RoboTracer_Intrinsics_Depth 见报告 | 0.4151 | 0.4151 | 未定义 | PASS（公开基准） |
| TraceSpatial / 2D / upstream / 3D Start | 100 | 100 | 100 | 0 | 83%（论文） | 0.00% | 0.00%；同集官方参照 RoboTracer_Intrinsics_Depth=75% | 未定义 | PASS（公开基准） |
| TraceSpatial / 2D / upstream / 3D End | 100 | 100 | 100 | 0 | 63%（论文） | 0.00% | 0.00%；同集官方参照 RoboTracer_Intrinsics_Depth=57% | 未定义 | PASS（公开基准） |
| TraceSpatial / 2D / upstream / Success | 100 | 100 | 100 | 0 | 44%（论文） | 0.00% | 0.00%；同集官方参照 RoboTracer_Intrinsics_Depth=52% | 未定义 | PASS（公开基准） |
| TraceSpatial / 2D / upstream / 2D Start-in-Mask | 100 | 100 | 100 | 0 | — | 58.00% | 58.00%；同集官方参照 RoboTracer_Intrinsics_Depth=58% | 未定义 | PASS（公开基准） |
| TraceSpatial / 2D / upstream / 2D End-in-BBox | 100 | 100 | 100 | 0 | — | 52.00% | 52.00%；同集官方参照 RoboTracer_Intrinsics_Depth=43% | 未定义 | PASS（公开基准） |
| TraceSpatial / 2D / upstream / 2D DFD | 100 | 100 | 100 | 0 | 越低越好；RoboTracer_Intrinsics_Depth 见报告 | 0.2682 | 0.2682 | 未定义 | PASS（公开基准） |
| TraceSpatial / 2D / upstream / 3D DFD | 100 | 100 | 100 | 0 | 越低越好；RoboTracer_Intrinsics_Depth 见报告 | 0.0000 | 0.0000 | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / upstream / 3D Start | 100 | 98 | 95 | 2 | 83%（论文） | 63.16% | 63.16%；同集官方参照 RoboTracer_Intrinsics_Depth=75% | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / upstream / 3D End | 100 | 98 | 95 | 2 | 63%（论文） | 56.84% | 56.84%；同集官方参照 RoboTracer_Intrinsics_Depth=57% | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / upstream / Success | 100 | 98 | 95 | 2 | 44%（论文） | 37.50% | 37.50%；同集官方参照 RoboTracer_Intrinsics_Depth=52% | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / upstream / 2D Start-in-Mask | 100 | 98 | 95 | 2 | — | 61.05% | 61.05%；同集官方参照 RoboTracer_Intrinsics_Depth=58% | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / upstream / 2D End-in-BBox | 100 | 98 | 95 | 2 | — | 46.32% | 46.32%；同集官方参照 RoboTracer_Intrinsics_Depth=43% | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / upstream / No-Collision Rate | 100 | 98 | 95 | 2 | — | 83.16% | 83.16% | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / upstream / 3D Start 距离(m) | 100 | 98 | 95 | 2 | 越低越好；RoboTracer_Intrinsics_Depth 见报告 | 0.2912 | 0.2912 | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / upstream / 3D End 距离(m) | 100 | 98 | 95 | 2 | 越低越好；RoboTracer_Intrinsics_Depth 见报告 | 0.2430 | 0.2430 | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / upstream / 2D DFD | 100 | 98 | 95 | 2 | 越低越好；RoboTracer_Intrinsics_Depth 见报告 | 0.2582 | 0.2582 | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / upstream / 3D DFD | 100 | 98 | 95 | 2 | 越低越好；RoboTracer_Intrinsics_Depth 见报告 | 0.5374 | 0.5374 | 未定义 | PASS（公开基准） |
| CV-Bench / Acc(ADE20K) | 2638 | 2638 | 2638 | 0 | 94.58%（论文 Table 2） | 77.88% | 77.88%（官方 FlagEvalMM get_result） | 未定义 | PASS（公开基准） |
| CV-Bench / Acc(COCO) | 2638 | 2638 | 2638 | 0 | 94.58%（论文 Table 2） | 85.96% | 85.96%（官方 FlagEvalMM get_result） | 未定义 | PASS（公开基准） |
| CV-Bench / Acc(Omni3D) | 2638 | 2638 | 2638 | 0 | 94.58%（论文 Table 2） | 94.33% | 94.33%（官方 FlagEvalMM get_result） | 未定义 | PASS（公开基准） |
| CV-Bench / Acc_2D | 2638 | 2638 | 2638 | 0 | 94.58%（论文 Table 2） | 81.92% | 81.92%（官方 FlagEvalMM get_result） | 未定义 | PASS（公开基准） |
| CV-Bench / Acc_3D | 2638 | 2638 | 2638 | 0 | 94.58%（论文 Table 2） | 94.33% | 94.33%（官方 FlagEvalMM get_result） | 未定义 | PASS（公开基准） |
| CV-Bench / CV-Bench 综合分 | 2638 | 2638 | 2638 | 0 | 94.58%（论文 Table 2） | 88.12% | 88.12%（官方 FlagEvalMM get_result） | 未定义 | PASS（公开基准） |

## 稳定性检查（每基准固定样例 × 2 次重复推理）

| 基准 | 样例数 | 原始输出一致 | 变化数 | 变化率 | 终态全 OK |
|---|---|---|---|---|---|
| RefSpatial-Bench | 12 | 5 | 7 | 0.5833 | True |
| TraceSpatial-Bench | 12 | 3 | 9 | 0.75 | True |
| CV-Bench | 12 | 12 | 0 | 0.0 | True |

## 离线重判分复现

- 比较指标数：**12**；不一致：**0**
- 可复现：**True**
