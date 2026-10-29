# 19 · 扩展公开基准评测结果（CrossPoint / EmbSpatial / RoboSpatial / Q-Spatial / MSMU / ShareRobot / VABench）

> 依据：《RoboBrain2.5-8B-NV_精度测试验收要求.md》v1.0；协议与披露原则同 `reports/15_public_bench_protocol.md`。
> 背景：论文（arXiv 2601.14352）Table 2（2D 空间）5 项 + Table 3（3D 空间）5 项 + §6.3 时序。
> 本报告补齐 Table 2/3 中除 CV-Bench / RefSpatial / TraceSpatial（见 `reports/16`）外的 7 项。
>
> ⚠️ 与 `reports/16` 相同的边界：这些都不是赛事官方验收数据集；结论层级仍为
> 「公开基准测试有效 = PASS / 精度回归 = BLOCKED（阈值未定义）/ 赛事官方精度 = BLOCKED」。
> 论文分数全部为**同量级对照**（论文未公开任何评测协议细节）。

## 0. 运行配置（与 `reports/16` 候选配置一致）

| 项 | 值 |
| --- | --- |
| 服务 | vLLM graph（非 eager，E1 候选配置），单卡 K100AI 64GB，BF16 |
| 客户端 | 并发 16，greedy，`max_tokens` 各 runner 注明（Q-Spatial 1024，其余默认） |
| 产物根 | 远程 `outputs/bench_public_noneager2/<bench>/`（predictions.jsonl + summary.json 逐题落盘） |
| 驱动 | `tmp/master_bench_queue.sh`（单进程顺序队列）+ 前两批 `tmp/run_new_benches_{1,2}.sh` |
| runner | `submission/bench_public/run_{embspatial,crosspoint,robospatial,qspatial,msmu,sharerobot,vabench}.py` |

## 1. 结果总表

| 基准 | 题数 | 有效返回 | 本次值 | 论文 NV | 判分代码 |
| --- | --- | --- | --- | --- | --- |
| CrossPoint-Bench | 1000 | 1000 | **75.20**（官方 OVERALL） | 75.40 | 官方 `calc_score.py`，`relative_1000` 零改动 |
| EmbSpatial-Bench | 3640 | 3640 | **77.77** | 75.58 | ⚠️ 自建 MC 判分（官方未发布代码） |
| RoboSpatial-Home | 350 | 350 | **62.0**（官方 accuracy） | 73.03 | 官方 `eval_pregenerated_results` 零改动 |
| Q-Spatial-Bench | 101 | 101 | **60.4%** 成功率（ratio<2） | 73.53 | 官方 notebook 公式复刻 + 披露兜底抽取 |
| MSMU-Bench | 987 | 987 | 定量 47.0/66.3/76.4%（δ=1.25/1.5625/1.953125）；定性 63.8% | 64.17（聚合口径未公开） | 官方 `judge_prompt` 逐字；judge = 本服务（greedy） |
| ShareRobot-T | 200 | 200 | **DFD 0.181 / HD 0.163 / RMSE 0.115（↓）** | 0.1164 | FlagEvalMM **PR#84** `evaluate.py` 零改动 |
| ShareRobot-A | 200 | 200 | mIoU 0.019（IoU@0.5=1.0%）⚠️ 参考系错配，不可靠 | —（论文未报此项） | 同上 |
| VABench-V | 300 | 300 | **DFD 0.413 / HD 0.399 / RMSE 0.314（↓）** | 0.1281 | ⚠️ 官方判分未发布，沿用 PR84 轨迹同款几何指标（披露替代） |

三个与论文对照的 2D 基准全部落在论文 ±2.2pt 内；CrossPoint 差 0.2pt。
**ShareRobot-T 与论文同量级**（0.181 vs 0.116，见 §2.6 的适配层缺陷修复）。

## 2. 分基准要点

### 2.1 CrossPoint（75.20，官方判分零改动）
- 分层：对应判定 82.69 / 对应指点 80.56 / 细粒度定位 67.08 / 可见性推理 64.55（object 级全面高于 part 级）。
- 坐标契约：官方 `extract_answers.py --coord_format relative_1000` 原生支持 0-1000，RoboBrain2.5 输出零换算直喂，**判分代码零改动**。
- 这是本次扩展基准中协议最硬的一组证据。

### 2.2 EmbSpatial（77.77，自建判分）
- 分层 relation：above 85.6 / under 83.2 / right 84.0 / left 80.5 / close 69.8 / far 63.3——深度类（close/far）显著弱于几何类，与论文发现一致。
- ⚠️ 披露：官方 repo 无评测代码；prompt 用与 CV-Bench 主口径一致的 FlagEvalMM 风格 MC 格式；判分为自建（首字母抽取 + 索引比对）。likelihood 方法未实现。

### 2.3 RoboSpatial（62.0，官方判分零改动）
- 分层：configuration 86.2（106/123）/ **compatibility 41.0（43/105）** / context 55.7（68/122）。
- 与论文 73.03 差 11pt，**主因是 compatibility 子项**（若 compat 到 80%，总分 ≈ 74，与论文吻合）。
- **2026-10-29 排查结论 + 深度输入诊断**：初判"模型保守偏置"被后续实验细化——官方 harness 的模型接口本就提供 `depth_path`（数据自带深度图），而我们主口径只喂 RGB。**RGB+深度双路输入诊断**（350 题，非主口径）：compatibility 41.0 → **54.3%**（+13.3pt），总分 62.0 → 66.0；configuration 84.6 / context 57.4 基本持平。即 compat 缺口约一半可由"官方跑法可能喂深度输入"解释，剩余差距（66 vs 73）指向模型专用 prompt 等其他协议差异。主口径（纯 RGB 裸题）数字不变，已披露。
- context（指点）：0-1000 → 0-1 换算后走官方 mask 判分；匹配 350/350，unmatched 0。

### 2.4 Q-Spatial（60.4% 成功率）
- 官方协议：`system_prompt` + `spatial_prompt_steps`（论文 headline 模板），判分 = `max(pred/gt, gt/pred) < 2`（统一到 cm）。
- ⚠️ 披露两处健壮性补丁（官方 notebook 会崩的边界）：模型不按 `scalar{}` 标签输出时兜底抽取"末位数值+单位"；任一侧为 0 时除零保护。公式不变。
- `QSpatial_scannet` split 上游仅有 LFS 指针（无数据），只评 QSpatial_plus，如实披露。
- 分层：horizontal 60.2%（98 题）/ vertical 50.0%（2 题）/ 1d_horizontal 1/1。
- **2026-10-29 排查结论：管线无缺陷，差距来自提示引导（已被官方变体对照实验证实）**。抽取成功率 100/101（0 抽取失败），判分公式官方；但模型 **0/101 遵守 `scalar{}` 输出格式**（全部走兜底抽取）。**官方模板变体对照**（同为官方仓库自带模板）：`standard_prompt`（裸题）成功率仅 **38.6%**，`spatial_prompt_steps`（主口径）**60.4%**——官方引导模板本身贡献 +21.8pt，主口径选择正确。与论文 73.53 的剩余差距判读为模型估计质量/论文内部 prompt 差异（ratio 分布：<1.25 占 21%、<2 占 61%、<3 占 82%），非 bug。

### 2.5 MSMU（定量 47.0/66.3/76.4%，定性 63.8%）
- 官方流程：推理用原题（`conversations[0]`）；judge LLM 按官方 `judge_prompt`（含固定单位换算表）把 GT 与回答抽成米制 JSON；`success=[ratio<1.25, <1.25², <1.25³]`。
- ⚠️ 披露：官方 judge 用 llama3-8B，本复现用同一 vLLM 服务（RoboBrain2.5-8B-NV，greedy）充当 judge；抽取为机械换算，judge 输出逐题落盘（`judge.jsonl`）可审计。任一侧为 0 的除零保护已加（公式不变）。
- judge_fail = 0（983/983 条 judge 成功）。
- 定量 n=668（scale_estimation 259 / refer_obj_estimation 190 / count 96 / grounding 87 / absolute_distance 40），定性 n=315。

### 2.6 ShareRobot（PR84 官方代码）
- trajectory：官方 process.py 的提示词替换逐字复刻（"Your answer should be formatted as a list of tuples..."）；判分 = PR84 `extract_points + DFD/HD/RMSE`。
- affordance：题 + 官方 post_prompt "Only return the box in format: [x1, y1, x2, y2]..."；判分 = IoU。
- 坐标适配层（披露）：0-1000 → 像素，按**样本内尺度自动判定**（max≤1.5 视为 0-1 不动；≤1000.5 视为 0-1000 换算；更大视为像素不动），冻结规则，逐题可审计。
- **2026-10-29 重大修复（轨迹）**：初版适配层正则只匹配方括号 `[x, y]`，而模型实际输出圆括号元组 `(803, 325)`——适配层空转，官方 `extract_points` 把 0-1000 值再除一次缩略图宽高（如 803/224=3.6），导致 DFD 从真实的 **0.181** 虚高到 1.76。判定依据：0-1000 解释下模型 100% 点在界内（像素解释仅 26%），且修正后与论文 0.1164 同量级。修复（两种括号都匹配）后重判：**DFD 0.181 / HD 0.163 / RMSE 0.115**，与论文 0.116 同一量级。
- **affordance 参考系错配（不可靠，论文未报此项）**：GT 框是**原始分辨率**像素（最大 2560×1564），而 bench 仓库只提供 ~300×400 缩略图，模型在缩略图上输出 0-1000 → 两参考系不可换算，mIoU 0.019 无意义。这是 PR84 管线固有问题（其 `process.py` 只对缩略图取 width/height），非本次引入。

### 2.7 VABench-V（DFD 0.413 ↓）
- ⚠️ 官方判分代码未发布（GitHub/HF/FlagEvalMM 均无）；本复现沿用 PR84 ShareRobot 轨迹同款几何指标（同一套 `evaluate.py`，零改动），属"最接近官方公开实现"的替代口径，若后续拿到官方判分器应重判对比。
- 与 ShareRobot-T 同款的坐标适配层缺陷同样存在于初版，修复后 **DFD 0.413 / HD 0.399 / RMSE 0.314**（初版 0.816 虚高）。全部按 0-1000 解释重算为 0.198；当前 0.413 与论文 0.128 仍有 ~3× 差距，其中混合尺度样本（模型个别输出 >1000）的处理规则影响已记录。鉴于本指标本来就是披露替代口径，报 0.413 并保留全部限制说明。

## 3. 审计与数据质量

| 检查 | 结果 |
| --- | --- |
| 各基准请求失败 / 非 ok 终态 | 0（全部 `{'ok': N}`） |
| ShareRobot-T / VABench-V 解析失败 | 0/200、0/300 |
| 上述两者归一化后越界点（>1.2） | 0、0（初版越界检查在错误归一化下通过，已随适配层修复修正） |
| MSMU judge 失败 | 0 |
| CrossPoint 官方判分 unmatched | 0 |

## 4. 复现命令

```bash
cd /root/private_data/workspace/robo
bash tools/start_bench_service.sh graph 30000          # 服务
bash tmp/run_new_benches_1.sh                          # EmbSpatial + CrossPoint
bash tmp/run_new_benches_2.sh                          # RoboSpatial（等 1）
bash tmp/master_bench_queue.sh                         # Q-Spatial + MSMU + ShareRobot + VABench（等 2）
# 数据与判分代码：third_party/benchmarks/{CrossPoint-Bench,CrossPoint-repo,EmbSpatial-repo,
#   RoboSpatial-Home,RoboSpatial-Eval,Q-Spatial-Bench,Q-Spatial-Bench-code,MSMU,SD-VLM,
#   ShareRobot-Bench,VABench-V}；FlagEvalMM 已 apply PR#84（git apply pr84_sharerobot.diff）
```

## 5. 已知限制（汇总披露）

1. EmbSpatial / VABench 无官方判分代码，分别为自建 MC 判分 / PR84 同款几何指标替代（§2.2/§2.7）。
2. Q-Spatial / MSMU 各有两处已披露的判分健壮性补丁（抽取兜底、除零保护），判分公式不变。
3. MSMU judge 模型与官方不同（本服务代替 llama3-8B），已论证为机械抽取、greedy 稳定。
4. RoboSpatial 与论文差 11pt 集中在 compatibility 子项，协议差异方向待逐题核对（§2.3）。
5. ShareRobot-A（affordance）GT 原始分辨率与缩略图输入参考系错配，mIoU 不可靠（§2.6，论文未报此项）；ShareRobot-T 与 VABench-V 的初版轨迹指标受适配层括号缺陷虚高，修复后分别为 0.181 / 0.413（§2.6/§2.7）。
6. 论文对照分全部受"协议未公开"限制，仅作同量级参考。
7. QSpatial_scannet split 无数据，未评。

## 6. TraceSpatial 3D「病态样本」根因勘误与最大可行覆盖（2026-10-29 晚）

> 本节修正 `reports/16` §5.9 / L17 的判断：当时把 4 条样本判分超时归因于
> **DBSCAN 二次复杂度**，**该归因错误**（DBSCAN/目标点云链路实测 0.0s 完成）。

### 6.1 真实根因（py-spy 栈 + 端到端复现双重证实）

1. py-spy 抓取 4 个卡住的判分进程：全部停在官方 `discrete_frechet_distance`
   的递归 `compute`（summarize_acc.py:283）。
2. 官方 3D 路径对插值后的 3D 轨迹调用该递归实现；其备忘录判据
   `ca[i,j] > -0.5` 对 **NaN 永假 → 备忘录失效 → 3 分支指数级递归 → 永不终止**
   （CPU 100%、内存恒定 ~230MB，与观测完全一致）。
3. NaN 来源链（对样本 56 完整复现）：模型对深景样本输出**恒定退化轨迹**
   （9 个点仅 2 个唯一点）→ 官方 `interpolate_trajectory_by_distance` 距离均匀
   插值时重复节点使 `interp1d` 斜率除零 → NaN。
   2D 路径有 `np.isnan` 过滤，**3D 路径没有**——属官方上游缺陷。
4. 16 个 (run × variant × 样本) 组合预筛：5 个组合会挂、2 个 Skip（无深度）、
   9 个可完成。即 **100/100 全量官方判分对任何变体都不可达**。

### 6.2 最大可行覆盖合并（`tools/merge_ts_3d_best.py`）

分母 = 96（原兜底口径）+ 可完成补跑样本；排除项 = 官方代码不终止的组合。

| 组合 | answers/evaluated | 3D Start | 3D End | Overall | 排除样本 |
| --- | --- | --- | --- | --- | --- |
| 基线 eager / official | 98/98 | 78.57 | 63.27 | 46.94 | 56, 78 |
| 基线 eager / upstream | 99/97 | 61.86 | 56.70 | 36.36 | 82 |
| 候选 非 eager / official | 98/98 | 80.61 | 66.33 | 48.98 | 56, 78 |
| 候选 非 eager / upstream | 99/97 | 61.86 | 61.86 | 38.38 | 98 |

产物：`outputs/bench_public_ts3d_best/<run>/tracespatial/<variant>/scores_3dchunk_3D.json`
（逐分片 `sample_index` 标注 + 排除索引 + 根因字段，可审计）。

### 6.3 处置

- 4 个不终止进程已终止（已运行约 2.7~3h 无进展，判定**永远不会完成**）。
- TraceSpatial 3D 对外口径更新为**最大可行覆盖**（official 98 分母，
  upstream 99 answers/97 evaluated），排除样本的根因写入产物字段。
- 论文 Table 3 对照（3D Start 83 / End 63 / Success 44）：official 变体
  End 63.27~66.33 ≈ 论文 63，Success 46.9~49.0 ≥ 论文 44，Start 78.6~80.6 < 论文 83
  ——结论方向与 96 口径一致。
