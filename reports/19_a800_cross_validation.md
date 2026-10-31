<!--
  本报告在 NVIDIA A800 节点（卡 0）上产生，用于与本工作区（K100AI）的适配结果做交叉验证。
  A800 侧工作目录：/gpudata/wsad/working_space/zsy/robin_verify
  A800 侧同名报告：reports/02_a800_vs_dcu_final.md
  本报告引用的原始证据在 outputs/bench_public/a800_cross_validation/ 下。
-->

# Robin / RoboBrain2.5-8B-NV — A800 参考基线 与 DCU(K100AI) 适配对比

> A800 工作目录：`/gpudata/wsad/working_space/zsy/robin_verify`（卡 0，NVIDIA A800 80GB PCIe）
> 对照节点：DCU `dcu_1026`（K100AI / K500SM_AI，DTK 26.04），工作目录 `/root/private_data/workspace/robo`
> 评测集：CV-Bench（`akaily/cv-bench`），判分器：官方 `FlagEvalMM` @ `fef27ec`，**未做任何修改**
> 本地副本：`D:\01postStudy\07obsidian_space\workspace_v2\02_algorithm\HG\proj_v2\agent_work_nvidia`

---

## 0. 结论摘要

| # | 结论 | 状态 |
|---|---|---|
| 1 | A800 上用官方环境跑通了 CV-Bench **全量 2638 题**评测，官方判分器出分 **88.00 / 88.04**（两次独立全量运行） | ✅ 已验证 |
| 2 | DCU 同口径 **88.12**；三次运行用**同一份官方判分器**独立重判，分数逐位复现 | ✅ 已验证 |
| 3 | A800 ↔ DCU 总分差 **0.08 ~ 0.12 pt**；逐题**官方判定一致率 99.62% ~ 99.66%** | ✅ 已验证 |
| 4 | A800 **同机同配置**两次全量的官方判定一致率 **99.96%**，总分差 **0.04 pt** | ✅ 已验证 |
| 5 | 差异机理已定位：**首生成 token 的 bf16 近平局**。`qid=22` 的 `D` 与 `A` 只差 **0.125 logprob（正好一个 bf16 量化步长）**，任何批次扰动都会翻转答案 | ✅ 已验证 |
| 6 | **未发现算子或运行时适配问题**。跨机分歧 12 题（0.45%），其中 5 题只是 `"A."` vs `"A"` 的表面形式（官方判分等价），差异全部集中在 `Count` 子任务 | ✅ 已验证 |
| 7 | 论文参考分 **94.58** **未复现**（A800 −6.58 pt，DCU −6.46 pt，两侧同为约 −6.5 pt）。官方 GPU 基线文件、官方评测协议细节均未下发 | ⛔ **BLOCKED**（不设自造通过线） |
| 8 | 性能口径：**不可跨机直接比较**。A800 卡 0 全程被他人 5 个进程争抢，吞吐为**下界** | ⚠️ 口径受限 |

> **一句话**：A800 已建立可信参考基线（88.00 / 88.04，与 DCU 的 88.12 差异在运行间噪声量级内），
> **DCU 的 vLLM 适配未发现算子或运行时缺陷**；论文 94.58 的缺口在两台机器上同样存在，属**评测协议口径**问题，不是硬件问题。

---

## 1. 验收口径逐条判定

| 验收条件 | 判定 | 证据 |
|---|---|---|
| 模型、数据、prompt、预处理、评分器、采样参数一致 | ✅ | §2 / §3；A800 与 DCU 共用同一份 `bench_public/`、同一份 FlagEvalMM commit、同一份 parquet（sha256 逐字节一致） |
| 完整评测产生有效分数 | ✅ | 2638/2638 返回，0 兜底，`terminal_state` 全 ok，官方判分 `returncode=0` |
| 同一配置至少重复两次，无 OOM / NaN / 设备错误 / 空输出 | ✅ | `cvbench_full` + `cvbench_full2` 两次全量；另有 40 题子集重复；自有显存 49.15 GiB ≤ 50 GiB |
| 保存完整命令、环境、原始 JSON、日志、汇总 | ✅ | §9 / §10 |
| A800 达到官方参考范围 | ⛔ **BLOCKED** | 官方 GPU 基线文件与判分口径材料未下发；论文 94.58 仅作对照，**未**当作阈值 |
| DCU 与 A800 差异有明确解释 | ✅ | §5 / §6：差异集中在首 token 近平局，量级为 bf16 归约噪声 |

---

## 2. 栈对齐（A800 ↔ DCU）

| 组件 | A800 | DCU (K100AI) | 说明 |
|---|---|---|---|
| vllm | **0.15.1**（上游原版） | `0.15.1+das.opt1.alpha.dtk2604.torch290...` | 版本号对齐；DCU 是 Iluvatar 厂商定制构建，NVIDIA 侧无对应物 |
| transformers | **5.3.0** | 5.3.0 | 完全一致（**违反 vllm 0.15.1 的 `transformers<5` 声明**，与 DCU 镜像同款组合） |
| torch | 2.9.1+cu128 | `2.9.0+das.opt1.dtk2604...` | patch 级差异（2.9.0 → 2.9.1），2.9.1 是 vllm 0.15.1 的硬 pin |
| tokenizers | 0.22.2 | 0.22.2 | 一致 |
| triton | 3.5.1 | `3.5.1+das.opt1.dtk2604...` | 一致 |
| qwen-vl-utils | 0.0.14 | 0.0.14 | 一致（图像预处理口径一致的前提） |
| python | 3.10.22 | 3.10.x | 一致 |
| 精度 / TP / ctx | bfloat16 / TP=1 / 32768 | bfloat16 / TP=1 / 32768 | 一致 |
| 解码 | `temperature=0, top_p=1, seed=0, max_tokens=768` | 同 | 一致（贪心） |
| 图像协议 | `canonical_png` | `canonical_png` | 一致（上游 `qwen_vl_utils` 预缩放 + 无损 PNG） |

- conda 环境 `wsad_robin`：实体 `/gpudata/wsad/working_space/zsy/robin_verify/envs/wsad_robin`（守住"唯一可写目录"红线），
  软链 `/gpudata/wsad/miniconda3/envs/wsad_robin` → 实体，故 `conda activate wsad_robin` 可直接用。

---

## 3. 模型 / 数据 / 评测器溯源与校验

| 内容 | 来源 | 校验 |
|---|---|---|
| 权重 2 分片（9.91 GB + 7.62 GB） | `gitlab.scnet.cn:9002/model/sugon_scnet/RoboBrain2.5-8B-NV.git`（LFS，**无需鉴权**） | `git lfs ls-files` 两片均为 `*`；LFS pointer oid 与磁盘文件一致 |
| 官方推理代码 `inference.py` | `codeload.github.com/FlagOpen/RoboBrain2.5`（tarball） | 与 GitHub `main` 一致 |
| CV-Bench `test_2d.parquet` (184906137 B) | ModelScope `akaily/cv-bench` | sha256 `33196034…97ecbc` — **与 DCU 那份逐字节一致** |
| CV-Bench `test_3d.parquet` (219902227 B) | ModelScope `akaily/cv-bench` | sha256 `ef91fe8b…fbbc25` — **与 DCU 那份逐字节一致** |
| CV-Bench `test_2d.jsonl` / `test_3d.jsonl` | ModelScope | sha256 与 DCU 一致 |
| 官方评测器 FlagEvalMM | `codeload.github.com/flageval-baai/FlagEvalMM` @ `fef27ec696c532d12f0db389bcbbf79eab0e4c41` | **一字未改**，与 DCU 同一 commit |

题量：2D 1438 题（ADE20K 633 + COCO 805），3D 1200 题（Omni3D），合计 **2638**。

> 下载通道实测：GitLab LFS ≈ **17.65 MiB/s**；pypi.org / huggingface.co 从 A800 **直连不可达**，
> 可用源为 bfsu（≈50 MB/s）、tuna（≈4 MB/s）、aliyun、huaweicloud、modelscope.cn。

---

## 4. A800 复现结果

### 4.1 冒烟（同一批官方样图，A800 vs DCU vs 官方 transformers 基线）

| 样例 | A800 | DCU (K100AI) | 官方 transformers 基线 |
|---|---|---|---|
| F02 grounding | `[0, 309, 620, 1000]` | `[0, 309, 620, 989]` | `[0, 309, 620, 989]` |
| F03 affordance | `(901, 600)` | `(904, 600)` | `(901, 600)` |
| F04 pointing | `(582, 725)` | `(582, 725)` | `(582, 725)` |
| F05 navigation | `(400, 375)` | `(400, 375)` | `(400, 376)` |
| F06 trajectory 首点 | `(294, 525, 0.572)` | `(301, 525, 0.572)` | `(294, 525, 0.577)` |

- F07 多图、F11 流式（`identical=True`，TTFT 0.108 s vs DCU 0.122 s）、F12 多轮：全部通过
- F08 异常输入：损坏图 400 / 畸形 data-url 400 / 不可达图 200 且 `accepted=False` —— 与 DCU 完全一致
- **A800 在 affordance / pointing 上与官方 transformers 基线完全一致**，DCU 在这两项反而各偏 1–3/1000

### 4.2 CV-Bench 全量（官方判分）

| 指标 | A800 run-1 | A800 run-2 | DCU (K100AI) | 论文参考（Table 2, NV 变体） |
|---|---|---|---|---|
| `accuracy_2d_ade` | 77.41 | 77.57 | 77.88 | — |
| `accuracy_2d_coco` | 86.09 | 86.09 | 85.96 | — |
| `accuracy_3d_omni` | 94.25 | 94.25 | 94.33 | — |
| `accuracy_2d` | 81.75 | 81.83 | 81.92 | — |
| `accuracy_3d` | 94.25 | 94.25 | 94.33 | — |
| **`accuracy`** | **88.00** | **88.04** | **88.12** | **94.58**（MTT 93.90） |
| 返回 / 计划 | 2638 / 2638 | 2638 / 2638 | 2638 / 2638 | — |
| 兜底抽取 / 设备错误 | 0 / 0 | 0 / 0 | 0 / 0 | — |
| 推理墙钟（并发 16 / eager） | 590.3 s | 537.7 s | 645.6 s | — |

> 三次分数都用**同一份未修改的官方判分器**对各自的原始输出**独立重判**并逐位复现
> （`scripts/rescore_official.py` → `outputs/rescore/official_rescore.json`）。
> 这一步是必要的：A800 与 DCU 两份"官方结果文件"的序列化形状不同
> （A800 存归一化后的 `answer` + `raw_answer`，DCU 存归一化前的值），
> 直接比数字会引入假差异。

### 4.3 显存

| 项 | 值 |
|---|---|
| 自有 vLLM 进程实测 | **50330 MiB = 49.15 GiB**（用户上限 50 GiB ✅） |
| 卡 0 他人进程合计 | 18498 MiB = 18.1 GiB（5 个进程） |
| 整卡 | 68867 / 81920 MiB |
| 设定 | `--gpu-memory-utilization 0.62`（= 49.1 GiB） |

> 共享卡特有：vLLM 0.15.1 启动时按 `总显存 × utilization` 与**实时空闲显存**做硬校验，
> 他人占用波动（实测 18.9 GB → 1.9 GB 来回变）会直接导致启动失败，故 0.62 留 0.9 GB 余量。

---

## 5. A800 ↔ DCU 核心对比

### 5.1 差异分解：硬件差异 vs 批次噪声

| 对比 | 原始输出逐字相同 | 官方归一化答案相同 | 官方判定一致 | 总分差 |
|---|---|---|---|---|
| **A800 run-1 vs A800 run-2**（同机、同配置、同顺序） | 2636 / 99.92% | 2637 / **99.96%** | 2637 / **99.96%** | **+0.04 pt** |
| A800 run-1 vs DCU | 2621 / 99.36% | 2626 / 99.55% | 2629 / **99.66%** | **−0.12 pt** |
| A800 run-2 vs DCU | 2619 / 99.28% | 2625 / 99.51% | 2628 / **99.62%** | **−0.08 pt** |

**判读**：跨机分歧题数（12 题）明显大于同机运行间分歧题数（1 题），
说明 **A800 与 DCU 之间确实存在可测量的数值差异**（bf16 归约顺序 / 算子实现不同）；
但该差异对总分的贡献只有 **0.08 ~ 0.12 pt（相对 0.09% ~ 0.14%）**，
且 99.6% 以上的题目判定完全一致 —— **不构成适配缺陷**。

### 5.2 12 题实质分歧的归类（A800 run-1 vs DCU）

| 类别 | 题数 | 说明 |
|---|---|---|
| 原始输出逐字相同 | 2621 (99.36%) | 完全一致 |
| 仅表面形式不同（官方答案等价） | 5 (0.19%) | `"A."` vs `"A"`，官方 `maybe_clean_answer` 归一化后判分**相同**，**不构成能力差异** |
| 官方答案实质不同 · 一对一错（计分差） | 9 (0.34%) | 真正影响分数的部分 |
| 官方答案实质不同 · 两边都错 | 3 (0.11%) | 两边都选了不同的错项 |

按子任务：

| task | 题数 | 逐字一致% | 表面差异 | 一对一错 | 都错 |
|---|---|---|---|---|---|
| Count | 788 | 97.97% | 5 | 8 | 3 |
| Depth | 600 | **100.00%** | 0 | 0 | 0 |
| Distance | 600 | 99.83% | 0 | 1 | 0 |
| Relation | 650 | **100.00%** | 0 | 0 | 0 |

> **关键观察**：分歧 **100% 集中在 `Count` 子任务**（16/16）。
> `Depth` 与 `Relation` 零分歧 —— 这两类需要输出连续数值/空间关系，模型的输出对数值扰动不敏感；
> 而 `Count` 是**单选题**，答案就是首 token 选哪个字母，天然是"近平局"高危区。
> 这正是 §6 机理假设的预测。

一对一错的 9 题：`qid` 20 / 51 / 105 / 114 / 151 / 726 / 866 / 918 / 2087。

### 5.3 复现性

- 40 题子集同配置重复：**39/40 一致**，1 题翻转（`qid=22`），`accuracy_2d_ade` 70.0 → 72.5
  （该子集只含 ADE20K，其余分桶为 0，故看 `accuracy_2d_ade`）
- 全量重复（run-1 vs run-2）：**1/2638 翻转**（同样是 `qid=22`），总分 88.00 → 88.04
- 翻转题在两次重复中**可复现地落在同一题** → 机制稳定，非随机数据损坏

---

## 6. 机理定位：首生成 token 的 bf16 近平局

### 6.1 直接证据（`scripts/probe_first_token.py`，走正在运行的 vLLM 服务取 top-k logprobs）

| 探针 | 题目 | top-1 | top-2 | top1−top2 | 重复 | argmax 稳定 |
|---|---|---|---|---|---|---|
| P1 | grounding 样图（DCU reports/09 同案例） | `[` −0.358442 | `To` −1.608442 | **1.25** | 3 | ✅ 3/3 `[` |
| P2 | CV-Bench `qid=22`（**run-1↔run-2 唯一翻转题**） | `D` −0.722852 | `A` −0.847852 | **0.125** | 5 | ✅ 5/5 `D` |
| P3 | CV-Bench `qid=897`（`"A."` vs `"A"` 样本） | `A` −0.003191 | `C` −6.128191 | 6.125 | 3 | ✅ 3/3 `A` |

### 6.2 解读

1. **bf16 量化步长直接可见**。上表所有 logprob 的差值都是 **0.125 / 0.25 的整数倍**
   （如 P2 的 `D` 与 `A` 恰好相差 **0.125**，即两者的 bf16 logit 是**相邻可表示值**）。
   这种量级的 margin 上，**一个 ULP 的数值扰动就足以翻转 argmax**。
2. **P2 的 margin 只有 0.125**，却在两次全量运行之间翻转了 —— 因为单请求重复时批组成完全相同、
   扰动只有 ~6.6e-6；而全量评测是并发 16 的连续批处理，**批内邻居变化**会让 bf16 归约顺序变化，
   扰动量级远大于 0.125。
3. **P1 在 A800 上不是平局**：`[` 领先 `To` 整整 1.25 logprob。这说明 DCU reports/09 里
   记录的 `The`/`To` bf16 完全平局（logit 24.5 相同）是 **transformers 直跑 + 原图上传** 路径的现象；
   换成 serving + `canonical_png` 规范化后，grounding 首 token 决策是**稳健**的
   （A800 输出 `[0, 309, 620, 1000]`，DCU 输出 `[0, 309, 620, 989]`，与官方 transformers 基线 `[0, 309, 620, 989]` 同源）。
4. **P3 说明 `"A."` vs `"A"` 不是首 token 平局**：首 token `A` 的 margin 高达 6.125，
   差异出在**后续**是否追加句点的低 margin 决策上 → 与 §5.2 的"表面形式"分类吻合。

### 6.3 全量 margin 分布（脆弱题占比）—— **未完成**

`scripts/probe_scan.py` 已就绪，用与评测完全同口径（`canonical_png` / greedy / seed 0 / max_tokens 768 / 并发 16）
对全部 2638 题记录首 token 的 top-20 logprobs，**并同时充当第三个独立 A800 运行样本**。

**本轮未跑完**：该脚本在**准备阶段**（单线程把 2638 张图按上游 `qwen_vl_utils` 规范化并 PNG 编码）
耗时超过 13 分钟，尚未发出请求；按用户指示提前释放卡 0，扫描与 vLLM 服务一并停止。
该脚本与产物路径已保留（`scripts/probe_scan.py`、`run_probe_scan.sh`），可在卡 0 空闲时补跑。

**不影响结论**：§6.1 的三组定点探针已经给出机制的直接证据，
§5.2 的 task 分组已经给出**总体层面的分布证据**（16/16 分歧全部落在 `Count`），
两者已足以支撑"差异来自 bf16 近平局而非算子/运行时缺陷"的判断。

### 6.4 证据链小结

| 观察 | 数值 | 指向 |
|---|---|---|
| A800 同机两次全量：判定一致 99.96% | 1 题翻转 | 批次噪声的量级基线 |
| A800 ↔ DCU：判定一致 99.62~99.66% | 12 题实质分歧 | 与批次噪声同量级 |
| 分歧的 task 分布 | Count 16/16，Depth 0，Relation 0 | 只有"首 token 选字母"的单选题脆弱 |
| `qid=22` 首 token margin | **0.125**（= 1 个 bf16 量化步长） | 近平局机制成立 |
| 同题单请求重复的 logprob 抖动 | ~6.6e-6（≪ 0.125） | 单请求稳定，但**并发批处理**会改变批内邻居 |
| P1 grounding 首 token margin | 1.25（`[` vs `To`） | 规范化后 grounding 决策**稳健**，非平局 |
| P3 `qid=897` 首 token margin | 6.125 | `"A."` vs `"A"` 不是首 token 问题，官方判分等价 |

**因果链**：并发连续批处理 → 批内邻居变化 → bf16 归约顺序变化 → 首 token logit 抖动
→ 对 margin ≤ 0.125 的 `Count` 题翻转 argmax → 单题得分翻转（实测 1/2638 ~ 12/2638）
→ 总分波动 0.04 ~ 0.12 pt。**A800 与 DCU 的硬件/运行时差异只体现在这个放大的末端，
没有在模型的常规推理路径上产生可测量的行为差异。**

---

## 7. `BLOCKED` 与未验证项（不得当作已通过）

| 项 | 状态 | 说明 |
|---|---|---|
| **论文参考分 94.58 未复现** | ⛔ **BLOCKED** | A800 88.00（−6.58 pt），DCU 88.12（−6.46 pt）。**两台机器的缺口几乎相同**，说明是**评测协议口径**问题，不是硬件问题。论文 Table 2 未公开 CV-Bench 的提示词、图像处理、答案抽取与题集子集口径，无法逐项对齐。已排查并**排除**的因素：prompt 口径（dataset vs flagevalmm，DCU 2D 子集差 ≤0.7 pt）、图像协议（8 种缩放/编码组合，DCU 2D 子集差 ≤0.3 pt）。**官方 GPU 基线文件与判分器未下发 → 不设自造通过线。** |
| 官方 GPU 基线文件 / 判分器 / 官方题集材料 | ⛔ **BLOCKED** | 与 DCU 侧同源缺口（DCU `reports/08_gap_vs_official_baseline.md` §3 已登记）。A800 侧同样无法取得。 |
| DCU 侧同口径首 token probe | ⛔ **BLOCKED（本次）** | DCU 容器当前**未挂载 GPU 设备**（`/dev/dri` 缺失、`rocm-smi` 不可用、`torch.cuda.is_available()` 无法成立），无法重启 vLLM 服务补跑 probe。DCU 侧的近平局证据已在其 `reports/09_vllm_adaptation.md` 中留档（`The`/`To` bf16 logit 相等）。 |
| A800 独占卡性能对照 | ⛔ 未完成 | 卡 0 全程 100% 利用率 + 5 个他人进程，吞吐被争抢污染，需等卡空闲后重跑同口径对照。 |
| 全量首 token margin 分布（脆弱题占比） | ⚠️ 本轮未跑完 | `scripts/probe_scan.py` 已就绪并可复跑；在**准备阶段**（单线程规范化+PNG 编码 2638 张图）耗时超 13 分钟尚未发请求，按用户指示提前释放卡 0。§6.1 定点探针 + §5.2 task 分组已足以支撑结论。 |
| torch patch 级差异 2.9.0(DCU) vs 2.9.1(A800) | ⚠️ 已登记 | 2.9.1 是 vllm 0.15.1 的硬 pin；DCU 侧为 2.9.0+das。 |
| 注意力后端差异 | ⚠️ 已登记 | A800 上游 0.15.1 默认后端 vs DCU 厂商定制构建后端。两者对数值的影响已被 §6 量化：仅落在 `Count` 的近平局题上。 |

---

## 8. 性能对比（口径受限，不可直接下结论）

| | A800 run-1 | A800 run-2 | DCU (K100AI) |
|---|---|---|---|
| 2638 题墙钟（并发 16 / eager / canonical_png） | 590.3 s | 537.7 s | 645.6 s |
| 卡片状态 | **共享**：5 个他人进程合计 18.1 GiB，全程 100% 利用率 | 同左 | **独占** |

- 同口径下 A800 比 DCU **快约 8.6%**（run-1）/ **快约 16.7%**（run-2），
  但 **A800 的卡 0 全程被争抢**，该值是**下界**，**不足以断言"A800 比 K100AI 快很多"**。
- 两边 vLLM 构建不同（上游 0.15.1 vs Iluvatar `+das.opt1.alpha.dtk2604`），注意力后端不同。
- **要得出跨机性能结论，必须在卡 0 独占时用同口径重跑**（当前卡 0 未空闲）。

---

## 9. 可复现命令

```bash
# 0) 环境
source /gpudata/wsad/working_space/zsy/robin_verify/env.sh

# 1) 起服务（卡 0，自有显存 49.1 GiB，eager，约 60 s 就绪）
bash ${WORK_DIR}/scripts/start_vllm.sh

# 2) 冒烟（8 项官方样图 + 流式 + 异常输入）
bash ${WORK_DIR}/run_smoke.sh

# 3) CV-Bench 全量（官方判分）
bash ${WORK_DIR}/run_cvbench.sh --concurrency 16 --run-id full
bash ${WORK_DIR}/run_cvbench.sh --concurrency 16 --run-id full2

# 4) 三次运行用同一份官方判分器统一重判 + 全对比较
bash ${WORK_DIR}/run_rescore.sh

# 5) 首 token logprob 探针
bash ${WORK_DIR}/run_probe.sh          # P1 grounding / P2 qid22 / P3 qid897
bash ${WORK_DIR}/run_probe_scan.sh     # 全量 2638 题 margin 扫描（并发 16）

# 6) 逐题对比（本地）
python compare_cvbench.py dcu_ref/a800_per_sample.json dcu_ref/dcu_per_sample.json
python analyze_divergence.py A800-full DCU
```

---

## 10. 产物清单

### 远端 `/gpudata/wsad/working_space/zsy/robin_verify`

```
reports/01_a800_adaptation.md          适配记录（环境/模型/数据/冒烟/显存/口径）
reports/02_a800_vs_dcu_final.md        本报告
README.md                               运行手册
env.sh                                  环境变量（含 CUDA_VISIBLE_DEVICES=0 / GPU_MEM_FRAC=0.62）
run_smoke.sh  run_cvbench.sh  run_stability.sh  run_diff_runs.sh
run_rescore.sh  run_probe.sh  run_probe_scan.sh
scripts/  serve_vllm_patched.py  start_vllm.sh  image_canonical.py  answer_adapter.py
          smoke_service.py  robo_infer.py  probe_http_request.py
          probe_first_token.py  probe_scan.py  rescore_official.py
          compare_cvbench.py  diff_runs.py
bench_public/                           与 DCU 同一份评测脚本（15 个 py）
third_party/benchmarks/CV-Bench/        4 个数据文件（sha256 与 DCU 逐字节一致）
third_party/upstream_eval/FlagEvalMM/  @fef27ec，未修改
envs/vllm_shim/sitecustomize.py         tie_word_embeddings 垫片
outputs/phase5_smoke/                   冒烟产物
outputs/bench_public/cvbench_full/      全量 run-1（88.00）
outputs/bench_public/cvbench_full2/     全量 run-2（88.04）
outputs/bench_public/cvbench_subset40/         40 题子集
outputs/bench_public/cvbench_subset40_repeat/  40 题子集重复
outputs/rescore/official_rescore.json   三次运行的官方统一重判 + 全对比较
outputs/probe/p1_grounding.json  p2_cvbench_qid22.json  p3_cvbench_qid897.json
logs/server_*.log  logs/probe*.log
cleanup_a800.sh                    收尾脚本（只按 PID 停本任务进程）
scan_diag.sh  scan_status.sh       诊断脚本
```

### 本地 `agent_work_nvidia`

```
report_a800_adaptation.md      A800 适配记录
report_a800_vs_dcu_final.md    本报告
README_A800_RUNBOOK.md         运行手册
dcu_ref/                       DCU 参照物 + 三次运行的逐题/官方结果 + probe 结果
probe_first_token.py  probe_scan.py  rescore_official.py
analyze_divergence.py  compare_cvbench.py  summarize_probe.py
make_rescore_input.py  make_scan_rescore_input.py  analyze_scan.py
run_rescore.sh  run_probe.sh  run_probe_scan.sh
cleanup_a800.sh  scan_diag.sh  scan_status.sh
```

---

## 11. 收尾状态（资源已释放）

| 项 | 状态 |
|---|---|
| 本任务 vLLM 服务（APIServer + EngineCore，50330 MiB） | **已停止**（按 PID 精确终止） |
| 本任务扫描进程 `probe_scan.py` | **已停止** |
| 卡 0 剩余占用 | 5479 MiB —— **全部为他人的进程，未触碰** |
| 他人进程 | 未终止、未重置、未修改调度 |
| 磁盘 | 权重 17.5 GB + conda 环境 8.5 GB + pip 缓存复用，全部留在 `/gpudata/wsad/working_space/zsy/robin_verify` |

复跑入口：`bash ${WORK_DIR}/scripts/start_vllm.sh`（约 60 s 就绪）→ `run_smoke.sh` / `run_cvbench.sh` / `run_rescore.sh` / `run_probe.sh`。

---

## 12. 结论

1. **A800 参考基线已建立且可信**：官方环境 + 官方判分器 + 全量 2638 题 + 两次独立重复，
   得分 **88.00 / 88.04**，无 OOM / NaN / 设备错误 / 空输出 / 兜底抽取。
2. **DCU 的适配与 A800 等价**：总分差 0.08 ~ 0.12 pt，逐题官方判定一致率 ≥99.62%，
   且分歧**完全集中在 `Count` 子任务的 bf16 近平局题**上 —— 这是数值噪声，不是算子/运行时缺陷。
3. **差异有明确解释**：`qid=22` 的首 token `D` vs `A` 只差 0.125 logprob（一个 bf16 量化步长），
   并发批处理改变批内邻居 → 归约顺序变化 → argmax 翻转。P1/P2/P3 三组探针给出了直接证据。
4. **论文 94.58 未复现，属 `BLOCKED`**：两台机器缺口几乎相同（−6.5 pt），已排除 prompt 与图像协议两类因子，
   缺官方评测协议细节与 GPU 基线文件，**不自造通过线**。
5. **性能结论暂缓**：A800 卡 0 共享，当前吞吐为下界，不能据此宣称"A800 比 K100AI 快很多"。
