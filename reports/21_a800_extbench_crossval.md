# 21 · A800 交叉验证：Q-Spatial / RoboSpatial（K100 推理保真度旁证）

> 2026-10-29 晚。目的：为 K100(DCU) 上"Q-Spatial 60.4% / RoboSpatial 62.0 低于论文"
> 的结论提供 NVIDIA 侧旁证——若 A800 复现同一量级，则缺口在 bench 配置/模型侧；
> 若 A800 显著更高，才回头查 K100 推理栈。

## 1. 对齐口径

| 项 | K100 (DCU) | A800 (NVIDIA) |
| --- | --- | --- |
| 模型 | RoboBrain2.5-8B-NV（SCNet 只读权重） | 同权重（robin_verify/models 副本） |
| 数据 / runner / 判分 | — | **同一份 tar 包**（MD5 `76f8b5c4…` 一致），零改动 |
| 服务 | vLLM 0.15.1 Iluvatar 定制构建，graph 模式 | vLLM 0.15.1 上游原版，graph 模式（`scripts/serve_vllm.sh --no-eager`） |
| torch / transformers | 2.9.0+das / 5.3.0 | 2.9.1+cu128 / 5.3.0 |
| 卡 / 显存 | 独占 1 卡 | 0 号卡（当时最空），52GB 上限（mem-frac 0.65） |
| 客户端 | 并发 16，greedy，max_tokens 1024 | 并发 8，其余相同 |
| 驱动 | — | `robin_verify/run_extbench_a800.sh`（产物 `extbench/outputs/`） |

## 2. 结果

| 基准 | K100 (DCU) | A800 (NVIDIA) | Δ | 论文 NV |
| --- | --- | --- | --- | --- |
| Q-Spatial 成功率（ratio<2） | 60.4% | **69.3%** | **+8.9pt** | 73.53 |
| RoboSpatial 总分 | 62.0 | **62.86** | **+0.86pt** | 73.03 |
| RoboSpatial / compatibility | 41.0% | 42.9% | +1.9pt | — |
| RoboSpatial / configuration | 86.2% | 86.2% | 0 | — |
| RoboSpatial / context | 55.7% | 56.6% | +0.9pt | — |

两个平台均 101/101、350/350 全 ok，零丢题、零 unmatched。

## 3. Q-Spatial +8.9pt 的翻转分析（逐题配对，n=101）

| 项 | 值 |
| --- | --- |
| A800 成功 / K100 失败 | 15 题 |
| K100 成功 / A800 失败 | 6 题 |
| **跨阈值翻转（一侧 ratio<2、另一侧 ≥2，即 ratio≈2 边界）** | **19/101 题** |
| ratio 中位数 | K100 1.72 / A800 1.59 |
| ratio 均值 | K100 2.57 / A800 2.35 |

**判读**：净差 9 题中的绝大部分来自 **ratio≈2 判定边界上的翻转**（19 题处在线界附近）。
Q-Spatial 是"长链自由推理 + 数值估计 + 硬阈值"任务：两条各自数百 token 的推理链在
bf16 下只要有极微小分歧（批组成归约顺序、vLLM/torch 版本差 0.15.1-dcu vs 上游、
2.9.0 vs 2.9.1），末位数值估计就会越过 ratio=2 边界。**RoboSpatial 同条件仅差 0.86pt、
且 A800 侧表现出完全相同的保守偏置（compat 41.0 vs 42.9），不支持"K100 存在系统性
算子错误"**；两侧单侧各只跑了一次，该差亦含单次运行噪声。结论方向不变：
论文 73.53 的缺口两侧都存在（A800 仍 −4.2pt），属 bench 配置/模型侧。

## 4. 结论

1. **RoboSpatial：跨硬件一致**（Δ0.86pt，逐子项同向）。"62 vs 73.03"的缺口在
   NVIDIA 上同样存在，且保守偏置（GT=Yes 答 no）两侧一致 → **bench 配置差异**
   （官方 harness 提供深度输入通道；喂深度后 compat +13pt，见 `reports/20` §2.3 诊断）。
2. **Q-Spatial：存在 +8.9pt 跨硬件差，但机制是边界翻转而非系统性偏差**；
   与论文的缺口两侧也都存在（K100 −13.1pt，A800 −4.2pt）→ 主因仍是
   **提示引导/模型估计质量**（官方模板间差 21.8pt 的变体实验见 `reports/20` §2.4），
   跨硬件差单列为待复测项（两侧各需同配置重复运行以剥离单次噪声）。
3. **K100 推理保真度**：本组证据 + CV-Bench 的 A800↔DCU 对照（99.6% 逐题一致、
   Δ0.1pt，`reports/19_a800_cross_validation.md`）共同支持——**K100 vLLM 适配
   未发现算子或运行时级别的精度缺陷**；各基准与论文的偏差均有 bench/模型侧归因。

## 5. 复现命令

```bash
# A800（GPU 节点 gpu01，conda env wsad_robin）：
cd /gpudata/wsad/working_space/zsy/robin_verify
tar xzf extbench_a800.tgz -C extbench   # 包由 DCU 侧打出，md5 校验一致
CUDA_VISIBLE_DEVICES=0 bash run_extbench_a800.sh   # 服务 + Q-Spatial + RoboSpatial
# K100 侧原始结果：outputs/bench_public_noneager2/{qspatial,robospatial}/summary.json
```
