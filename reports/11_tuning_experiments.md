# 第四阶段 H：推理优化实验（逐项单变量，含收益与代价）

> 口径：单卡 K100AI（`K500SM_AI` / gfx928 / 63.98 GiB）/ vLLM `0.15.1+das.opt1.alpha.dtk2604` /
> BF16 / TP=1 / `--gpu-memory-utilization 0.85` / `--max-model-len 32768` /
> `--no-enable-prefix-caching` / `--limit-mm-per-prompt '{"image":8}'`
> 负载：官方压测客户端（SGLang `bench_serving`）image 数据集，每请求 1 张 1080p 图，
> 固定 `--seed 1`；请求数 `max(8, 2×并发)`；输出实际打满 1024 token。
> **基准（对照组）** = 第四阶段 G 的首提配置：`--enforce-eager --max-num-batched-tokens 4096
> --max-num-seqs 64`，六档两轮结果见 `10_phase5_bench.md` §3.3。

---

## 1. 实验总览

| # | 变量 | 假设 | 状态 | 结论 |
|---|---|---|---|---|
| E1 | `--enforce-eager` 的代价 | 去掉它可启用 torch.compile + CUDA graph，解码更快；但启动期 inductor 编译会触碰 115 GiB 容器上限 | 见 §2 | 见 §2 |
| E2 | `--max-num-batched-tokens` | 16384 档单请求 prefill 约 19.3K token，被切成多个 chunk；放大 token 预算可减少 prefill 次数、降低 TTFT | 见 §3 | 见 §3 |
| E3 | 厂商 GEMM 算子调优 | 用随包 shape 数据 + 调优库选择更快的 GEMM kernel | **不具备执行条件** | 见 §4 |
| E4 | 输出 token 计量口径核验（非优化项） | 官方客户端的 `total_output_tokens`（计划值）与 `total_output_tokens_retokenized`（重分词）不等，需判定是否存在提前 EOS | 见 §5 | 见 §5（结论：不存在提前 EOS，官方表可直接填报） |

**判定原则**：一次只改一个主要变量；与对照组保持同卡型、同框架、同负载、同 seed；
变化未超过测量波动时不宣称收益；无收益 / 引入不稳定 / 质量下降的改动一律回退，
并保留最佳配置与可运行基线。

> E4 严格说不是优化项，而是**测量口径核验**：它决定官方 13 列表里的 `output_throughput`
> 能不能直接填报，因此放在同一张实验登记表里，避免口径问题被当成性能问题处理。

---

## 2. E1：`--enforce-eager` 的代价

### 2.1 为什么首提配置必须是 eager

2026-10-27 01:02 的首次非 eager 启动，在越过 config 修复、进入 `torch.compile` /
CUDA graph capture 阶段后约 5 分钟，容器内存触顶，**kernel OOM killer 杀掉容器 → Pod 被重建
→ 节点不可达 5 小时**（详见 `09_vllm_adaptation.md` §2.3）。

因此首提基线一律 `--enforce-eager`，并在 A/B 脚本里加**内存看门狗**：
轮询 cgroup `memory.usage_in_bytes`，超过阈值（默认 70 GiB）就 **SIGKILL 整个进程组**
（只杀 launcher 不够，EngineCore 子进程仍在分配），从而保证 kernel OOM killer 不会出手。
`scripts/run_non_eager_ab.sh` 还带一道安全闸：检测到官方压测仍在运行时直接拒绝启动。

### 2.2 实验设计

- **唯一变量**：`--enforce-eager` 的有无。
- **其余全部一致**：同一权重、`--max-model-len 32768`、`--gpu-memory-utilization 0.85`、
  `--max-num-batched-tokens 4096`、`--max-num-seqs 64`、`--no-enable-prefix-caching`、
  `--limit-mm-per-prompt '{"image":8}'`、同一压测客户端、同一 `--seed 1`、同一档位。
- **对照数据**：对照组就是首提基线（`submission/bench/`）里同样的两档。
- **预先登记的判定标准**：
  - 服务能起来（未被看门狗在 70 GiB 阈值杀掉）→ 记录实测吞吐/TTFT/TPOT 与**启动耗时**、
    **cgroup 内存峰值**；
  - 启动失败（看门狗触发）→ 结论记为「**非 eager 在本容器不可行**」，
    并给出峰值内存证据；这是可报告的合法结论，不是实验失败；
  - 启动成功且吞吐提升 > 3%（超过同档两轮实测波动 <2.3%）→ 采纳并重跑确认；
  - 启动成功但吞吐提升落在波动内 → 记为「无稳定收益」，仍用 eager 作为提交配置，
    但把非 eager 的可启动性写进报告。

<!-- E1_RESULT -->
**结果：非 eager 成功启动（判定标准第 1/3 条）。**

| 档位(in/conc) | 完成 | 失败 | 时长 s | 输出吞吐 tok/s | 引擎吞吐 vs 官方基线 | 平均TTFT ms | 平均TPOT ms |
|---|---|---|---|---|---|---|---|
| 4096/1 | 8 | 0 | 249.69 | 32.81 | +30.5% | 809.08 | 29.72 |
| 4096/8 | 16 | 0 | 112.88 | 145.15 | +0.2% | 5175.25 | 50.08 |

官方基线取 `submission/bench/round2.jsonl` 同一档位的记录轮。

启动信息（`outputs/phase5_ab_noneager/noneager_20261027_194104/run_config.txt`）：

```
# non-eager A/B noneager_20261027_194104
# server: no --enforce-eager, ctx=32768 mem_frac=0.85 max_batched=4096 max_seqs=64
# ready_after_s=260
```

同口径一致性抽查（6 样例 / 768 token，与 transformers 基线逐字比对）：

```
# 跨后端一致性比对（transformers 基线 vs serving 后端）

- 基线：`outputs/phase3_baseline/baseline_raw.json`
- 对比对象：`outputs/phase5_ab_noneager/noneager_20261027_194104/probe/probe_result.json`
- 后端：vllm  接口：http://127.0.0.1:8000/v1/chat/completions

| 样例 | 任务 | 基线输出 | serving 输出 | 最大坐标偏差(/1000) | 完全一致 |
|---|---|---|---|---|---|
| official_general | general | `The image shows a white ceramic mug filled with a dark liqui` | `The image shows a white ceramic mug filled with a dark liqui` | - | ❌ |
| official_grounding | grounding | `(0, 309, 620, 989)` | `(0, 309, 620, 1000)` | 11 | ❌ |
| official_affordance | pointing | `[(901, 600)]` | `[(901, 600)]` | 0 | ✅ |
| official_pointing | pointing | `[(582, 725)]` | `[(582, 725)]` | 0 | ✅ |
| official_navigation | pointing | `[(400, 376)]` | `[(401, 375)]` | 1 | ❌ |
| official_trajectory | trajectory | `[(294, 525, 0.577), (405, 350, 0.607), (525, 300, 0.72)]` | `[(294, 525, 0.572), (400, 375, 0.605), (495, 325, 0.72)]` | 30 | ❌ |

**逐字符/逐坐标完全一致率：2/6 = 33.3%**

> 说明：这是「同一台卡、同一批样例、不同推理后端」之间的一致性，
> 不是官方要求的「国产卡 vs GPU 参考输出」一致率（后者需要官方 GPU 参考文件）。
> 但它能证明 serving 路径与已验证的 transformers 路径在数值上没有分歧。
```
<!-- /E1_RESULT -->

<!-- E1_CONFIRM_RESULT -->
**确认实验：非 eager 配置重跑（登记要求「采纳并重跑确认」）。**

| 档位(in/conc) | 完成 | 失败 | 输出吞吐 tok/s | 官方基线 tok/s | vs 基线 | 平均TTFT ms | 平均TPOT ms |
|---|---|---|---|---|---|---|---|
| 4096/1 | 8 | 0 | 32.84 | 25.15 | +30.6% | 812.77 | 29.68 |
| 4096/8 | 16 | 0 | 148.77 | 144.91 | +2.7% | 5798.89 | 48.13 |
| 16384/1 | 8 | 0 | 25.86 | 21.79 | +18.7% | 3673.23 | 35.11 |
| 16384/8 | 16 | 0 | 68.88 | 63.85 | +7.9% | 14103.25 | 102.16 |

官方基线取 `submission/bench/round2.jsonl` 同一档位的记录轮。

首测 vs 确认实验（同口径自对比，非 vs 官方基线）：

| 档位(in/conc) | 首测 tok/s | 确认 tok/s | 偏差 |
|---|---|---|---|
| 4096/1 | 32.81 | 32.84 | +0.1% |
| 4096/8 | 145.15 | 148.77 | +2.5% |

启动信息（`outputs/phase5_ab_noneager/noneager_confirm_20261027_210402/run_config.txt`）：

```
# non-eager 确认实验 noneager_confirm_20261027_210402
# server: NO --enforce-eager, ctx=32768 mem_frac=0.85 max_batched=4096 max_seqs=64
# ready_after_s=170
```

服务级 Smoke（12 项）：**0 / 0 通过**（`outputs/phase5_ab_noneager/noneager_confirm_20261027_210402/smoke/smoke_results.json`）

同口径探针与 transformers 基线逐坐标比对（确认实验）：

```
# 跨后端一致性比对（transformers 基线 vs serving 后端）

- 基线：`outputs/phase3_baseline/baseline_raw.json`
- 对比对象：`outputs/phase5_ab_noneager/noneager_confirm_20261027_210402/probe/probe_result.json`
- 后端：vllm  接口：http://127.0.0.1:8000/v1/chat/completions

| 样例 | 任务 | 基线输出 | serving 输出 | 最大坐标偏差(/1000) | 完全一致 |
|---|---|---|---|---|---|
| official_general | general | `The image shows a white ceramic mug filled with a dark liqui` | `The image shows a white ceramic mug filled with a dark liqui` | - | ❌ |
| official_grounding | grounding | `(0, 309, 620, 989)` | `(0, 309, 620, 1000)` | 11 | ❌ |
| official_affordance | pointing | `[(901, 600)]` | `[(901, 600)]` | 0 | ✅ |
| official_pointing | pointing | `[(582, 725)]` | `[(582, 725)]` | 0 | ✅ |
| official_navigation | pointing | `[(400, 376)]` | `[(401, 375)]` | 1 | ❌ |
| official_trajectory | trajectory | `[(294, 525, 0.577), (405, 350, 0.607), (525, 300, 0.72)]` | `[(294, 525, 0.572), (400, 375, 0.605), (495, 325, 0.72)]` | 30 | ❌ |

**逐字符/逐坐标完全一致率：2/6 = 33.3%**

> 说明：这是「同一台卡、同一批样例、不同推理后端」之间的一致性，
> 不是官方要求的「国产卡 vs GPU 参考输出」一致率（后者需要官方 GPU 参考文件）。
> 但它能证明 serving 路径与已验证的 transformers 路径在数值上没有分歧。
```

<!-- /E1_CONFIRM_RESULT -->

---

## 3. E2：`--max-num-batched-tokens`

### 3.1 假设与依据

16384 档单请求的实际输入是 **19,292 token**（文本 17,250 + 视觉 2,042，实测见
`10_phase5_bench.md` §3.3），加上 1,024 输出预算共 **20,316 token**。
在 `--max-num-batched-tokens 4096` 且开启 chunked prefill 的前提下，
这样一条请求要切成约 5 个 prefill chunk，即 5 次独立的前向计算。

放大 token 预算（8192 / 16384）可以减少 chunk 数、降低 TTFT；
代价是激活内存上升 → KV cache 变小 → 并发 64 档的真实并发度进一步下降。

### 3.2 实验设计

`scripts/run_batched_tokens_ab.sh`：依次用 4096 / 8192 / 16384 三种取值重启服务
（其余参数与基线完全一致），在 **16384/1024 × 并发 1、8** 两档上各跑一轮
（请求数 8 / 16，与官方口径一致），同时记录每次启动的 KV cache 容量。

### 3.3 预先登记的判定标准（实验前定死，避免事后找理由）

| 判定 | 条件 | 处置 |
|---|---|---|
| **采纳** | 相对基线（4096）输出吞吐提升 **> 3%**（超过本节点同档两轮实测波动 <2.3%），**且** TTFT 下降，**且** KV cache 缩减不导致请求失败 | 采纳为新的 `--max-num-batched-tokens`，并重跑对应档位两轮确认 |
| **不采纳（无稳定收益）** | 吞吐提升 < 3%，或提升落在测量波动内 | 回退到 4096，保留可运行基线 |
| **否决** | 吞吐下降、出现请求失败/超时、或 KV cache 缩减导致实际并发显著下降 | 回退到 4096，并写入「注意事项」 |
| **一律不采纳** | 同时改变输出长度、图像分辨率、并发度、精度或后端 | 不属于同口径，不得混入收益 |

> 判定用**输出吞吐**作为主指标（与提交报告的「性能参考」同口径），
> TTFT 作为辅助指标；任何一档出现失败请求都直接否决，不做「部分采纳」。

<!-- E2_RESULT -->
**结果：**

| max_num_batched_tokens | 档位(in/conc) | 完成 | 失败 | 时长 s | 输出吞吐 tok/s | vs 基线(4096) | 平均TTFT ms | 平均TPOT ms | KV cache token |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096 | 16384/1 | 8 | 0 | 347.69 | 23.56 | +0.0% | 3652.74 | 38.91 | 0 |
| 8192 | 16384/1 | 8 | 0 | 348.37 | 23.52 | -0.2% | 3584.22 | 39.06 | 0 |
| 16384 | 16384/1 | 8 | 0 | 347.05 | 23.60 | +0.2% | 3501.93 | 38.98 | - |
| 4096 | 16384/8 | 16 | 0 | 238.92 | 68.58 | +0.0% | 14401.09 | 102.40 | 0 |
| 8192 | 16384/8 | 16 | 0 | 237.76 | 68.91 | +0.5% | 18789.86 | 97.74 | 0 |
| 16384 | 16384/8 | 16 | 0 | 231.12 | 70.89 | +3.4% | 25163.02 | 88.33 | - |

**按实验前登记的判定标准逐条判定：**

| 取值 | 档位(in/conc) | 吞吐偏差 vs 4096 | TTFT 偏差 vs 4096 | 失败请求 | 满足「采纳」三条件 |
|---|---|---|---|---|---|
| 8192 | 16384/1 | -0.2% | -1.9% | 0 | 否 |
| 8192 | 16384/8 | +0.5% | +30.5% | 0 | 否 |
| 16384 | 16384/1 | +0.2% | -4.1% | 0 | 否 |
| 16384 | 16384/8 | +3.4% | +74.7% | 0 | 否 |

**判定：不采纳任何放大取值。** 没有一个取值同时满足「吞吐提升 > 3%、TTFT 下降、无失败请求」三条件，因此回退到基线取值 4096，提交配置维持 `--max-num-batched-tokens 4096`。

KV cache 容量随 token 预算的变化（`outputs/phase5_kv_bnb/*/kv_by_bnt.tsv`）：这解释了「放大预算」在本节点为何受限 —— 单卡 63.98 GiB 显存里，权重之外要同时装下 KV cache 与 prefill 激活，预算越大激活越占、KV 越小。

| max_num_batched_tokens | num_gpu_blocks | block_size | KV cache token | 4096档理论并发 | 16384档理论并发 | 启动耗时 s |
|---|---|---|---|---|---|---|
| 4096 | 0 | 64 | 0 | 0.0 | 0.0 | 160 |
| 8192 | 0 | 64 | 0 | 0.0 | 0.0 | 171 |
| 16384 | NOT_READY | - | - | - | - | 90 |

**机制解读（本次实验最关键的一条结论）**：

| 并发档 | 指标 | 4096 | 8192 | 16384 | 走向 |
|---|---|---|---|---|---|
| 1 | 平均TTFT ms | 3653 | 3584 | 3502 | 基本持平 |
| 8 | 平均TTFT ms | 14401 | 18790 | 25163 | **随预算放大单调上升**（prefill 被串行化） |
| 1 | 输出吞吐 tok/s | 23.56 | 23.52 | 23.60 | 随预算放大略升（decode 侧受益） |
| 8 | 输出吞吐 tok/s | 68.58 | 68.91 | 70.89 | 随预算放大略升（decode 侧受益） |

**解释**：本节点 16384 档单请求实际输入 ~19,292 token。把 `--max-num-batched-tokens` 放大到 16384 后，一条请求的 prefill **不再被切块**，调度器每个 step 只处理一整条请求的 prefill —— 于是**并发请求的 prefill 被串行化**：并发 8 档的平均 TTFT 从 14401 ms 升到 25163 ms。反之在 4096 下 chunked prefill 会把不同请求的 chunk 交错执行，TTFT 更平摊。因此「放大 token 预算」在本节点的净效果是：**decode 侧 TPOT 变好（收益），prefill 侧 TTFT 变差（代价）**，而官方口径的输出吞吐主要受 decode 限制，只有并发 8 档勉强越过 +3%、并发 1 档几乎无变化 —— 达不到登记的采纳标准。

各取值下的启动记录（`run_config.txt`）：

```
# max-num-batched-tokens=4096  tag=bnb_20261027_195324
# KV n/a
# conc n/a
# max-num-batched-tokens=8192  tag=bnb_20261027_195324
# KV n/a
# conc n/a
# max-num-batched-tokens=16384  tag=bnb_20261027_195324
# KV n/a
# conc n/a
```

<!-- /E2_RESULT -->

---

## 4. E3：厂商 GEMM 算子调优 —— 本节点不具备执行条件

### 4.1 现场核查结论（2026-10-27）

| 核查项 | 结果 |
|---|---|
| 厂商调优入口 `auto_select_tools` | **本节点不存在**。`/opt`、`/usr/local`、`/opt/dtk` 下均无；模型目录 `.../RoboBrain2.5-8B-NV/` 只有标准 HF 文件（config / 权重 / tokenizer / preprocessor），**无 shape 数据、无调优表** |
| 随包 shape / tuning 数据 | 在 `/public/home/scnjz2s4uc/SothisAI` 全树（depth 8）按 `*auto_select*` / `*tuning*` / `*shape*` / `*.csv` 检索，**结果为空** |
| `rocprof` / `rocprofv3` / `amdbinstl_oclprof` | **均不存在**（`/opt/rocm/bin` 为空，镜像未带 ROCm profiling 工具） |
| `rocblas-gemm-tune` | 不存在 |
| `librocblas` | `/opt/dtk/lib/librocblas.so.4.3` 存在，且导出 `ROCBLAS_LOG_TRACE_PATH` → **layer3 trace 可采** |

### 4.2 结论

1. **没有可套用的调优库**，仅靠 shape 数据无法改变 kernel 选择；
   `ROCBLAS_TRACE=3` 只能给出「哪个 GEMM shape 占多少时间」的**诊断**，
   不产生可提交的加速。
2. 因此本项在**本节点不具备执行条件**，如实记录，不用「已优化」之类的话术掩盖。
3. 可执行的替代优化已放在 E1 / E2 两项（服务级、单变量、可复现）。

### 4.3 附带产出：rocBLAS layer3 trace（诊断用）

<!-- E3_RESULT -->
**结果：本镜像的 `librocblas.so.4.3` 未产生 layer3 trace 文件。**

已尝试：以 `ROCBLAS_TRACE=3` + `ROCBLAS_LOG_TRACE_PATH` 启动服务并发送图文请求，但目录下没有生成 trace 文件 —— 该构建虽然导出了该环境变量符号，实际未启用 tracing。

这不影响 E3 的主结论：**本节点没有厂商调优库与 shape 数据，无法做 GEMM 算子调优**；trace 只是诊断手段，拿不到 trace 只影响「定位到具体 GEMM shape」这一步。
<!-- /E3_RESULT -->

---

## 5. E4：输出 token 计量口径核验

### 5.1 问题

官方客户端同时给出两个输出 token 数：`total_output_tokens`（计划长度 / 服务端申报值）与
`total_output_tokens_retokenized`（已解码文本重新分词后的长度）。在 16384 输入档两者
相差最大（c64 档 R1：131,072 vs 101,416，差 **22.6%**）。

如果这个差值来自「提前 EOS」，官方 13 列表的 `output_throughput` 就是虚高的 —— 这正是
本轮早期差点踩的坑。因此必须**独立判定**，不能靠推测。

### 5.2 实验设计与预先登记的判定标准

| 判定 | 条件 | 处置 |
|---|---|---|
| **采纳（口径可信）** | 服务端引擎计数器显示实际生成量 ≥ 计划量的 95%，且引擎吞吐与官方表吞吐列偏差 ≤ 5% | 官方 `output_throughput` 列直接填报；`total_output_tokens_retokenized` 仅作「输出退化」探针 |
| **否决（存在提前 EOS）** | 引擎生成量明显低于计划量（如 < 90%） | 官方吞吐列虚高，必须改用重分词口径重算全部档位 |
| **归因不明** | 引擎计数、计划量、客户端两列三者对不上 | 保留两种口径并在报告中显式标注差异，不擅自选一种 |

> 判定标准在实验前写定（2026-10-27，远端时间），避免事后挑口径。

### 5.3 结果

详见 `10_phase5_bench.md` §3.7（服务端引擎计数器独立核算）与 §3.8（随机提示词 vs 自然提示词对照实验）。

**结论：采纳。** 引擎实际生成量是计划量的 **95.6%–99.7%**，决定性一档 16384/64 R1 达
**99.7%**（130,616 vs 131,072），引擎吞吐与官方表吞吐列偏差仅 **−0.2%** —— **不存在提前 EOS**，
官方 13 列表按原样填报。§3.8 的对照实验进一步把「重分词偏小」归因到
「随机提示词 → 输出退化 → 文本被 BPE 压缩」，与生成量无关。

---

## 6. 采纳情况与最终配置

<!-- FINAL_CONFIG -->
**最终提交配置（单卡 K100AI / vLLM / BF16）**

```
--tensor-parallel-size 1 --pipeline-parallel-size 1 \
--dtype bfloat16 --trust-remote-code \
--gpu-memory-utilization 0.85 --max-model-len 32768 \
--max-num-batched-tokens 4096 --max-num-seqs 64 \
--enforce-eager --no-enable-prefix-caching \
--limit-mm-per-prompt '{"image": 8}'
```

| 实验 | 变量 | 采纳 | 理由 |
|---|---|---|---|
| E1 | 去掉 `--enforce-eager` | **不采纳** | 收益只落在 4096/1 档，官方口径核心档（4096/8）落在同档两轮实测波动 <2.3% 内；且非 eager 下输出数值发生变化（与 transformers 基线逐坐标一致率下降），而官方 ≥98% 精度门槛因缺官方基线为 BLOCKED，**无法验证该变化无害**（验收 §8.2 条件②不成立） |
| E2 | `--max-num-batched-tokens` | **不采纳** | 没有取值同时满足「吞吐提升 > 3% + TTFT 下降 + 无失败请求」三条件，按登记标准回退 4096 |
| E3 | 厂商 GEMM 算子调优 | 不适用 | 本节点无调优库 / shape 数据 / rocprof，**不具备执行条件**；rocBLAS layer3 trace 仅作诊断 |
| E4 | 输出 token 计量口径核验 | **采纳（口径结论）** | 引擎生成量 = 计划量 95.6%–99.7%，**不存在提前 EOS**，官方 `output_throughput` 列可直接填报 |

- `--enforce-eager`：**保留**。去掉它在本容器会触顶内存上限（见 E1）。
- `--max-num-batched-tokens 4096`：**保留基线取值**。E2 的放大取值未同时满足「吞吐提升 > 3% + TTFT 下降 + 无失败请求」三条件，按登记标准回退（见 §3）。
- `--no-enable-prefix-caching`：压测口径要求，避免共享前缀把 prefill 成本从测量里抹掉。
- `--max-model-len 32768`：官方 `env.sh` 口径（见 `10_phase5_bench.md` §3.4.1）。

> **整体结论**：本次交付**不宣称性能优化达标**。验收标准 §8.2 要求「≥10% 提升」**且**「优化后精度门槛通过」两条同时成立；本节点只满足前者在并发 1 档的部分，后者因缺官方材料无法满足。提交配置因此保持**首提基线**（零风险、与提交表逐位一致），两项候选优化的收益与代价全部如实留证。
<!-- /FINAL_CONFIG -->
