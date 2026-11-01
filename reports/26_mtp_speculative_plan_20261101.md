# MTP / 投机解码路线评估与实验计划（RoboBrain2.5-8B-NV · K100AI）

> 阶段：E2 结束后的 **E3 方案设计**（本轮未启动任何实验、未改服务配置）
> 日期：2026-11-01
> 目标硬件：K100AI 单卡（运行时 `K500SM_AI` / gfx928 / 63.98 GiB）
> 后端：vLLM `0.15.1+das.opt1.alpha.dtk2604.torch290.2605261031.gdb9e4b` / BF16 / TP=1
> 依据：`agent_skill/tuning_skills/hygon-dcu-operator-tuning/SKILL.md`、
> `agent_skill/tuning_skills/references/experiment-template.md`、
> `agent_work/RoboBrain2.5-8B-NV_验收标准.md`
>
> **本轮全部为只读勘察 + CPU 离线仿真**。运行中的 `fullbench_candidate_20261101_114614`
> 压测未被中断：期间仅读取产物文件、跑 CPU 侧 numba 仿真（`nice -n 10`），
> 未启停服务、未改配置、未占 GPU。

---

## 1. 结论先行

| 问题 | 结论 |
| --- | --- |
| 这个 checkpoint 能用原生 MTP 吗？ | **不能**。`config.json` 无 `num_nextn_predict_layers`，权重索引 750 个 tensor 中 MTP 类为 **NONE**，最大 layer index = 35（36 层，无 nextn 层）。vLLM 的 `method="mtp"` 只认 `deepseek_mtp / qwen3_next_mtp / glm4_moe_mtp / ernie_mtp …` 这些自带 nextn 头的模型族，Qwen3-VL 不在其中 |
| `BAAI/RoboBrain2.5-8B-MT` 是 MTP 版吗？ | **不是**。官方技术报告原文：「同系列另有 BAAI/RoboBrain2.5-8B-MT（架构与训练数据相同、性能相近，**区别仅在训练集群：NV 为 NVIDIA 集群，MT 为摩尔线程集群**）」。MT = Moore Threads。该仓库在 HF 上为 gated（401），且架构与 NV 相同，同样无 MTP 头 |
| 那 MTP 思路还能落地吗？ | **能，但要换成等价手段：投机解码（speculative decoding）**。本机 vLLM 0.15.1 支持 `ngram` / `suffix` / `draft_model` / `medusa` / `mlp_speculator` / `eagle` 等方法 |
| 值得做吗？ | **分场景**。实测接受率：真实任务输出族接受率仅 0.28–0.56（低于 60% 盈亏线，**大概率亏**）；但**官方压测负载族（gibberish 随机 token 提示）接受率高达 0.87–0.93，tokens/forward 可达 2.24–5.40**——而赛事六档压测用的正是该负载 |
| 最大的风险 | **不是收益，是精度不可验证**。官方 ≥98% 一致率所需 GPU 参考输出与判分器至今 **BLOCKED**，无法证明开投机解码后输出仍逐 token 一致 |

---

## 2. 现场事实（全部为本次实测，非推断）

### 2.1 模型侧

```
config.json  architectures = ["Qwen3VLForConditionalGeneration"], model_type = qwen3_vl
             text_config: 36 层 / hidden 4096 / 32 头 / 8 KV 头 / vocab 151936 / BF16
             vision_config: 27 层 / hidden 1152 / deepstack_visual_indexes [8,16,24]
model.safetensors.index.json: total_tensors = 750
             mtp-like tensors = NONE     max_layer_idx = 35
```

即：**没有 nextn / MTP 投影头**，与 Qwen3-VL 8B 原版结构一致。权重分片 2 个、约 17.5 GB。

`BAAI/RoboBrain2.5-8B-MT` 的 `config.json` 与 NV 同架构（`Qwen3VLForConditionalGeneration`，
hidden_size 2560，4B 规模），HF 上 gated（`config -> 401`），且 NV 本体匿名访问同样 401
（本地权重来自 SCNet AIHub），4B 匿名可访问（`config -> 200`）。

### 2.2 后端侧（vLLM 0.15.1 + DTK 26.04）

`vllm/config/speculative.py` 实际支持的 `method`：

```
ngram | medusa | mlp_speculator | draft_model | suffix
deepseek_mtp | mimo_mtp | glm4_moe_mtp | glm4_moe_lite_mtp | ernie_mtp
exaone_moe_mtp | qwen3_next_mtp | qwen3_5_mtp | longcat_flash_mtp | mtp
pangu_ultra_moe_mtp | step3p5_mtp | eagle | eagle3
```

配置解析实测（CPU-only，不起服务）：

| 输入 | 解析结果 |
| --- | --- |
| `{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":3,"prompt_lookup_max":8}` | OK，min=3 max=8 |
| `{"method":"ngram","num_speculative_tokens":8,"prompt_lookup_min":2,"prompt_lookup_max":8}` | OK，min=2 max=8 |
| `{"method":"ngram","num_speculative_tokens":4}` | OK，min=max=5（默认） |
| `{"method":"suffix","num_speculative_tokens":4}` | **FAIL**：`ImportError: Arctic Inference is required for suffix decoding`（`arctic-inference==0.1.1` 未装；清华源可达，`pypi-arctic 200`） |

### 2.3 注意力后端闸门（决定性）

当前最优配置使用 `VLLM_ATTENTION_BACKEND=ROCM_AITER_FA`。该后端
（`vllm/v1/attention/backends/rocm_aiter_fa.py`）：

- 全文**没有任何** `is_spec_decode` / `draft` 引用（grep 只命中 `kv_cache_spec`）；
- decode 走 `torch.ops.aiter.paged_attention_v1`；
- 滑动窗分支硬编码 `max_seqlen_q=1,  # optimize this`（第 1130 行）。

对照上游 `vllm/v1/attention/backends/flash_attn.py`：显式处理 spec-decode
（`fast_build` 注释「used when there will be few iterations i.e. spec-decode」、
`max_query_len` 分支、`cu_seqlens_q`/`cu_seqlens_k` 分离）；
官方 `rocm_attn.py` decode 走 `chunked_prefill_paged_decode`，**用 varlen 语义**，
天然支持每请求多 query token。

而 `vllm/v1/attention/backends/utils.py::split_decodes_prefills_and_extends` 的
`decode_threshold` 取自后端的 `reorder_batch_threshold`：

- aiter FA：`reorder_batch_threshold = 1`
- 投机解码时每请求 query 数为 `1 + num_speculative_tokens`（>1），会被判定为
  **prefill/extend** 而非 decode，走 `extend_forward` → `aiter.flash_attn_varlen_func`
  （`cu_seqlens_q == cu_seqlens_k`），语义上等价于「把草稿 token 当成一次 extend 预填充」。

**风险判定**：这条路径**可能正确**，但**未经证实**。aiter FA 的 extend 分支用
`cu_seqlens_k=cu_seqlens_q`，而投机解码校验要求 KV 长度 = 已缓存长度 + 草稿数，
两者对齐方式不同。这是**必须实测验证**的点，不能靠读码断定。

### 2.4 采样口径与精度可比性（已确认，对本方案有利）

官方客户端 `sglang/bench_serving.py` 的请求体默认值：

```python
payload["temperature"] = 0.0        # greedy
payload["ignore_eos"]  = not args.disable_ignore_eos   # 本项目用 --disable-ignore-eos → False
payload["max_tokens"]  = request_func_input.output_len
```

vLLM greedy 下的投机解码接受判定走 `rejection_greedy_sample_kernel`
（`vllm/v1/sample/rejection_sampler.py:384-397, 642+`）：**以 target argmax 为准**，
草稿 token 只影响「这一步多吐几个 token」，不接受就丢弃并回退到 argmax。
因此 **greedy 下最终 token 序列理论上与不开投机解码一致**——这是本方案精度可控的关键依据。

**但**：官方 ≥98% 一致率的判分器与 GPU 参考输出尚未下发（`BLOCKED`），
「理论一致」不能当作「已验证一致」，仍需自建对照实测。

---

## 3. 离线接受率仿真（决定性证据，零 GPU）

脚本：`agent_work/tools/spec_accept_sim.py`（远程 `/root/private_data/workspace/robo/tools/spec_accept_sim.py`）
产物：`agent_work/evidence/spec_accept_sim.json`（远程 `outputs/spec_accept_sim.json`）

方法：用 **vLLM 自己那份 numba 提案器内核**
`_find_longest_matched_ngram_and_propose_tokens`（`vllm/v1/spec_decode/ngram_proposer.py`）
对**已保存的真实生成文本**逐位置重放：历史 = prompt(+image token) + 已生成前缀，
每步取 draft，与真实续写逐 token 比对算接受数。语义与线上完全一致，不是估计。

### 3.1 结果

| 负载族 | ngram(min,max) | k | 命中率 | 接受率(占草稿) | tokens/forward |
| --- | --- | ---: | ---: | ---: | ---: |
| tasks/with-prompt（真实任务输出） | (5,5) | 4 | 0.215 | 0.479 | 1.413 |
| tasks/with-prompt | (3,8) | 4 | 0.405 | 0.430 | 1.696 |
| tasks/with-prompt | (2,8) | 4 | 0.577 | 0.383 | 1.884 |
| tasks/with-prompt | (3,8) | 16 | 0.405 | 0.172 | 2.105 |
| tasks/gen-only（不含 prompt） | (3,8) | 2 | 0.355 | 0.551 | 1.391 |
| **bench_gibberish（官方压测负载族）** | (5,5) | 4 | 0.333 | **0.931** | **2.242** |
| **bench_gibberish** | (3,8) | 4 | 0.375 | 0.897 | 2.346 |
| **bench_gibberish** | (2,8) | 4 | 0.408 | 0.865 | 2.413 |
| **bench_gibberish** | (3,8) | 8 | 0.375 | 0.833 | **3.500** |
| **bench_gibberish** | (3,8) | 16 | 0.375 | 0.749 | **5.400** |

### 3.2 解读

1. **真实任务输出接受率只有 0.28–0.56**，且 k 越大接受率越低。
   按既有工程经验「接受率 p < 60% 时任何草稿长度都亏」，
   **对真实任务流量开 n-gram 投机解码，大概率是负收益**。
2. **官方六档压测负载接受率 0.87–0.93，tokens/forward 可达 2.2–5.4**。
   原因已定位：`outputs/phase5_token_accounting/result.json` 里
   `gibberish_random_tokens` 族 `repeat_ratio_40gram = 0.5726`，
   `text_head`/`text_tail` 显示模型输出**在自我重复同一段描述**
   （"The image depicts a white ceramic mug…" 反复出现）；
   而 `natural_prose` 族 `repeat_ratio_40gram = 0.0`。
   投机解码正是靠这种重复获利。
3. **因此收益高度依赖负载**：压测榜（gibberish 提示）会显著受益，
   真实业务流量（自然提示、多样化输出）基本无收益甚至亏损。
   这是本方案**最大的价值判断点**，不是技术难点。
4. k 的最优值也依赖负载：真实任务 k=2 最优（接受率 0.551）；
   压测负载 k=16 时 tokens/forward 最高（5.40）。
   说明 **k 需要按负载分别标定，不能一个值通吃**。

---

## 4. 候选方案与取舍

| 方案 | 可行性 | 精度风险 | 预期收益 | 结论 |
| --- | --- | --- | --- | --- |
| **A. 原生 MTP** | ✗ 无 nextn 头 | — | — | **不可行**，排除 |
| **B. n-gram 自投机**（`method=ngram`） | ✓ 配置已验证可解析 | 低（greedy 下 argmax 保序） | 压测负载高、真实流量低 | **主方案**，先做 |
| **C. suffix decoding** | ✗ 缺 `arctic-inference`；清华源可达可装 | 低 | 理论优于 n-gram（树状） | **备选**，装依赖后对比 |
| **D. 4B 草稿模型**（`method=draft_model`） | ✓ 4B 匿名可下载；tokenizer 与 NV 同为 Qwen2Tokenizer/vocab 151936 | 中（多一个模型常驻显存，吃 KV cache 预算） | 接受率更高但草稿本身耗显存与算力 | **备选**，需算显存账 |
| **E. EAGLE/Medusa** | ✗ 需训练 head，超出推理调优范围 | — | — | 超出授权范围，排除 |
| **F. SGLang 侧 DSPARK** | ✗ 赛事材料里的 DSpark 是 DeepSeek-V4/MiniCPM-V-4.7 的方案 | — | — | 与本模型无关，排除 |

**关于 D 的显存账**：8B-NV 本体 BF16 约 17.5 GB，K100AI 63.98 GiB，
当前 `--gpu-memory-utilization 0.85` 下 KV cache 容量为 253,312 token（见交付报告 §9）。
再挂 4B BF16（约 8 GB）会压缩 KV cache，**可能触发 16384/并发64 档的 OOM**。
这必须单独核算，不能与 B 混做。

---

## 5. 实验计划（单变量，逐级放行）

前置门（**全部不通过则不启动 E3**）：

- [x] G1 当前同卡压测 `fullbench_candidate_20261101_114614` 已于 12:47 跑完；
      16K/c64 档 rc=4 的原因是脚本守卫 `[guard] 已有服务进程运行，拒绝启动`，
      **不是设备或服务故障**，该档与基线同缺、可比性不受影响
- [x] G2 卡指纹已记录：`host:crdnotebook-2114695384420274177-scnjz2s4uc-41571:boot:62ce6c81-9a7b-44b6-b994-b5054faa7f6d`，
      eager 基线（09:07:01 起）与 candidate（11:46:14 起）**完全同卡**，可直接相减
- [x] G3 官方压测客户端采样口径已确认（temperature=0.0 / ignore_eos=False / max_tokens=output_len）

#### 同卡现成基线（E3 的对照口径，2026-11-01 实测）

> ⚠️ **本表是 2026-11-01 12:40 的成稿快照，不随后续补跑更新。**
> 16384/并发64 档在成稿时两侧均未跑（NOT_RUN），此后已在同一张卡上另行补跑；
> **最终六档数据以 `reports/25_tuning_delivery_20261031.md` 与 `submission/bench/` 为准**，
> 本表仅用于 E3 实验设计阶段估算 E3 候选的收益量级。

| 档位 | eager 基线 tok/s（2 轮） | candidate 最优版 tok/s（2 轮） | 相对提升 |
| --- | --- | --- | ---: |
| 4096/c1 | 21.78 / 21.76 | 28.79 / 28.71 | **+32.1%** |
| 4096/c8 | 101.51 / 101.79 | 113.68（仅 1 轮） | +12.0% |
| 16384/c1 | 16.42 / 16.57 | 19.80 / 19.06 | +18.1% |
| 16384/c8 | 44.33 / 43.99 | 45.28（仅 1 轮） | +2.4% |
| 4096/c64 | 133.47 / 134.24 | 140.70（仅 1 轮） | +5.2% |
| 16384/c64 | NOT_RUN（成稿时两套同缺） | NOT_RUN | — |

> E3 的 spec-decode 候选应与 **candidate 最优版**这一列相减（不是与 eager 相减）：
> spec decode 是叠加在现有最优配置之上的单变量。

### E3-1：服务级单变量 AB（核心实验）

| 项 | 内容 |
| --- | --- |
| 唯一主要变量 | `--speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":3,"prompt_lookup_max":8}'` |
| 保持不变 | 模型、权重、dtype、TP/PP、attention 后端、rocBLAS 调优库、graph、`--max-num-batched-tokens 4096`、`--max-num-seqs 64`、`--limit-mm-per-prompt`、`--no-enable-prefix-caching`、prompt/图像/seed/请求数 |
| 档位 | 先只跑 **4096/并发1** 与 **4096/并发8**（各 2 轮），跑通再扩 |
| 计时口径 | 与现有 `scripts/run_full_bench.sh` 完全一致（同一客户端、同一 `--disable-ignore-eos`、同一 seed） |
| 判据 | 收益需 > 轮间波动（基线轮间极差：4096/c1 0.07%、4096/c8 0.52%）；报告 accepted tokens/forward 实测值 |
| 回退 | 去掉 `--speculative-config` 一项即可，其余不动 |

### E3-2：精度等价性自证（必须做，且必须先于任何性能宣称）

因官方判分器 `BLOCKED`，采用**自建对照**（明确标注为内部指标，不冒充官方 98%）：

1. 固定 32 条真实任务样本（沿用 `samples/bench_public` 已冻结集），同一服务、同一 greedy 配置；
2. A 组：关闭投机解码；B 组：开启；
3. 逐条比对 **完整 token 序列**（非字符串模糊匹配）是否完全一致；
4. 一致率 < 100% 时，逐条定位首个分歧位置，判断是 argmax 平局还是真实偏差；
5. 阈值：无分歧 = 通过；出现分歧必须定位原因并评估是否可接受，**不自行放宽阈值**。

### E3-3：注意力后端交叉验证（若 E3-1 出现数值异常时启用）

若 E3-1 输出与 E3-2 出现不可解释的分歧，则说明 aiter FA 在投机校验下可能有问题。
此时**单独**做一组：`ROCM_AITER_FA` vs 默认后端，其余全同，定位是否为后端问题。
这一步是**排障**，不作为收益实验，且不得与 E3-1 的结论混算。

### E3-4：负载相关性验证（决定是否保留）

在 **4096/并发1**（压测负载）与 **真实任务提示**（如 qspatial 真实提示）两种流量下各测一次接受率。
若确认「只在 gibberish 压测负载受益」，必须在报告里显式写明这一限制，
**不得把压测收益当作通用收益宣称**。

---

## 6. 风险与停止条件

| 风险 | 处置 |
| --- | --- |
| aiter FA 不支持投机校验（§2.3） | E3-1 若输出异常 → 走 E3-3 排障；仍不行则**停止本路线**，如实记录，不强行换后端 |
| 精度不可验证 | 官方判分器 BLOCKED 状态不变时，精度结论只能写「自建对照一致率」，不得写「通过官方 98%」 |
| 收益只在 gibberish 负载成立 | E3-4 显式披露；报告口径与赛事压测口径对齐，同时标注适用边界 |
| KV cache 被草稿模型挤占 | 方案 D 单独立项，先算显存账再决定 |
| 与最优版rocBLAS 调优库交互 | 投机解码改变 GEMM shape 分布（decode 阶段 batch 形态变化），**必须复核调优库仍命中**（`ROCBLAS_DEBUG_GENERALITY=1` 看 kernel 号），不假设仍最优 |
| 压测中途失败 | 保留失败原始数据；不降并发、不减图、不缩短输出冒充通过 |

---

## 7. 产物与证据路径

| 内容 | 路径 |
| --- | --- |
| 本方案 | `agent_work/reports/26_mtp_speculative_plan_20261101.md` |
| 接受率仿真脚本 | `agent_work/tools/spec_accept_sim.py` → 远程 `/root/private_data/workspace/robo/tools/spec_accept_sim.py` |
| 接受率仿真原始数据 | `agent_work/evidence/spec_accept_sim.json` → 远程 `/root/private_data/workspace/robo/outputs/spec_accept_sim.json` |
| 仿真运行日志 | 远程 `/root/private_data/workspace/robo/logs/spec_accept_sim.log` |
| 只读勘察脚本 | 远程 `/root/private_data/workspace/robo/tmp/check_*.sh`（9 个，含 config/后端/采样/上游对照） |

---

## 8. 已验证事实 / 待验证假设 / 阻塞

**已验证事实**
- checkpoint 无 MTP/nextn 头；`8B-MT` 是摩尔线程集群变体，非 MTP
- vLLM 0.15.1 的 `ngram` 配置可正确解析；`suffix` 因缺 `arctic-inference` 不可用（可装）
- aiter FA 后端无任何 spec-decode 专用处理；官方 `rocm_attn.py` 走 varlen
- 官方压测客户端 `temperature=0.0`、`ignore_eos=False`
- greedy 下投机解码以 target argmax 判定接受，输出序列理论保序
- 离线实测接受率：真实任务 0.28–0.56；官方压测负载 0.87–0.93
- 本轮未干扰运行中的同卡压测

**待验证假设**
- aiter FA 在投机校验下的数值正确性（E3-1 + E3-3）
- 真实服务下的接受率是否与离线仿真一致（并发、batch 形态会改变重复度）
- 调优 rocBLAS 库在投机解码 shape 分布下是否仍命中（E3-1 需复核 kernel 号）

**阻塞**
- 官方 ≥98% 一致率：GPU 参考输出、官方测试集、官方判分器**均未下发** → 继续 `BLOCKED`
- 官方镜像链接 / 平台截图 / 区域地址：待用户补
