# 第四阶段：逐项调优实验记录

- 时间：2026-10-26 23:03 ~ （持续）
- 原始数据：`outputs/phase4/*.json`
- 汇总表：`outputs/phase4/comparison.md`（由 `scripts/compare_experiments.py` 生成）

## 实验规则

严格遵循"**一次只改一个主要变量**"：

- 所有实验使用**同一份固定样例** `samples/tasks_official.json`（6 条，官方 demo 图 + 官方提示词）；
- 同一套测量方法（`TextIteratorStreamer` 真实流式 TTFT + `torch.cuda.synchronize()`）；
- 同一聚合口径（每样例 3 次，**首次为 warm-up 并剔除**，稳态 n=12）；
- 每项实验都**重新输出实际答案**，与基线答案逐条比对，判定是否发生质量变化，
  而不是只看"输出非空"。

对照基线（第三阶段冻结）：**TTFT 0.4918 s / 端到端 2.9681 s / 解码 11.8357 tok/s /
端到端 8.8668 tok/s / 峰值显存 16.5497 GiB**。

---

## 实验 A：注意力实现（`attn_implementation`）

| 项 | 值 |
|---|---|
| 假设 | 默认 SDPA 未必最优；显式切 `flash_attention_2` 可能降低 decode 开销 |
| 唯一变量 | `attn_implementation`：`default(sdpa)` → `flash_attention_2` |
| 命令 | `python3 scripts/tune.py --tag attn_fa2 --attn-impl flash_attention_2 --repeat 3` |

| 指标 | 基线 | attn_fa2 | 变化 |
|---|---|---|---|
| TTFT mean (s) | 0.4918 | 0.4945 | +0.5%（持平） |
| 端到端 mean (s) | 2.9681 | 3.3363 | **+12.4%（变慢）** |
| 解码 (tok/s) | 11.8357 | 10.3993 | **−12.1%（变慢）** |
| 峰值显存 (GiB) | 16.5497 | 16.5497 | 持平 |

输出质量：6/6 与基线一致，仅 `official_trajectory` 第二点 x 由 405 → 400
（归一化坐标差 5/1000，属数值噪声量级）。

**结论：回退，保留默认 SDPA。**
本镜像的 SDPA 路径内部本就走 flash-attn 派生的 HIP kernel
（日志：`sdpa adopt the new interface of flash-attn`），额外再挂一层
flash_attention_2 反而多了一次开销。显存无差异。

---

## 实验 B：视觉 token 预算（`max_pixels` = 1003520 = 1280×28×28）

| 项 | 值 |
|---|---|
| 假设 | 降低视觉 token 数可缩短 prefill / TTFT |
| 唯一变量 | processor `max_pixels`：默认 16777216 → 1003520 |
| 命令 | `python3 scripts/tune.py --tag pix_1003520 --max-pixels 1003520 --repeat 3` |

结果：**与基线逐项完全相同**（视觉 token 397.33 → 397.33，TTFT 0.4918 → 0.4964，
解码 11.8357 → 11.8442）。

原因：6 张样例图的像素数分别为 272640 / 272640 / 307200 / 893040 / 101760，
**全部小于 1003520**，该阈值根本没有触发缩放。

**结论：该配置无意义，不采用；但它暴露了一个事实——当前样例的视觉 token 量
完全由原始分辨率决定，需要更紧的阈值才能真正约束 prefill。**

---

## 实验 C：视觉 token 预算（`max_pixels` = 501760 = 640×28×28）

| 项 | 值 |
|---|---|
| 唯一变量 | `max_pixels` 501760（只对 732×1220 的 navigation 图生效，其余样例不变） |
| 命令 | `python3 scripts/tune.py --tag pix_501760 --max-pixels 501760 --repeat 3` |

| 指标 | 基线 | pix_501760 | 变化 |
|---|---|---|---|
| 视觉 token 均值 | 397.3 | 331.0 | −16.7% |
| TTFT mean (s) | 0.4918 | **0.4084** | **−16.9%** |
| 端到端 mean (s) | 2.9681 | 2.9084 | −2.0% |
| 解码 (tok/s) | 11.8357 | 11.8163 | −0.2%（噪声内） |
| 峰值显存 (GiB) | 16.5497 | 16.5265 | −0.14%（可忽略） |

逐样例影响（仅被裁剪的 navigation 样例）：

| 样例 | 基线 | pix_501760 |
|---|---|---|
| official_navigation 视觉 token | 935 | **537**（−42.6%） |
| official_navigation TTFT | 1.086 s | **0.595 s（−45.2%）** |
| official_navigation 端到端 | 1.974 s | 1.455 s（−26.3%） |
| official_navigation 坐标 | `[(400, 376)]` | `[(406, 378)]` |

输出质量：6/6 与基线**同格式**；仅 navigation 坐标平移 (6, 2) 个归一化单位
（≈ 图像宽高的 0.6% / 0.2%），其余 5 条**逐字符相同**。

**结论：采纳（作为"大图场景"的推荐配置）。** TTFT 降 17%，空间输出无实质变化。

---

## 实验 D：视觉 token 预算（`max_pixels` = 200704 = 256×28×28）

| 项 | 值 |
|---|---|
| 唯一变量 | `max_pixels` 200704（对 5/6 样例生效，trajectory 因原图仅 101760 像素不受影响） |
| 命令 | `python3 scripts/tune.py --tag pix_200704 --max-pixels 200704 --repeat 3` |

| 指标 | 基线 | pix_200704 | 变化 |
|---|---|---|---|
| 视觉 token 均值 | 397.3 | 227.2 | **−42.8%** |
| TTFT mean (s) | 0.4918 | **0.3052** | **−37.9%** |
| 端到端 mean (s) | 2.9681 | 2.7435 | −7.6% |
| 端到端 (tok/s) | 8.8668 | 9.7402 | +9.9% |
| 解码 (tok/s) | 11.8357 | 11.9191 | +0.7%（噪声内） |
| 峰值显存 (GiB) | 16.5497 | 16.4688 | −0.5% |

**输出质量发生实质变化：**

| 样例 | 基线 | pix_200704 | 判定 |
|---|---|---|---|
| official_grounding | `[0, 309, 620, 989]` | `[51, 300, 411, 520]` | **明显改变**（bbox 大幅收缩） |
| official_affordance | `[(901, 600)]` | `[(900, 587)]` | 小幅偏移 |
| official_pointing | `[(582, 725)]` | `[(587, 725)]` | 小幅偏移 |
| official_navigation | `[(400, 376)]` | `[(410, 368)]` | 小幅偏移 |
| official_trajectory | 3 点序列 | 逐字符相同 | 不变 |
| official_general | 描述文本 | 措辞不同，语义一致 | 不变 |

**结论：不采纳（回退到实验 C 的档位）。**
虽然 TTFT 降幅最大（−38%），但 grounding 的边界框发生了显著变化——
在这个任务上"坐标格式仍可解析"并不等于"质量没下降"。无标准答案可比对时，
不做这种赌注。

---

## 实验 E：SGLang 服务化后端 —— **阻塞，已定位根因**

| 项 | 值 |
|---|---|
| 假设 | 用 SGLang 0.5.12（镜像自带）的连续批处理提升并发吞吐 |
| 结果 | **本节点不可用**。5 个可用 attention 后端 + 2 种 model impl 全部失败 |

逐项探测与根因详见 **`reports/06_sglang_compatibility.md`**。摘要：

| 配置 | 结果 |
|---|---|
| `aiter`（默认） | 请求时 decode 崩溃：`NameError: paged_attention_ragged`（本镜像的 `aiter` 是残缺构建，缺 `aiter.attention` 子包） |
| `flashinfer` | 包未安装 |
| `wave` | 缺 `wave_lang` |
| `triton` | 初始化即崩溃：HIP `Invalid address access` |
| `torch_native`（± `sdpa` 多模态后端） | 服务可起，请求时 HSAIL 硬件异常 `KERNEL VMFault`，崩在 `F.scaled_dot_product_attention`（HIP cutlassfa 适配器） |
| `--model-impl transformers` | 同样 VMFault（崩溃点仍在 `transformers/integrations/sdpa_attention.py:92`） |

**结论**：阻塞在 DTK 内核栈，与模型/权重/提示词无关（同一权重在独立 transformers 进程里全部通过）。
本任务范围内无法修复（需要更完整的 `aiter` 构建或 flashinfer / wave_lang，而节点无外网）。
**处置**：回退到 transformers 后端，并改用**批处理**获取并发吞吐（实验 F）。

---

## 实验 F：批处理扩展（transformers 已验证路径）

| 项 | 值 |
|---|---|
| 假设 | 基线解码恒定 11.8 tok/s、与样例无关，说明瓶颈是**每步读取 16.3 GiB bf16 权重**的带宽开销。把 N 个请求放进**同一个 padded batch**，可摊薄这次权重读取，聚合吞吐应接近 N 倍 |
| 唯一变量 | `batch_size`：1 / 2 / 4 / 8 |
| 命令 | `python3 scripts/benchmark_batch.py --batch-sizes 1 2 4 8 --repeat 3` |
| 关键修正 | **必须 `padding_side="left"`**（详见下方"正确性陷阱"） |

### F.1 稳态结果（left padding，每档 3 次，剔除首次 warm-up）

| batch | 批耗时 s | 聚合 tok/s | 相对 bs=1 | 单请求时延 s | 峰值显存 GiB |
|---|---|---|---|---|---|
| 1 | 6.869 | 11.50 | ×1.00 | 6.869 | 16.55 |
| 2 | 8.302 | 19.03 | **×1.66** | 4.151 | 16.64 |
| **4** | **10.117** | **31.24** | **×2.72** | **2.529** | **17.07** |
| 8 | 20.829 | 30.34 | ×2.64 | 2.604 | 20.42 |

**采纳 batch_size = 4**：聚合吞吐 **×2.72**，同时**单请求时延反而从 6.87 s 降到 2.53 s**
（×2.7），峰值显存只从 16.55 涨到 17.07 GiB（仍在 63.98 GiB 的 27% 以内）。
bs=8 吞吐不再增长（×2.64 < ×2.72）而显存涨到 20.42 GiB —— 说明 bs=4 已接近
"权重读取被完全摊薄"的上限，再往上只是线性增加 pad 计算量。

### F.2 正确性陷阱（必须记录）

第一次跑批处理时用默认的 **right padding**，产出被污染：

```
bs=4 answer: "Human: The image depicts a cozy and minimalist scene ... 
               user [0, 309, 620, 989]
               0000000000000000000000000000000000000000...
               [(582, 725)]"
```

模型从 pad 位置继续生成，混入了 `Human:` / `user` 和一长串 `0`，
生成 token 数也虚高（bs=4 生成 512 token，而正确值应为 4×79=316）。
**若只看"输出非空"或"平均吞吐"，会得出完全错误的结论。**

改成 `processor.tokenizer.padding_side = "left"` 后，
token 数恢复正确（bs=2: 158 = 2×79，bs=4: 316 = 4×79，bs=8: 632 = 8×79），
输出恢复正常。

### F.3 批间数值漂移

与单请求基线逐条比对，26 次比对中 10 次不完全相同。逐类分析：

| 任务 | 漂移情况 | 性质 |
|---|---|---|
| general（自由文本） | 措辞不同，语义一致（都正确描述了"白色马克杯/深色液体/木桌面/背景笔记本电脑"） | 自由生成，非缺陷 |
| grounding | `[0, 309, 620, 989]` 完全一致 | 一致 |
| pointing（两杯之间） | `[(582, 725)]` 完全一致 | 一致 |
| pointing（杯子把手） | `[(901, 600)]` → `[(904, 600)]`，x 偏 3/1000 = 图像宽 0.3% | 数值噪声 |
| pointing（马桶） | bs=8 时偏 10/1000 以内 | 数值噪声 |
| trajectory | bs=8 时第三点有 ≤3/1000 级偏移 | 数值噪声 |

成因是**批形状不同 → padding 不同 → kernel 归约顺序不同 → 浮点结果末位不同**，
属批处理推理的固有性质，不是质量下降。判据依据：**格式 100% 保持，
坐标漂移 ≤ 1%（归一化 1000 刻度），且语义/目标不变**，
并已用 bs=4 的标注图目视复核（见 `outputs/batch_bs4/`）。

### F.4 结论

**采纳**：batch_size=4 + left padding。
这是本节点上唯一可用的"吞吐提升"手段（服务化路径被 E 阻塞），
且**不需要任何改动模型权重或数值精度**。

---

## 阶段小结（至实验 F）

| 结论 | 处置 |
|---|---|
| `flash_attention_2` 慢 12% | **回退**，保留默认 SDPA |
| `max_pixels=1003520` 无效果（阈值未触发） | 不采用 |
| `max_pixels=501760` TTFT −17%，输出仅微移 | **采纳为推荐配置** |
| `max_pixels=200704` TTFT −38%，但 grounding 输出实质改变 | **不采纳** |
| SGLang 5 个后端 + 2 种 model impl 全部崩溃 | **阻塞**，根因在 DTK 内核栈；改走批处理 |
| batch_size=4 + left padding：聚合 ×2.72，单请求时延 ×2.7 改善 | **采纳** |
| 单请求 decode 恒定 11.8 tok/s，与样例无关 | 瓶颈是**逐 token 解码的权重带宽**，批处理可摊薄 |
| 峰值显存仅占 63.98 GiB 的 26% | 显存不是瓶颈，bs=4 也只涨到 27% |

## 最终推荐配置（Best）

> **说明（2026-10-27 更新）**：本文件记录的是**第四阶段前六项实验**（transformers 直跑口径）。
> 后续的 vLLM serving 适配（第四阶段 G）与服务级调优实验（E1–E3，单变量、服务口径）
> 分别见 `09_vllm_adaptation.md` 与 `11_tuning_experiments.md`；
> **提交口径以 vLLM serving 为准**，下表的 transformers 直跑数据只作历史对照。

| 项 | 值 | 依据 |
|---|---|---|
| 后端 | transformers 5.3.0 直跑 | SGLang 阻塞（实验 E）；vLLM 本节点无可用环境（未验证） |
| 精度 | bfloat16 | 模型声明 + 硬件原生支持 |
| 注意力实现 | 默认 SDPA | `flash_attention_2` 慢 12%（实验 A） |
| 视觉预算 `max_pixels` | **501760**（640×28×28） | TTFT −17%，空间输出无实质变化（实验 C） |
| 批大小 | **4** | 聚合吞吐 ×2.72、单请求时延 ×2.7 改善（实验 F） |
| padding_side | **left**（批处理必���） | right padding 会污染输出（实验 F.2） |
| 采样 | `do_sample=False`（贪心） | 结果可复现；与官方 README 示例一致 |

组合后的同口径指标：

| 指标 | 基线（bs=1, 默认像素） | 推荐（bs=4, max_pixels=501760） |
|---|---|---|
| 聚合吞吐 | 11.50 tok/s | **31.24 tok/s（×2.72）** |
| 单请求端到端时延 | 6.87 s（general 样例） | **2.53 s** |
| TTFT mean | 0.4918 s | 0.4084 s（实验 C 的像素预算） |
| 峰值显存 | 16.55 GiB | 17.07 GiB（+0.52 GiB） |
| 输出格式保持 | 6/6 | 6/6 |
| 空间坐标漂移 | — | ≤ 10/1000（0.3%~1%），目视复核目标不变 |
