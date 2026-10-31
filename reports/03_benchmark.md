# 第三阶段：功能与性能基线（transformers 后端，单卡 K500SM_AI）

- 时间：2026-10-26 22:53 ~ 23:05
- 原始数据：`outputs/phase3_baseline/baseline_raw.json`
- 汇总表：`outputs/phase3_baseline/baseline_summary.md`
- 运行日志：`logs/phase3_baseline_20261026_225349.log`

## 1. 测量口径

| 项 | 设定 |
|---|---|
| 设备 | 1 × K500SM_AI（63.98 GiB），`torch.cuda.device_count()=1` |
| 精度 | bfloat16（`dtype="auto"` → 读取 config 中的 bfloat16） |
| 后端 | transformers 5.3.0 `generate()`，attn 走 SDPA（内部调用 flash-attn） |
| 样例 | `samples/tasks_official.json` 6 条，**官方 demo 图 + 官方提示词** |
| 重复 | 每样例 3 次；**第 1 次计为 warm-up，不计入聚合统计** |
| 计时 | 真实流式（`TextIteratorStreamer`），TTFT = 首个 token 到达时刻；前后均 `torch.cuda.synchronize()` |
| 输出长度 | 按任务固定：general 128 / grounding 64 / pointing 64 / trajectory 192 |
| 采样 | `do_sample=False`（贪心），保证可复现 |

**样例覆盖范围**（`expect` 为人工判据，非官方 benchmark）：

| sample | task | 图像尺寸 | 期望 |
|---|---|---|---|
| official_general | general | 640×426 | 描述图中场景 |
| official_grounding | grounding | 426×640 | 归一化 bbox |
| official_affordance | pointing | 640×426 | 归一化点 |
| official_pointing | pointing | 640×480 | 归一化点 |
| official_navigation | pointing | 732×1220 | 归一化点 |
| official_trajectory | trajectory | 424×240 | 3D 关键点序列 |

> 样例仅覆盖 **图像+文本单轮** 的 4 类任务。视频输入（Robo-Dopamine 的 GRM 流水线）尚未进入基线。

## 2. 稳态聚合（n=12，即 6 样例 × 2 次非 warm-up 重复）

| 指标 | mean | median | min | max |
|---|---|---|---|---|
| TTFT (s) | **0.4918** | 0.3991 | 0.2487 | 1.1078 |
| 端到端时延 (s) | **2.9681** | 1.9405 | 1.2273 | 6.9995 |
| 解码吞吐 (tok/s) | **11.8357** | 11.9395 | 11.2400 | 12.0820 |
| 端到端吞吐 (tok/s) | **8.8668** | 8.7940 | 5.0070 | 11.4090 |
| 峰值显存 reserved (GiB) | **16.5497** | 16.5430 | 16.4340 | 16.6860 |

模型加载耗时：**62.12 s**（冷缓存 96 s / 热缓存 62 s，取决于权重分片是否在 page cache 中）。

## 3. 逐样例（稳态）

| sample | task | prompt tok | gen tok | TTFT s | total s | decode tok/s | peak VRAM GiB |
|---|---|---|---|---|---|---|---|
| official_general | general | 277 | 78 | 0.399 | 6.93~7.00 | 11.95~11.82 | 16.543 |
| official_grounding | grounding | — | 18 | 0.399 | 1.90 | 12.01~11.96 | 16.543 |
| official_affordance | pointing | — | 10 | 0.392 | 1.23 | 11.84~11.94 | 16.543 |
| official_pointing | pointing | — | 10 | 0.421 | 1.26 | 11.85~11.94 | 16.549 |
| official_navigation | pointing | — | 10 | 1.097 | 1.99 | 11.24 | 16.686 |
| official_trajectory | trajectory | — | 51 | 0.251 | 4.47 | 12.07~12.08 | 16.434~16.686 |

## 4. 瓶颈定位（基于上述测量）

1. **解码速度是主瓶颈**：所有样例的 decode 吞吐几乎恒定在 **11.2~12.1 tok/s**，
   与样例长度、图像尺寸无关 → 瓶颈是 **逐 token 解码的显存带宽 / kernel 效率**，
   不是 prefill，也不是视觉编码。
2. **prefill（TTFT）随视觉 token 数增长**：
   `official_navigation`（732×1220）TTFT 1.10 s vs `official_trajectory`（424×240）0.25 s，
   约 4.4 倍差距 ≈ 图像面积比（4.4×）。**图像分辨率是 TTFT 的直接杠杆**。
3. **视觉编码不构成瓶颈**：warm-up 首次请求 TTFT 5.42 s（official_general），
   稳态降到 0.40 s，差值是一次性的 kernel 编译/缓存开销，非稳定态成本。
4. **显存占用低**：峰值 16.69 GiB / 63.98 GiB ≈ **26%**，KV cache 与并发还有很大余量。

## 5. 输出正确性（关键）

三次重复输出**完全一致**（`do_sample=False`），且均符合官方输出格式：

```
official_grounding  -> [0, 309, 620, 989]
official_affordance -> [(901, 600)]
official_pointing   -> [(582, 725)]
official_navigation -> [(400, 376)]
official_trajectory -> [(294, 525, 0.577), (405, 350, 0.607), (525, 300, 0.720)]
```

坐标为 **0~1000 归一化值**（与上游 `draw_on_image` 的 `rel_to_abs` 约定一致）。
说明模型在 K500SM_AI 上不仅"能跑"，而且**微调后的空间输出格式完整保留**。

### 5.1 官方解析正则的覆盖边界（验收标准 §4.3 要求的单独检查）

验收标准 §4.3 明确：「**不能仅靠上游绘图正则是否匹配来判模型正确性**；
其对整数深度、科学计数法等格式的覆盖需要另行检查」。

`scripts/test_parse_coverage.py` 用 **20 条用例**（真实录到的输出 + 构造的边界格式）
跑一遍官方三条正则，逐条记录「原文 → 解析值 / 失败原因」：

| 类别 | 官方正则 | 覆盖情况 |
|---|---|---|
| trajectory | `(\d+),\s*(\d+),\s*([+-]?\d+\.\d+)` | 覆盖小数深度（含负数、带前导 0）；**不覆盖**整数深度 `[(1,2,3)]`、科学计数法 `1e-3`、缺前导 0 的 `.577`、浮点坐标 |
| pointing | `\(\s*(\d+)\s*,\s*(\d+)\s*\)` | 覆盖整数坐标（可多个点）；**不覆盖**浮点坐标 `901.5`、缺括号 |
| grounding | `\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]` | 覆盖整数四元组；**不覆盖**浮点坐标、三个数 |

实测 **12/20 可解析、8 条暴露缺口**；本节点**保持官方口径不改**
（改了就会与官方评测不一致），只做**记录**：

- 原始记录：`outputs/phase4_parse_coverage/parse_coverage.md`（逐条原文/解析值/失败原因）
  与 `parse_coverage.json`；
- 脚本会断言自身正则与 `scripts/robo_infer.py` **逐字一致**，防止日后改了实现却没改口径；
- **解析失败不得剔除出官方计分分母** —— 这条落在 `submission/verify.py`：
  输出解析失败记为未命中且留在分母里，请求失败同样计入分母（早期版本会把失败样本
  `continue` 掉，等于靠丢样本把一致率做高，已修正）；
  可执行测试 `tools/test_verify_denominator.py` 断言 4 道题（1 对 / 1 请求失败 / 1 不可解析 /
  1 答错）得到 `scored=4`、`errors=1`、一致率 **25%** 而非剔除后的 50%。
- vLLM 服务侧「分析模式」输出的 `<bbox>x y x y</bbox>` 由 `scripts/answer_adapter.py`
  做**纯格式桥**（四个整数原样搬运）后再交给官方正则解析 —— 见 `10_phase5_bench.md`。

## 6. 基线口径（后续优化以此为对照）

- 后端：transformers 5.3.0 直跑，单请求
- 解码吞吐：**11.84 tok/s**
- TTFT：**0.4918 s**（mean）/ 0.3991 s（median）
- 峰值显存：**16.55 GiB**
- 模型加载：**62.12 s**

## 7. 尚未纳入基线的项

- 视频 / 多图输入（GRM 时间价值估计，官方 `Robo-Dopamine` 流水线）
- 并发 / 批处理吞吐
- SGLang / vLLM 服务化后端
- 图像分辨率参数（`min_pixels` / `max_pixels`）的影响

以上进入第四阶段。
