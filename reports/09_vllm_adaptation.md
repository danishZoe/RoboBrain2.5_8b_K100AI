# 第四阶段 G：vLLM serving 适配（官方指定框架之一）

> 节点：单卡 K100AI（`K500SM_AI` / `gfx928` / 63.98 GiB），DTK 26.04，torch 2.9.0，transformers 5.3.0，vLLM `0.15.1+das.opt1.alpha.dtk2604`
> 官方依据：PDF 第 2 页「T02 视觉多模态理解模型（VL / VLA）**SGLang / vLLM 适配**」；计件粒度 = 模型 × 卡型 × 框架，**SGLang 与 vLLM 分别计件**
> 本报告结论：**vLLM 路径已跑通并产出有效推理输出**；SGLang 路径不可用（见 `06_sglang_compatibility.md`）

---

## 1. 一句话结论

vLLM 在本节点**跑通了**：服务起来、能接收带图请求、四类任务全部可被官方正则解析、
多图/流式/多轮/异常输入全部通过服务级冒烟，吞吐约为 transformers 直跑的 **2 倍**。

过程中踩到并解决了 **4 个真实阻塞**（2 个架构级 API 不兼容、1 个容器内存上限导致 Pod 被杀、1 个脚本 bug），
以及 **2 个此前被误判的"精度不一致"根因**：

1. **grounding 输出对数值扰动敏感**——实际根因不是"数值扰动"，而是
   **官方 upstream 的 `qwen_vl_utils` 对图像做了二次重采样**，serving 路径只做了一次；
   像素最大差 0.47（归一化 ±1 空间），直接翻转了模型首个生成 token 的选择。
2. **`<bbox>` 与 `[x, y, x, y]` 的表面格式不一致**——同一后端在不同回答模式下表面形式不同，
   官方正则只认方括号形式，需要一层**纯格式桥**（不改动任何数值）。

> 官方「精度 ≤2% 劣化 / 一致率 ≥98%」仍为 `BLOCKED`（缺官方 GPU 参考输出与判分器），
> 本报告给出的是**同卡不同后端**的一致性证据，不替代官方精度验收。见 `08_gap_vs_official_baseline.md`。

---

## 2. 四个阻塞与解决

### 2.1 阻塞一：vLLM 0.15.1 × transformers 5.3.0 的 config API 错位（致命）

**现象**（`logs/vllm_try1_20261027_005056.log`）：

```
File "vllm/model_executor/models/qwen3.py", line 397, in load_weights
    skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
AttributeError: 'Qwen3VLTextConfig' object has no attribute 'tie_word_embeddings'
...
RuntimeError: Engine core initialization failed.
```

**根因**：多模态模型里 `self.config` 是**子 config**（`Qwen3VLTextConfig`），
而 transformers 5.3.0 把 `tie_word_embeddings` 挪到了**顶层复合 config**（`Qwen3VLConfig`），
子 config 上根本没有这个属性。

**语义核对（用模型自身文件，不用猜）**：

| 证据 | 值 | 结论 |
|---|---|---|
| `config.json` 顶层 `tie_word_embeddings` | `false` | 未绑定 |
| `model.safetensors.index.json` 中 `lm_head.weight` | **存在** | 未绑定（与上一行互证） |
| 因此 vLLM 应当 | **加载** `lm_head`（`skip_prefixes=None`） | 值必须是 **False** |

**解决**（两个等价实现，**都不改 site-packages**）：

| 实现 | 文件 | 用途 |
|---|---|---|
| 运行时垫片 | `scripts/serve_vllm_patched.py` + `envs/vllm_shim/sitecustomize.py` | 本节点使用（共享环境不可改） |
| 可提交 diff | `patches/0001-vllm-qwen3-tie-word-embeddings.patch` | 提交到官方仓库时用 |

```python
# 官方 patch 的一行改动
-            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
+            skip_prefixes=(["lm_head."] if getattr(self.config, "tie_word_embeddings", False) else None),
```

### 2.2 阻塞二：垫片必须放在 `sitecustomize`（spawn 子进程不继承）

第一版垫片只打在**父进程**，`AttributeError` 依旧。原因：

```
vllm/envs.py:64:  VLLM_WORKER_MULTIPROC_METHOD: Literal["fork", "spawn"] = "spawn"
```

vLLM 用 **spawn** 起 EngineCore，子进程是全新解释器，**不继承**父进程的 monkey-patch。
因此垫片改为 `sitecustomize.py`（`site` 在**每个**解释器启动时自动 import），
由 `scripts/serve_vllm_patched.py` 通过 `PYTHONPATH` + `ROBO_VLLM_TIE_SHIM=1` 注入。
验证方式：在**独立解释器**里确认属性已生效：

```bash
ROBO_VLLM_TIE_SHIM=1 ROBO_MODEL_PATH="$MODEL_PATH" \
PYTHONPATH="envs/vllm_shim:$PYTHONPATH" python3 -c \
  "from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig as C; print(C.tie_word_embeddings)"
# -> [sitecustomize] Qwen3VLTextConfig.tie_word_embeddings = False
# -> False
```

### 2.3 阻塞三：vLLM 启动期 OOM → 容器被杀 → Pod 被重建

**现象**：01:02:34 启动 vLLM（try3），01:07 SSH 通道即断，此后 5 小时不可达；
主机名从 `...-scnjz2s4uc-68757` 变为 `...-scnjz2s4uc-**92379**` → **实例被重新创建**。

**取证**：

| 证据 | 值 |
|---|---|
| try3 日志最后一行 | `01:05:40 Dynamo bytecode transform time: 7.17 s`（正进入 inductor 编译） |
| 容器内存硬上限 | `/sys/fs/cgroup/memory/memory.limit_in_bytes` = **123480309760 = 115 GiB** |
| 宿主内存 / 核 | 528 GB / 32 核（容器受限 115 GiB / 6 核） |

`torch.compile` + CUDA graph capture 是 vLLM 启动期内存峰值最高的环节，触顶 115 GiB 后
容器被 OOM killer 杀掉 → Pod 重建 → SSH 侧车失效。

**解决**：首跑一律 `--enforce-eager`（跳过编译与 graph 捕获）。
实测容器内存峰值仅 **11.2 GB**，启动耗时 ~200 s，服务正常就绪。
完整降内存参数阶梯见 `00_RUNBOOK.md` §5.5。

> 纪律：`--enforce-eager` 会影响吞吐，**首提性能基线只能取一次**。
> 正确顺序是「先用 eager 确认能起来 → 再用非 eager 跑首提基线」。

### 2.4 阻塞四：脚本自身 bug（`set -u` + DTK `env.sh`）

`set -uo pipefail` + `source ./env.sh` → `/opt/dtk/env.sh: line 38: CMAKE_PREFIX_PATH: unbound variable` → 脚本立即退出。
已把全部 6 个 `.sh` 脚本的 `set -uo pipefail` 改为 `set -o pipefail`（各脚本已统一使用 `${VAR:-}` 默认值）。

---

## 3. 根因一：grounding 首个 token 是平局，而官方管线喂的是**二次重采样**后的像素

这是本轮最重要的发现，也是此前把问题误判为「数值扰动」的地方。

### 3.1 现象

| 后端 / 路径 | grounding 样例输出 | 官方正则可解析 |
|---|---|---|
| transformers + 官方 `qwen_vl_utils` 取图 | `[0, 309, 620, 989]` | ✅ |
| vLLM + 直接上传原图 | 长篇分析 + `<bbox>0 298 622 998</bbox>` | ❌ |
| transformers + 直接 PIL 取图（探针） | 长篇分析 + `[20, 290, 480, 780]` | ✅（在末尾） |

### 3.2 逐条排除

| 假设 | 验证方法 | 结果 |
|---|---|---|
| 提示词不同 | 与官方 `inference.py` 的 `build_prompt` 逐字比对；`processor.chat_template` 与 `tokenizer.chat_template` 完全相同（均 5292 字符） | **排除** |
| 采样参数不同 | 探针显式 `temperature=0 / top_p=1 / seed=0`；官方默认 `do_sample=True, temperature=0.7`，但本次两侧都是贪心 | **排除** |
| 图像分辨率不同 | `max_pixels=501760` 与默认 `16,777,216` 下 `pixel_values` `torch.equal=True` | **排除** |
| 随机非确定性 | 3 个独立进程 × 4 次 = 12 次 greedy，输出 **12/12 完全相同** | **排除** |
| 模型 dtype 不同 | 两条路径均为 `torch.bfloat16`（顶层与子模块一致） | **排除** |
| **图像预处理路径不同** | 见 3.3 | **命中** |

### 3.3 机制

官方 upstream（`third_party/RoboBrain2.5/inference.py:81`）用
`process_vision_info(messages)` 取图，该函数内部（`qwen_vl_utils/vision_process.py:93`）执行：

```python
patch_factor = image_patch_size * SPATIAL_MERGE_SIZE   # 14 * 2 = 28
resized_height, resized_width = smart_resize(height, width, factor=28, ...)
image = image.resize((resized_width, resized_height))  # PIL BICUBIC
```

随后 processor 再按 `patch_size 16 × merge_size 2 = 32` 做**第二次**缩放。
而 serving 路径直接上传原文件，只经历 processor 的那一次缩放：

```
官方路径 : 426x640 --PIL bicubic(factor 28)--> 420x644 --torchvision bicubic(factor 32)--> 416x640
serving  : 426x640 --------------------- torchvision bicubic(factor 32) -------------> 416x640
```

两次重采样的像素与一次重采样**不是同一个图**。实测：

```
[pixels] identical=False   input_ids identical=True   max|diff|=0.470588
```

token 网格（1040 patch / 290 token）完全相同，所以**光看 prompt token 数发现不了这个问题**。

### 3.4 像素差异如何翻转输出

模型首个生成 token 的分布（bf16，取自 `scripts/probe_tie.py`）：

| 取图路径 | 首个 token | logit | prob | 后续走向 |
|---|---|---|---|---|
| 官方 `qwen_vl_utils` | `[` | **26.125** | 0.6718 | 直出 `[0, 309, 620, 989]` |
| 直接 PIL / vLLM | `The` | **24.500** | 0.4056 | 转写分析模式 |
| 直接 PIL / vLLM | `To` | **24.500** | 0.4056 | **与 `The` 的 logit 在 bf16 下完全相等** |

`The` 与 `To` 的 logit 被 bf16 量化到同一个值 24.5 —— 二者本来就是平局，
**任何**像素级扰动都足以决定走「直出坐标」还是「先写一段分析」。
这就是为什么一旦换栈或换取图方式，输出会**整体换一种风格**而不是只有末位数字变化。

---

## 4. 修复一：客户端图像规范化（已验证逐位一致）

`scripts/image_canonical.py`：**用上游 `qwen_vl_utils.fetch_image` 预缩放，再用无损 PNG 上传**，
服务端只补上第二次缩放，于是像素与官方管线**逐位相同**。

```python
canonical = fetch_image({"image": path}, image_patch_size=14)   # 复用上游算法，不自己重写
canonical.save(buf, format="PNG")                               # 无损，避免 JPEG 再损失
```

验证（CPU，逐位比较 `pixel_values`）：

| 样例图 | 原图尺寸 | 规范化后 | `官方 == 规范化` | `官方 == 原图上传` |
|---|---|---|---|---|
| official_affordance.jpg | (640, 426) | (644, 420) | **True** | False |
| official_grounding.jpg | (426, 640) | (420, 644) | **True** | False |
| official_navigation.jpg | (732, 1220) | (728, 1232) | **True** | False |
| official_pointing.jpg | (640, 480) | (644, 476) | **True** | False |
| official_trajectory.jpg | (424, 240) | (420, 252) | **True** | False |

即：**5/5 样例图规范化后与官方像素完全相同，原图上传则全部不同**。

**修复效果**（同一服务、同一模型、同一批样例，仅取图方式不同）：

| 样例 | transformers 基线 | vLLM + 规范化 | vLLM + 原图（修复前） |
|---|---|---|---|
| official_grounding | `[0, 309, 620, 989]` | **`[0, 309, 620, 989]`（完全一致）** | 不可解析（`<bbox>` 形式） |
| official_affordance | `[(901, 600)]` | `[(904, 600)]`（Δ3） | `[(901, 600)]`（Δ0） |
| official_pointing | `[(582, 725)]` | **`[(582, 725)]`（完全一致）** | `[(587, 725)]`（Δ5） |
| official_navigation | `[(400, 376)]` | `[(400, 375)]`（Δ1） | `[(400, 372)]`（Δ4） |
| official_trajectory | 首点 (294, 525, 0.577) | (301, 525, 0.572)（**Δ7**） | (301, 525, 0.592)（Δ25） |
| **最大坐标偏差** | — | **7/1000 = 0.7%** | 25/1000 = 2.5% + 1 条不可解析 |

> 该修复只改变**输入预处理**，不改权重、不改采样参数、不改输出文本。

---

## 5. 修复二：输出格式桥（纯格式，不动数值）

模型的 grounding 回答有两种表面形式：方括号形式与 `<bbox>x y x y</bbox>` 形式。
官方正则（`inference.py:110-124`）只认前者：

```python
box_pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
```

`scripts/answer_adapter.py` 做**纯格式桥**：把 `<bbox>x1 y1 x2 y2</bbox>` 改写成 `[x1, y1, x2, y2]`，
四个整数**原样搬运**，不取整、不裁剪、不重排、不臆造；原始文本始终单独保留在结果里（`text` vs `text_normalized`）。

```python
RE_BBOX_TAG = re.compile(
    r"<bbox>\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)"
    r"\s+(-?\d+(?:\.\d+)?)\s*</bbox>")
```

已接入：`scripts/probe_http_request.py`、`scripts/smoke_service.py`。
`probe_http_request.py --no-canonical` 可关闭规范化做 A/B。

---

## 6. 性能对比（同卡、同样例、同贪心）

| 口径 | transformers 直跑（基线） | vLLM serving（eager） |
|---|---|---|
| 单请求解码吞吐 | 11.84 tok/s | **21–25 tok/s**（见第 7 节） |
| 单请求端到端（grounding） | 1.90 s | **0.91 s** |
| 并发吞吐（conc=8） | —（直跑无并发口径） | **158.47 tok/s** |
| 峰值显存 | 16.55 GiB | 见第 7 节 |
| 容器内存峰值 | — | 11.2 GB（启动期，eager） |

---

## 7. vLLM 全套验证结果

> 证据目录：`outputs/phase5_probe_canonical/`、`outputs/phase5_probe_nocanonical/`、
> `outputs/phase5_smoke/`、`outputs/phase5_bench/`、`outputs/vllm_mt768/`、`outputs/vllm_multi_image/`、`outputs/phase4/`

### 7.1 服务级冒烟（`scripts/smoke_service.py`，同一服务入口）

| 检查项 | 结果 |
|---|---|
| `/health` | 200 |
| `/v1/models` | 200，返回 `robobrain2.5-8b-nv`，`max_model_len=32768` |
| F01 general | 200，有效场景描述 |
| F02 grounding | 200，`bbox=[0, 309, 620, 989]` |
| F03 affordance | 200，`(904, 600)` |
| F04 pointing | 200，`(582, 725)` |
| F05 navigation | 200，`(401, 376)` |
| F06 trajectory | 200，3 点含深度 |
| F07 多图 general | 200，回复长度 2223 字符（两图均参与） |
| F08a 损坏图片 | **400**，未静默接受 |
| F08b 不可达图片 URL | 200 但内容为空（未产出伪结果） |
| F08c 畸形 data URL | **400** |
| F08d 空间任务 + 2 图 | 200 —— 上游契约要求空间任务**恰好 1 图**，该约束在**客户端**实现，服务端 schema 无关 |
| 流式 vs 非流式 | 拼接结果 **完全一致**（78 chunk，TTFT 0.1253 s） |
| 多轮图文 | 200，回复 546 字符 |

### 7.2 探针（6 样例，768 token 预算，官方提示词）

| 样例 | 规范化后 | 原图上传 | 基线 |
|---|---|---|---|
| general | 场景描述 | 场景描述 | 场景描述 |
| grounding | `[0, 309, 620, 989]` | `<bbox>0 298 622 998</bbox>`（原文） | `[0, 309, 620, 989]` |
| affordance | `[(904, 600)]` | `[(901, 600)]` | `[(901, 600)]` |
| pointing | `[(582, 725)]` | `[(587, 725)]` | `[(582, 725)]` |
| navigation | `[(400, 375)]` | `[(400, 372)]` | `[(400, 376)]` |
| trajectory | `[(301,525,0.572),(405,350,0.607),(525,300,0.722)]` | `[(301,525,0.592),(405,375,0.625),(520,325,0.722)]` | `[(294,525,0.577),(405,350,0.607),(525,300,0.720)]` |

请求 6/6 成功，解析 6/6 成功（两个口径均是）。

### 7.3 六档负载压测

见 `10_phase5_bench.md`（按验收标准 §7.1 的档位、两轮、失败项如实保留）。

### 7.4 压测口径的两处修正（本轮新发现，详见报告 10 §3.4）

这两处都会**直接改变结论**，且都不会从客户端的默认字段里暴露出来：

| # | 现象 | 根因 | 修正 |
|---|---|---|---|
| 1 | 16384 档请求被服务拒收 | `max_model_len=32768` 装不下「文本（随机 token 解码后重编码膨胀 ~5.4%）+ 视觉 2,042 + 输出 1,024」 | `MAX_LEN` → **32768**（官方 `env.sh` 口径） |
| 2 | 官方客户端吞吐虚高约 **4×**，但 `total_output_tokens` 显示 1024×N | ① 客户端 `total_output_tokens` 统计的是**计划长度**（`output_lens.append(output_len)`），真实长度在 `total_output_tokens_retokenized`；② `--disable-ignore-eos` 的语义是**允许提前 EOS**（`payload["ignore_eos"] = not args.disable_ignore_eos`），官方脚本不传该开关即为 `ignore_eos=True` | `bench.sh` 删掉该开关，回到官方默认 |

修正后官方客户端与自有客户端在同一档位（4096/1024/并发 1）交叉验证一致：

| 指标 | 官方客户端 | 自有客户端 | 相对偏差 |
|---|---|---|---|
| 输出吞吐 tok/s | 25.21 | 25.36 | +0.6% |
| 平均 TTFT ms | 798.62 | 725.30 | −9.2% |
| 平均 TPOT ms | 38.92 | 38.75 | −0.4% |

---

## 8. 提交配置建议（基于本次实测）

| 项 | 取值 | 依据 |
|---|---|---|
| 框架 | vLLM（可另计 SGLang 一件，但本节点不可用） | PDF 第 2、3 页 |
| 镜像 | `jupyter-sglang5.12-ubuntu22.04-dtk26.04-py3.10-qwen3.6:v1.0` | 用户提供，含 vLLM 0.15.1 dtk2604 |
| 卡型 | `CARD="K100-AI"` | 官方 `env.sh` 明文「K500-SM 按 K100-AI 计」 |
| 精度 | `bf16` | 模型权重 dtype，硬件原生支持 |
| TP | `1` | 本 Pod 仅 1 张卡；官方脚手架默认 4，**需在报告里标注口径** |
| `max_tokens` | **≥768** | 官方 `inference.py:93` 默认 |
| 图像预处理 | **客户端 `qwen_vl_utils` 规范化 + PNG 上传** | 见第 4 节，否则像素与官方不一致 |
| 输出处理 | `scripts/answer_adapter.py` 格式桥 | 官方正则只认 `[x, y, x, y]` |
| 启动参数 | `--enforce-eager`（首跑保命）；非 eager 版本另跑一次取首提基线 | 容器 115 GiB 上限 |
| 显存占比 | `0.85` | 与官方 `env.sh` 的 `MEM_FRAC` 一致 |
| `max_model_len` | **`32768`**（官方 `env.sh` 口径） | 16384 档单请求实际需要 ≈20,336 token（文本 ~17,270 + 视觉 2,042 + 输出 1,024），20480 装不下；模型 `max_position_embeddings=262144` 未改。实测该值不改变 KV cache 容量（均 253,312 token） |
| patch | `patches/0001-vllm-qwen3-tie-word-embeddings.patch` | vLLM×transformers API 错位 |

---

## 9. 尚存限制

1. **精度仍 `BLOCKED`**：官方 GPU 参考输出、官方测试集、官方判分器均未取得，
   第 4 节的一致率是**同卡不同后端**指标，不能替代官方 ≥98% 一致率。
2. **grounding 首 token 平局**：即使规范化后两后端仍可能落在不同分支
   （规范化后 grounding 完全一致，但 general 等自由文本仍有措辞差异）。
   官方判分必须在官方题集 + 官方参考输出上复测。
3. **单卡口径**：官方脚手架默认 TP=4，性能对比的「同负载」需在报告中显式说明。
4. **`--enforce-eager` 的吞吐代价未量化**：需与非 eager 版本做同口径 A/B。
5. **六档压测中并发 64 档受 KV cache 容量限制**，需在报告中如实标注实际并发度。
