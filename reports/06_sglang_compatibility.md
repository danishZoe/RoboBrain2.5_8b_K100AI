# 附：SGLang 服务化路径的兼容性排查（本节点不可用）

> **2026-10-27 17:40 修订**：本文根因描述已按厂商内部 PyPI 复查结果修正。
> 原表述「镜像内 `aiter` 是残缺构建」**不准确**——不是镜像装坏了，
> 而是**海光 aiter 分支在该 DTK 26.04 + torch 2.9.0 组合下，
> `aiter/ops/attention.py` 与 `aiter/ops/mha.py` 在 import 期就会抛 ValueError，
> 厂商因此在 `aiter/__init__.py` 里主动注释掉了这两条 star-import**。
> 厂商源里**所有** aiter 版本结构相同，换版本无效。详见 §6。

- 排查时间：2026-10-26 23:15 ~ 23:58（8 组后端组合探测）
- 修订时间：2026-10-27 15:00 ~ 17:40（联网后复查厂商内部 PyPI + 写兼容垫片实测）
- 目标：把 RoboBrain2.5-8B-NV 放到 SGLang 0.5.12+das.opt1.dtk2604 上做服务化，
  以获得连续批处理带来的并发吞吐收益
- 结论：**本节点上 SGLang 的全部可用 attention 后端均失败**，阻塞点在 DTK 内核栈，
  与模型、权重、提示词无关。服务化吞吐改由已验证的 vLLM 路径承担（见 `reports/09_vllm_adaptation.md`）。

---

## 1. 调用入口的版本差异

```bash
# 错误（本版本不存在该可执行文件）
setsid sglang.launch_server ...
# -> setsid: failed to execute sglang.launch_server: No such file or directory

# 正确
sglang serve --model-path ... --dtype bfloat16 \
             --mem-fraction-static 0.85 --context-length 8192 \
             --max-running-requests 16 \
             --limit-mm-data-per-request '{"image": 8}' \
             --trust-remote-code --attention-backend <BE> [--mm-attention-backend <MM>]
```

参数名差异（与上游文档/其他版本不同）：

| 上游写法 | 本版本 | 说明 |
|---|---|---|
| `sglang.launch_server` | `sglang serve` | CLI 结构改为子命令式（`serve` / `generate` / `version`） |
| `--limit-mm-per-prompt` | `--limit-mm-data-per-request` | 多模态数量上限参数改名 |
| `--enable-prefix-caching` | 无此参数 | **radix cache（前缀缓存）默认开启**，关闭用 `--disable-radix-cache` |
| `--mm-attention-backend` | 存在，可选 `sdpa / fa3 / fa4 / triton_attn / ascend_attn / aiter_attn / flashinfer_cudnn` | 视觉塔注意力后端 |
| `--model-impl` | 存在，可选 `auto / sglang / transformers / mindspore` | 可回退到 HF 模型实现 |

## 2. 后端逐项探测结果

探测方式：逐个启动服务 → 等待 `/v1/models` 就绪 → 发一次真实图像+文本请求 → 记录结果 → 关闭服务。
脚本：`scripts/probe_attn_backend.sh`；日志：`logs/probe_attn_*.log`。

| # | `--attention-backend` | `--mm-attention-backend` | `--model-impl` | 结果 |
|---|---|---|---|---|
| 1 | `aiter`（默认） | 默认 | sglang | ❌ 服务能起，**请求时 decode 崩溃**：`NameError: name 'paged_attention_ragged' is not defined`（`srt/layers/attention/aiter_backend.py:2937`） |
| 2 | `flashinfer` | — | sglang | ❌ 启动即退出：`flashinfer_python with minimum required version 0.6.11.post1 is not installed`（包未安装） |
| 3 | `wave` | — | sglang | ❌ 启动即退出：`ModuleNotFoundError: No module named 'wave_lang'` |
| 4 | `triton` | — | sglang | ❌ 初始化阶段崩溃：`Invalid address access: 0x7f8294b9c000, Error code: 3` → `Rank 0 scheduler died during initialization (exit code: -6)` |
| 5 | `torch_native` | 默认 | sglang | ❌ 服务就绪（190 s），**请求时崩溃**：`HSA_STATUS_ERROR_EXCEPTION ... code: 0x1016` / `KERNEL VMFault` |
| 6 | `torch_native` | `sdpa` | sglang | ❌ 同 5（同一 VMFault） |
| 7 | `aiter` | `sdpa` | sglang | ❌ 启动阶段退出 |
| 8 | `torch_native` | 默认 | `transformers` | ❌ 服务就绪（260 s），**请求时同样 VMFault** |

未测但客观不可用：`fa3` / `fa4`（需 flash-attn 3.x/4.x，本镜像只有 2.8.3）、
`flashinfer_cudnn`（需 flashinfer，未安装）、`ascend_attn`（昇腾硬件，无关）。

## 3. 根因

### 3.1 aiter 后端：`aiter/ops/attention.py` 与 `aiter/ops/mha.py` **在 import 期就会抛 ValueError**

SGLang 的 `aiter_backend.py` 在一个 `try: ... except ImportError:` 里导入：

```python
from aiter import (
    flash_attn_varlen_func, get_mla_metadata_info_v1, get_mla_metadata_v1,
    get_ps_metadata_info_v1, get_ps_metadata_v1, mha_batch_prefill_func,
    mla_prefill_ps_asm_fwd, mla_reduce_v1, paged_attention_ragged,
)
from aiter.mla import mla_decode_fwd, mla_prefill_fwd
from aiter.ops.triton.attention.unified_attention import unified_attention
```

镜像内 `aiter` 的实际情况（三重错配，**任何一条都会让整个 try 块失败**）：

| # | 错配 | 实测证据 |
|---|---|---|
| ① | **`aiter/ops/attention.py` 与 `aiter/ops/mha.py` 根本 import 不了** | 见下方 traceback |
| ② | `unified_attention` 实际在 `aiter/ops/triton/unified_attention.py`，SGLang 期望 `aiter/ops/triton/attention/unified_attention.py`（**少一层目录**） | `aiter/ops/triton/attention/` 下只有 `fp8_mqa_logits.py` 与 `pa_mqa_logits.py` |
| ③ | **6 个 MLA/PS 专用符号在海光 aiter 分支里没有实现**：`get_mla_metadata_v1` / `get_mla_metadata_info_v1` / `get_ps_metadata_v1` / `get_ps_metadata_info_v1` / `mla_prefill_ps_asm_fwd` / `mla_reduce_v1` | 全量扫描 `aiter` 包 0 命中 |

import 期失败的确切 traceback（`python3 -c "import aiter.ops.attention"`）：

```
File ".../aiter/ops/attention.py", line 54, in <module>
    def paged_attention_rocm(
File ".../aiter/jit/core.py", line 996, in decorator
    def custom_wrapper(*args, **kwargs):
File ".../aiter/jit/utils/torch_guard.py", line 248, in decorator
    schema = wrapper_register(calling_func)
File ".../aiter/jit/utils/torch_guard.py", line 228, in wrapper_register
    schema = torch.library.infer_schema(
File ".../torch/_library/infer_schema.py", line 190, in infer_schema
    ret = parse_return(return_annotation, error_fn)
ValueError: infer_schema(func): No return type annotation was provided. Please add one.
```

`@compile_ops` 装饰器在**定义时**就调用 `torch.library.infer_schema` 推导 schema，
而 `paged_attention_rocm` 没有返回类型注解 → 直接 ValueError。

**这也解释了 `aiter/__init__.py` 为什么把这两条 star-import 注释掉**：

```
50: # from .ops.enum import *
58: # from .ops.attention import *      ← 正是 attention
72: # from .ops.mha import *             ← 正是 mha
77: # from . import mla
```

厂商自己知道这两个模块在当前 torch 下导不进来，所以主动禁用了它们的顶层导出。
**SGLang 需要它们 → 必然 ImportError → except 分支只打印一句提示 →
所有符号保持未定义 → 第一次 decode 抛 `NameError`。**

**注意：服务能启动、权重能加载、端口能监听——这些都不代表服务可用。**

另外，`aiter/jit/build/` 下**没有** `module_pa_ragged` / `module_pa` / `module_attention` /
`module_attention_asm` / `module_mha` 的预编译产物，即便能 import，也仍需现场 JIT 编译
（依赖 hipcc + aiter csrc 源码 + 完整工具链），不在本任务范围内。

### 3.2 torch_native 后端：崩在 PyTorch HIP 的 SDPA 适配器里

崩溃前的最后一条 Python 帧：

```
/usr/local/lib/python3.10/dist-packages/transformers/integrations/sdpa_attention.py:92:
  UserWarning: sdpa adopt the new interface of flash-attn
  (Triggered internally at /pytorch/aten/src/aten/native/transformers/hip/cutlassfa_adapter.h:148.)
  attn_output = torch.nn.functional.scaled_dot_product_attention(
Callback: Queue 0x... aborting with error : HSA_STATUS_ERROR_EXCEPTION:
  An HSAIL operation resulted in a hardware exception. code: 0x1016
>>>>>>>> KERNEL VMFault !!!! <<<<<<
```

即：**HSA 层显存故障发生在 `F.scaled_dot_product_attention` 内部**
（DTK 的 cutlassfa 适配器）。第 8 组实验用了 `--model-impl transformers`
（完全走 HF 模型实现），崩溃点仍是 `transformers/integrations/sdpa_attention.py:92`
——说明**不是 sglang 自研模型代码的问题，而是这条 SDPA kernel 在 sglang 调用形态下
的形状/内存布局触发了硬件异常**。

对照证据：**同一条 SDPA 路径在独立 transformers 进程里完全正常**——
第三阶段基线的 6 条样例 × 3 次重复全部成功，最大视觉 token 935 条，无任何 VMFault。

## 4. 结论与处置

| 项 | 结论 |
|---|---|
| 模型 / 权重 / 提示词是否有问题 | **否**。同一权重在 transformers 与 vLLM 路径上跑通全部任务，输出格式与官方一致 |
| SGLang 是否可用于本节点 | **否**（当前镜像 + DTK 内核栈组合）。5 个可用后端全部失败，根因已定位到内核栈层面 |
| 阻塞是否可在本任务范围内修复 | **否**。见 §6：厂商源里所有 aiter 版本结构相同，换版本无效；可用组合需要 torch 2.10.0 栈，属新镜像/新实例 |
| 服务化吞吐怎么办 | 改用**已验证的 vLLM 路径**（`reports/09_vllm_adaptation.md`），本件按 vLLM 计件 |
| vLLM | `vllm 0.15.1+das.opt1.alpha.dtk2604.torch290.2605261031.gdb9e4b` 已跑通并完成全套验证 |

## 5. 复现方式

```bash
cd /root/private_data/workspace/robo && source ./env.sh

# 逐个探测后端组合（"main:mm:model_impl"，冒号可省）
bash scripts/probe_attn_backend.sh torch_native:sdpa aiter:sdpa torch_native::transformers

# 单次请求探针
python3 scripts/probe_one_request.py --port 30010 \
  --image samples/images/official_affordance.jpg --prompt "What is shown in this image?"
```

---

## 6. 2026-10-27 联网复查：厂商内部 PyPI 有 aiter，但**装了也救不了**

平台容器可通过 `source /root/private_data/workspace/robo/tools/env_net.sh` 获得公网访问
（代理文件实际在 `/root/private_data/.ai_user_info/ai_proxy`）。
厂商内部 PyPI 源：`https://pypi.sourcefind.cn/release/dtk2604/`（devpi，bases 为
`das_release/das1.8.1 / das1.8.2 / das1.8.4 / das1.8.5` + `open/pypi`）。

### 6.1 源里确实有 aiter（4 个版本）

| 版本 | DAS 线 | torch | 体积 |
|---|---|---|---|
| `0.1.5+das185.dtk2604.torch2100.2608131714.gb0979d` | das1.8.5 | 2.10.0 | 434 MiB |
| `0.1.3+das.opt1.dtk2604.torch2100.2606172003.g0a655d` | das1.8.4 | 2.10.0 | — |
| `0.1.3+das.opt1.dtk2604.torch290.2605211823.gccde69` | das1.8.2 | **2.9.0** | 293 MiB |
| `0.1.2+das.opt1.dtk2604.torch290.2605141109.g76b669` | das1.8.1 | **2.9.0** | — |

镜像内已装：`0.1.3+das.opt1.dtk2604.torch290.2606011547.gf165f1`（torch 2.9.0）。

### 6.2 但换版本无效 —— 三项逐一对拍

| 对拍项 | 镜像内已装 | 源里 `das.opt1+torch290` | 源里 `das185+torch2100` |
|---|---|---|---|
| 有无 `aiter/attention/` 子包 | ❌ | ❌ | ❌ |
| `ops/attention.py` 的 `paged_attention_rocm` 有无返回注解 | ❌ | ❌ | ❌ |
| 6 个 MLA/PS 符号（`get_mla_metadata_v1` 等） | ❌ | ❌ | ❌ |
| `ops/triton/unified_attention.py` | ✅ | ✅ | ✅ |
| `aiter/mla.py`（`mla_decode_fwd` / `mla_prefill_fwd`） | ✅ | ✅ | ✅ |

**三个 aiter 版本结构完全相同** → 换 aiter 版本解决不了 §3.1 的三重错配。

### 6.3 厂商自己的 SGLang 构建也依赖同一套缺失 API

| sglang 版本 | DAS 线 | torch | `aiter_backend.py` 的导入块 |
|---|---|---|---|
| `0.5.12+das.opt1.dtk2604`（镜像内已装） | das.opt1 | 2.9.0 | 与 §3.1 **完全相同的 9 符号 + 3 子模块** |
| `0.5.10rc0+das.opt2.alpha.dtk2604.torch290.2605181752.g91be88` | das.opt2 | **2.9.0** | 同上（已下载解包核对） |
| `0.5.12+das185.dtk2604.torch2100.2608121934.gcbf025` | das185 | 2.10.0 | 同上 |

→ 厂商的 SGLang 构建与厂商自己的 aiter 分支之间存在**接口口径不匹配**，
不是我们环境装坏了。

### 6.4 其他可用后端的依赖也不在本节点可达范围

| 后端依赖 | 源里有没有 | 判定 |
|---|---|---|
| `flashinfer-python` | 有（0.2.9 ~ 0.7.2.post1） | **来自 `open/pypi` 的上游 CUDA 轮子**，非 DTK 自建 → DCU 不可用 |
| `wave-lang` | **源里根本没有** | 不可用 |
| `triton` | 源里有 triton，但本节点初始化即 `Invalid address access` | 不可用 |

### 6.5 唯一可能的路径：torch 2.10.0 + das185 栈（需新实例）

厂商在 das1.8.5 上成套发布了 `torch 2.10.0` + `aiter 0.1.5+das185` +
`sglang 0.5.12+das185` + `sglang-kernel 0.4.2.post2+das185` + `sglang-router`。
这条栈上 `torch.library.infer_schema` 的行为可能与 torch 2.9.0 不同，
**厂商自建镜像里大概率可用**。

但这意味着：

- 需要**新建实例**（容器镜像在创建时固化，运行中不可更换）；
- 需要**新的卡时**；
- 即使可用，也是**另一个计件**（框架维度独立计件），需要重跑整套验收。

**本任务范围内无法完成**，如实记录，不做无依据的乐观假设。

### 6.6 已验证的兼容垫片（保留，供新实例参考）

`envs/sglang_shim/sitecustomize.py`（配合 `tools/try_sglang_aiter.sh`）：

- 不改 `site-packages`，不改权重/采样/注意力数学实现；
- A. 把**已存在**的符号显式 re-export 到 `aiter` 顶层；
- B. 把 `aiter.ops.triton.unified_attention` 注册为
  `aiter.ops.triton.attention.unified_attention` 的别名；
- C. 对 6 个缺失的 MLA/PS 符号注入**占位**，只在**真被调用**时抛
  `NotImplementedError`（不静默降级）。

实测（stage0，导入期）：

```
aiter.flash_attn_varlen_func     MISSING     ← 被 ① 阻断（模块导不进来）
aiter.paged_attention_ragged     MISSING     ← 被 ① 阻断
aiter.mha_batch_prefill_func     MISSING     ← 被 ① 阻断
aiter.get_mla_metadata_v1        placeholder ← ③ 占位
aiter.mla.mla_decode_fwd / mla_prefill_fwd : OK
aiter.ops.triton.attention.unified_attention : OK
```

**结论：垫片能补 ②③，但补不了 ①**（`aiter/ops/attention.py` 在本 torch 下无法 import，
JIT schema 推断在装饰期就抛 ValueError，非导入路径问题）。
→ 在 torch 2.9.0 栈上，SGLang 仍不可用；垫片仅在 das185/torch 2.10.0 新栈上可能有意义。
