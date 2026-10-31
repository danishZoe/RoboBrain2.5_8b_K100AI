# 第一阶段：环境与模型检查报告

- 检查时间：2026-10-26 22:15 ~ 22:50 (UTC+8)
- 检查节点：`crdnotebook-2114695384420274177-scnjz2s4uc-68757`
- 连接方式：Windows 本机 `ssh dcu_1026` → `ksai.scnet.cn:10312`，用户 `root`
- 原始日志：`logs/env_check_20261026_221511.log`

---

## 1. 主机 / 容器

| 项 | 实测值 |
|---|---|
| 运行形态 | **容器内**（`/.dockerenv` 存在，cgroup 为 `kubepods.slice/.../cri-containerd-78036a1e...scope`） |
| 宿主平台 | Kubernetes Pod（notebook 型开发机） |
| 操作系统 | Ubuntu 22.04.5 LTS |
| 容器内 docker | **不可用**（无 docker 二进制），无需也无需自行拉起镜像 |
| 平台自带入口 | Jupyter Lab 运行在 `0.0.0.0:8888`（pid 1 启动），另有 sshd |
| CPU | 32 核 |
| 内存 | 503 GiB 总量，约 437 GiB 可用 |
| 磁盘 | `ks_p300s_public` 61P（已用 46P，剩 16P），`/root/private_data` 与 `/public/home/scnjz2s4uc` 同一挂载 |

> 结论：**已处于用户提供的 `jupyter-sglang5.12-ubuntu22.04-dtk26.04-py3.10-qwen3.6:v1.0` 镜像内部**。
> 镜像内的 sglang 0.5.12 + Python 3.10 + DTK 26.04 与镜像名一致，无需额外启动容器。

## 2. 加速卡 / 驱动 / 运行时

| 项 | 实测值 |
|---|---|
| `hy-smi` | HCU 0，51.0 °C，126 W / 300 W 上限，VRAM 0%，HCU 0.0%，Mode Normal（空闲） |
| 设备名（torch 视角） | **`K500SM_AI`**（= 平台卡型 K100AI，见 `07_device_model_mapping.md`） |
| 架构 codename | `gfx928:sramecc+:xnack-`（KFD `gfx_target_version 90208`，节点名 `kongminge`） |
| 计算能力 | `(9, 2)` |
| 显存 | **63.98 GiB**（检查时空闲 63.19 GiB） |
| 多处理器数 | 120 |
| bf16 支持 | **True** |
| DTK 版本 | `hipconfig --version` → **6.3.26113** |
| 运行时栈 | `/opt/dtk`（hip/rocblas/rocfft/hipdnn/miopen/rccl/...）+ `/opt/hyhal`（驱动侧） |
| 设备节点 | `/dev/kfd`、`/dev/dri/card2`、`/dev/dri/renderD129` |

> **设备型号已核实（2026-10-27 补充）**：AGENTS.md 写"目标硬件：K100AI"，
> 运行时 Card Series 字符串为 **`K500SM_AI`**。二者指**同一机型**：架构 codename `gfx928`、
> KFD 节点名 `kongminge`、单卡 64 GB 三项与 proj_v1 已调优的 K100_AI 节点逐项吻合。
> 完整映射说明与复现命令见 `reports/07_device_model_mapping.md`。
>
> 统一写法：**K100AI（运行时 Card Series: `K500SM_AI`，gfx928，64 GB）**。

> **接口名提示**：torch 侧暴露的是 `torch.cuda.*`，但它映射的是天数 HCU 而非 NVIDIA GPU。
> 已通过实际张量计算验证该命名空间在本机可用（见第 4 节），未依赖接口名做推断。

## 3. Python 组件版本（镜像自带）

| 包 | 版本 |
|---|---|
| Python | 3.10.12 |
| torch | `2.9.0+das.opt1.dtk2604.2605281139.gd0fc8c` |
| torchvision / torchaudio | 0.24.0 / 2.9.0（均为 dtk2604 定制版） |
| triton | `3.5.1+das.opt1.dtk2604` |
| transformers | 5.3.0 |
| vllm | `0.15.1+das.opt1.alpha.dtk2604.torch290` |
| sglang / sglang-kernel | `0.5.12+das.opt1.dtk2604` / `0.4.2.post2` |
| flash-attn / flash-mla | 2.8.3 / 1.2.0（dtk2604 定制版） |
| accelerate | 1.13.0 |
| av / pillow / numpy | 16.1.0 / 11.3.0 / 1.25.0 |
| cv2 | 4.10.0 |
| 已缺失 | `qwen_vl_utils`、`matplotlib`、`decord`、`gradio` |

## 4. 设备可用性实测

```
torch.cuda.device_count()          -> 1
torch.cuda.get_device_name(0)      -> K500SM_AI
torch.cuda.get_device_capability(0)-> (9, 2)
torch.cuda.is_bf16_supported()     -> True
torch.cuda.mem_get_info()          -> free 63.19 GiB / total 63.98 GiB
```

在 `cuda:0` 上执行 2048³ bf16 矩阵乘：结果 **finite=True**，无设备错误。
（该尺寸仅 4.7 TFLOP 量级、计时被首调用开销主导，**不足以代表算力峰值**，
真实吞吐以第三/四阶段的模型实测为准。）

## 5. 模型权重与配置

只读目录：`/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV`

| 项 | 实测值 |
|---|---|
| 分片 | `model-00001-of-00002.safetensors` 9.91 GB + `model-00002-of-00002.safetensors` 7.62 GB |
| index 校验 | index 声明 2 个分片，实际 2 个，**MISSING: none** |
| 权重总大小 | index metadata 16.33 GiB（bf16） |
| 架构 | `Qwen3VLForConditionalGeneration` / `model_type: qwen3_vl` |
| 精度 | `bfloat16`（与硬件 bf16 支持一致） |
| 文本侧 | hidden 4096、36 层、32 heads / 8 KV heads、head_dim 128、vocab 151936 |
| 上下文 | `max_position_embeddings` 262144 |
| 视觉侧 | depth 27、hidden 1152、patch 16、merge 2、deepstack [8,16,24] |
| 处理器 | `Qwen3VLProcessor` / `Qwen2VLImageProcessorFast`；`processor_class=Qwen3VLProcessor` |
| 视频组件 | `video_preprocessor_config.json` 存在 |
| 聊天模板 | `chat_template.json`（标准 Qwen3-VL 模板，支持 system / tools） |
| 自定义代码 | **无**（`configuration.json` 仅声明 `framework: pytorch`），无需 `trust_remote_code` 才能加载权重 |

## 6. 后端支持矩阵

| 后端 | 状态 | 证据 |
|---|---|---|
| transformers 5.3.0 | ✅ 支持 | `transformers.models.qwen3_vl` 存在，含 `Qwen3VLForConditionalGeneration` / `Qwen3VLProcessor` |
| vLLM 0.15.1 (dtk2604) | ✅ 注册 | `_MULTIMODAL_MODELS` 含 `Qwen3VLForConditionalGeneration`、`Qwen3VLMoeForConditionalGeneration` |
| SGLang 0.5.12 (dtk2604) | ✅ 支持 | `srt/models/qwen3_vl.py` 存在，`EntryClass = Qwen3VLForConditionalGeneration` |

## 7. 网络可达性（决定离线策略）

| 目标 | 结果 |
|---|---|
| github.com | 超时（rc=124） |
| huggingface.co | 超时（rc=124） |
| pypi.org | 超时（rc=124） |

DNS 可解析，但 **HTTP 出网被完全阻断**。因此：

- 不能 `pip install`、不能 `git clone`、不能下载模型/样例图；
- **Windows 本机可出网**，改为在本机 `agent_work` 克隆官方仓库并下载 Linux wheel，
  再用 `scp` 传入远端离线安装（见第 8 节）。

## 8. 为补齐依赖所做的变更（全部在��作目录内）

官方 `requirements.txt` 是 NVIDIA 栈（`torch==2.5.1` / `vllm==0.7.3` / `nvidia-*-cu12`），
与本机 DTK 栈不兼容，**未采用**。仅补齐镜像中确实缺失、且为纯 CPU 侧的依赖：

| 包 | 版本 | 安装位置 | 理由 |
|---|---|---|---|
| qwen-vl-utils | 0.0.14 | `envs/pylibs` | 官方 `inference.py` 用其 `process_vision_info` 处理图像 |
| matplotlib | 3.10.7 | `envs/pylibs` | Robo-Dopamine 的奖励曲线可视化 |
| fonttools / kiwisolver / contourpy / cycler | 4.60.1 / 1.4.8 / 1.3.2 / 0.12.1 | `envs/pylibs` | matplotlib 依赖 |
| pyparsing | 3.2.3 | `envs/pylibs` | 镜像自带 2.4.7，matplotlib 3.10 需要 ≥3.0 |

安装方式：`pip install --no-deps --no-index --target envs/pylibs`，**未触碰系统 site-packages**，
通过 `env.sh` 的 `PYTHONPATH` 注入。公开 Python 环境、模型权重、驱动均未修改。

官方源码镜像（只读参考，运行时不直接依赖）：

- `third_party/RoboBrain2.5/` — `inference.py`（任务提示词模板与输出解析的**唯一权威来源**）、`assets/demo/*.jpg` 官方样例图
- `third_party/Robo-Dopamine/` — `examples/inference.py`（GRM 时间价值估计流水线）、`examples/demo_table/*.mp4` 三路视角视频

## 9. 推理路径选择

选择 **transformers 直跑** 作为基线路径，依据：

1. 改动最少——模型本身无需任何修改即可加载（无自定义代码、bf16 与硬件原生匹配）；
2. 已在 K500SM_AI 上实测跑通端到端图像+文本生成（见第二阶段）；
3. transformers 5.3.0 对 `qwen3_vl` 的支持与官方 `inference.py` 的调用方式完全一致
   （`AutoModelForImageTextToText` + `AutoProcessor` + `apply_chat_template`）。

SGLang / vLLM 作为**第四阶段的性能对比路径**，不作为跑通前提。

## 10. 已知限制

- 外网阻断：无法拉取官方 demo 图以外的更多样例；已用官方 demo 图 + 自绘合成图各一套。
- `decord` 缺失：Robo-Dopamine 的视频解码走 `cv2.VideoCapture`，不受影响；若某路径强制要 decord 需另行离线补包。
- `hy-smi --showpids` 在容器内不可用（无法列出其他进程占用），改用 `torch.cuda.mem_get_info()` 判断显存占用。
- 容器内 `/root/private_data/workspace/model/MiMo-V2.6-Distill-Qwen-9B` 是**上一次任务遗留的未完成下载**（仅 3.3 MB 配置），与本任务无关，未改动。
