# RoboBrain2.5-8B-NV on K100-AI + vLLM 0.15.1

> 本文按官方《技术报告模板》的八章结构撰写。所有数字来自本节点实测原始文件
> （`env.log` / `server.log` / `smoke.log` / `verify_result.json` / `bench/*.jsonl`），
> 未做任何删减或估算；无法测量的项目一律标注原因。

## 1. 简介

本文档记录 `BAAI/RoboBrain2.5-8B-NV` 在 `K100-AI + vLLM 0.15.1`（精度 `bf16`）环境的适配与验证结果。

RoboBrain2.5-8B-NV 是基于 **Qwen3-VL** 架构的具身多模态模型，约 8.77B 参数，
由 27 层 ViT 视觉塔 + 文本解码器组成，支持四类任务：
`general`（场景描述）、`grounding`（目标框）、`pointing`（可指点/空间指代）、
`trajectory`（3D 末端轨迹）。权重为 BF16 两个分片，共约 17.5 GB。

**与官方部署方式的核心差异（两条，均已在第 3 节给出可复现步骤）**：

1. **vLLM × transformers 的 config API 错位**：vLLM 0.15.1 从**子 config** 读取
   `tie_word_embeddings`，而 transformers 5.3.0 只在**顶层复合 config** 保留该字段，
   导致引擎在加载权重时抛 `AttributeError` 而启动失败。改为用 `sitecustomize` 垫片注入
   （等价的一行 patch 见 `patches/0001-vllm-qwen3-tie-word-embeddings.patch`）。
2. **图像预处理必须与官方 upstream 一致**：官方 `inference.py` 经 `qwen_vl_utils` 取图，
   对图像做**两次重采样**（先按 28 倍数、processor 再按 32 倍数）；
   直接上传原图会让服务端看到**不同的像素**（归一化空间最大差 0.47），
   足以翻转 grounding 首个生成 token，官方正则随即判为不可解析。
   客户端用 `scripts/image_canonical.py` 规范化后以 PNG 上传即可逐位对齐（最大差 0.000000）。

相关获取地址：

- 权重：超算互联网 AI 社区模型库（本任务分配路径见第 2 节）
- 镜像：`jupyter-sglang5.12-ubuntu22.04-dtk26.04-py3.10-qwen3.6:v1.0`
- 提交仓库：代码仓库 `https://developer.sourcefind.cn/codes/g_exhyd6h6/robobrain2.5_8b_k100ai`（光合开发者社区 GitLab，SSH 地址 `ssh://git@developer.sourcefind.cn:2222/g_exhyd6h6/robobrain2.5_8b_k100ai.git`）；分支 `main` = **最优版**，标签 `baseline-submission` = **首提基线（eager）**；本次交付代码 revision = `bc44891`（完整 `bc44891f178eeb1f765bbc0e4bcf1863eee97225`，379 个跟踪文件，逐文件校验值见 `MANIFEST.sha256`，详见 `.revision`；重构前的完整工作区保留在 `workbench` 分支 `e8930a6`）

参考文档：

- 上游模型说明与推理示例：`RoboBrain2.5/README.md`、`RoboBrain2.5/inference.py`

仓库内容：

| 文件 | 作用 | 产出 |
| --- | --- | --- |
| `env.sh` | 模型、镜像、框架、卡型、并行方式与启动参数，其余脚本读这里的变量 | — |
| `env_check.sh` | 采集卡型与重点组件版本 | `env.log` |
| `requirements-extra.txt` / `patches/` | 新增的包、代码改动 | — |
| `run.sh` | 模型启动脚本：装依赖 + 准备适配改动 + 拉起服务，轮询 `/health` | `server.log` |
| `smoke.sh` | Smoke 测试：健康检查、单轮 / 多轮 / 流式、四类任务、异常输入 | `smoke.log` |
| `verify.py` | 精度验证脚本：按官方基线文件跑评测并判定 | `verify_result.json` |
| `bench.sh` | 性能测试脚本：6 档负载，每档连测两次 | `bench/` |
| `reproduce.sh` | **一键复现入口**：依次执行 `run.sh → env_check.sh → smoke.sh → verify.py → bench.sh`，逐步打印真实退出码 | 全部产物 |

一键复现：

```bash
git clone ssh://git@developer.sourcefind.cn:2222/g_exhyd6h6/robobrain2.5_8b_k100ai.git \
  && cd robobrain2.5_8b_k100ai
bash reproduce.sh                          # 一键跑完下面 5 步（等价于逐条执行）
```

或逐步执行：

```bash
bash run.sh                             # 装依赖 + 准备适配改动 + 起服务（重复执行自动跳过已完成的步骤）
bash env_check.sh                       # 记录验证环境 → env.log
bash smoke.sh                           # Smoke 测试 → smoke.log
source env.sh && python verify.py       # 精度测评 → verify_result.json
BENCH_MODE=image bash bench.sh          # 性能测试 → bench/
```

### 1.1 仓库目录结构

仓库根目录即为**提交包**（官方模板要求的技术报告与全部脚本都在根）：

| 路径 | 内容 |
| --- | --- |
| `README.md` | **本技术报告**（官方模板 8 章） |
| `env.sh` / `env_check.sh` | 提交参数（模型/镜像/框架/卡型/并行/启动参数）与环境采集 → `env.log` |
| `requirements-extra.txt` / `patches/` | 新装依赖与代码改动（无上游写入，走 `sitecustomize` 垫片） |
| `run.sh` / `smoke.sh` / `verify.py` / `bench.sh` / `reproduce.sh` | 启动 / Smoke / 精度 / 性能 / 一键复现 |
| `env.log` `server.log` `smoke.log` `verify_result.json` | 各步骤原始产物 |
| `bench/` `bench_public/` `bench_verify/` `bench_pitfall_ignore_eos/` | 官方六档压测原始结果、公开基准产物、输出 token 计量对照、踩坑留证 |
| `baselines/` | 官方基线文件放置位（**当前为空**：官方未下发，见第 6 节） |
| `rocblas_tuned/` | GEMM 算子调优库（最优版资产，6 个差异文件，哈希见 `DIFF_FILES.sha256`） |
| `envs/vllm_shim/` | `sitecustomize` 垫片（config API 适配，见 3.3） |
| `reports/` | 交付证据链（52 份过程报告，含 `25_tuning_delivery_20261031.md` 完整六档与同卡 A/B） |
| `scripts/` `tools/` | 复现与核验工具链（压测、同卡判定、报告注入、提交包自检） |
| `samples/` `platform/` `third_party/` | 固定样例与校验值、平台截图位（待补）、上游推理示例与许可 |
| `docs/` | **归档**：工作区索引 `WORKSPACE_INDEX.md`、被裁下的一次性调试探针与样例叠加图 |
| `MANIFEST.sha256` | 提交物逐文件 sha256（173 条），用于防篡改核验 |

> **裁剪说明**：`scripts/` 与 `tools/` 中的一次性调试探针（`probe*` / `diag*` / `show*` /
> `peek*` / `dump*` 等，共 250 余个）已从主干移出，完整版本保留在归档分支 `workbench`
> 与 `docs/`，不丢失可追溯性。

## 2. 验证环境

镜像：`jupyter-sglang5.12-ubuntu22.04-dtk26.04-py3.10-qwen3.6:v1.0`

重点组件（取自 `env.log`）：

| 组件 | 版本 | 备注 |
| --- | --- | --- |
| DTK | `/opt/dtk`（`6.3.26113`） | 镜像自带 |
| torch | `2.9.0+das.opt1.dtk2604.2605281139.gd0fc8c` | 镜像自带 |
| triton | `3.5.1+das.opt1.dtk2604.torch290.2605281608.g799061` | 镜像自带 |
| vllm | `0.15.1+das.opt1.alpha.dtk2604.torch290.2605261031.gdb9e4b` | 镜像自带 |
| transformers | `5.3.0` | 镜像自带（**未升级**） |
| tokenizers | `0.22.2` | 镜像自带 |
| sglang | `0.5.12+das.opt1.dtk2604` | 镜像自带，本节点不可用（见第 8 节） |
| sglang-kernel | `0.4.2.post2` | 镜像自带 |
| torchvision | `0.24.0+das.opt1.dtk2604.torch290.2604091152.g66069c` | 镜像自带 |
| ray | `2.55.1` | 镜像自带 |
| qwen-vl-utils | `0.0.14` | **新装**，隔离在 `$WORK_DIR/envs/pylibs` |

- 卡型：`K100-AI`（运行时实际名称 `K500SM_AI`，架构 `gfx928`；官方 `env.sh` 明文「K500-SM 按 K100-AI 计」）
- 卡数：`1`（单机 TP=1；官方脚手架默认 TP=4）
- 模型路径：`/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV`
- 服务端口：`30000`

环境自检（`env.log` 关键部分原文）：

```text
image: jupyter-sglang5.12-ubuntu22.04-dtk26.04-py3.10-qwen3.6:v1.0
dtk: /opt/dtk
model_id: BAAI/RoboBrain2.5-8B-NV
served_name: robobrain2.5-8b-nv
framework: vllm  card: K100-AI  precision: bf16
tp: 1  max_len: 32768  mem_frac: 0.85  port: 30000
--- hy-smi ---
HCU     Temp     AvgPwr     Perf     PwrCap     VRAM%      HCU%      Mode
0       49.0C    133.0W     manual   300.0W     27%        0.0%      Normal
--- runtime ---
python: 3.10.x
torch: 2.9.0+das.opt1.dtk2604.2605281139.gd0fc8c
HIP: 6.3.xxxxx
Device: K500SM_AI x 1
--- packages ---
torch                 2.9.0+das.opt1.dtk2604.2605281139.gd0fc8c
triton                3.5.1+das.opt1.dtk2604.torch290.2605281608.g799061
transformers          5.3.0
tokenizers            0.22.2
vllm                  0.15.1+das.opt1.alpha.dtk2604.torch290.2605261031.gdb9e4b
sglang                0.5.12+das.opt1.dtk2604
sglang-kernel         0.4.2.post2
ray                   2.55.1
--- model config facts ---
architectures: ['Qwen3VLForConditionalGeneration']
tie_word_embeddings: False
vision depth: 27
n tensors: 750
shards: ['model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors']
lm_head present: True
--- container limits ---
cgroup memory limit: 123480309760 bytes
```

> `tie_word_embeddings: False` 与 `lm_head present: True` 互证「词嵌入未绑定」，
> 这是第 3.3 节垫片取值的依据，不是猜的。

超算互联网平台：

| 项 | 值 |
| --- | --- |
| 区域 | —（**待用户从超算互联网控制台补充**：本任务为 SCNet 容器节点，区域名形如 `ksai.scnet.cn` 对应的区域） |
| Notebook 规格 | CPU 6 核 / 内存 115 GiB（cgroup 上限 `cgroup memory limit: 123480309760 bytes`）/ 系统盘 —、文件存储 —（**待用户从 Notebook 创建页补充**） |
| 加速卡与显存 | `K100-AI`（运行时名 `K500SM_AI`，`gfx928`）×**1**；64 GB |
| 开发镜像 | `appstore.scnet.cn:5000/aihub/dcu/zhr5535/jupyter-sglang5.12-ubuntu22.04-dtk26.04-py3.10-qwen3.6:v1.0`（镜像名 `jupyter-sglang5.12-ubuntu22.04-dtk26.04-py3.10-qwen3.6:v1.0`） |
| 模型获取方式 | AI 社区模型库（只读路径） |
| 服务映射 | 后端 `127.0.0.1:30000` → 代理 — → 外部 —（**未对外暴露**：本节点未配置服务代理映射，服务仅在容器内 `127.0.0.1:30000` 供压测与评测使用） |
| 平台证据 | `platform/`（Notebook 创建页、环境自检、模型管理、服务映射）—— **待用户截图补入**，所需 4 张截图的清单见 `platform/README.md` |

## 3. 适配优化方案

### 3.1 基础镜像

基于 `jupyter-sglang5.12-ubuntu22.04-dtk26.04-py3.10-qwen3.6:v1.0`。
选它的原因：这是本任务可用的唯一同时带 **vLLM 0.15.1（dtk2604 插件版）** 与 **DTK 26.04** 的镜像；
同镜像自带的 SGLang 0.5.12 在本节点因镜像内 `aiter` 残缺构建而不可用（第 8 节）。

### 3.2 软件包升级

**本次没有升级任何核心依赖。** torch / transformers / vllm / triton 全部沿用镜像内版本，
因此不存在「依赖解析把国产加速卡版 torch 换成 CUDA 版」的风险。

唯一新增的是官方 upstream 推理链路需要的图像加载库：

| 包 | 镜像内版本 | 升级后版本 | 来源 | 原因 |
| --- | --- | --- | --- | --- |
| `qwen-vl-utils` | 无 | `0.0.14` | 离线 wheel | 官方 `inference.py` 的 `process_vision_info()` 依赖它 |

已写入 `requirements-extra.txt`，由 `run.sh` 以 `--no-deps --target $WORK_DIR/envs/pylibs` 安装
（装在**工作目录内的隔离前缀**，不污染公共 site-packages）：

```text
qwen-vl-utils==0.0.14      # 镜像未自带 → 新装：官方 inference.py 的 process_vision_info 依赖
matplotlib==3.10.7         # qwen-vl-utils 的连带依赖
contourpy==1.3.2           # matplotlib 的连带依赖
cycler==0.12.1             # matplotlib 的连带依赖
fonttools==4.60.1          # matplotlib 的连带依赖
kiwisolver==1.4.8          # matplotlib 的连带依赖
pyparsing==3.2.3           # matplotlib 的连带依赖
```

> **另有一处本地缓存适配（`bench.sh` 内自动完成）**：官方压测客户端的 image 数据集需要
> `AutoProcessor.from_pretrained(model_id)` 加载 processor，而 `model_id` 取自
> `--served-model-name`（服务名，不是路径），节点无外网会直接失败。
> `bench.sh` 在**工作目录内的 HF 缓存**里为服务名建一个指向本地权重的只读快照软链接
> （`$HF_HOME/hub/models--<served-name>/snapshots/main -> $MODEL_PATH`），
> 使 processor 能离线解析。**不触碰模型目录本身**，也不写公共环境。

### 3.3 代码改动

#### `patches/0001-vllm-qwen3-tie-word-embeddings.patch`

- 作用于：`vllm/model_executor/models/qwen3.py`（路径相对 `PATCH_ROOT`）
- 改了什么：把 `self.config.tie_word_embeddings` 改成
  `getattr(self.config, "tie_word_embeddings", False)`
- 为什么改：多模态模型里 `self.config` 是**文本子 config**（`Qwen3VLTextConfig`），
  transformers 5.3.0 把 `tie_word_embeddings` 挪到了**顶层复合 config**，
  子 config 上没有该属性 → `AttributeError: 'Qwen3VLTextConfig' object has no attribute
  'tie_word_embeddings'` → 引擎初始化失败
- 精度影响：**无**。取值仍为 `False`（由 `config.json` 顶层声明与权重索引中
  `lm_head.weight` 存在双向确认），语义与模型自身声明一致，不改变任何权重或计算

```diff
--- a/vllm/model_executor/models/qwen3.py
+++ b/vllm/model_executor/models/qwen3.py
@@ -394,7 +394,7 @@ class Qwen3ForCausalLM(SupportsLoRA):
                 skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
```

> **本节点的实际生效方式**：公共 site-packages 是只读环境，不能落 patch。
> 因此运行时用 `envs/vllm_shim/sitecustomize.py` 垫片在**每个解释器启动时**
> （含 vLLM 用 `spawn` 起的 EngineCore 子进程）给 `Qwen3VLTextConfig` 补上该属性，
> 值同样由模型自身的 `config.json` + 权重索引推导，不写死。
> 两者语义等价；`patches/` 里的 diff 用于提交上游仓库。

#### `scripts/image_canonical.py` + `scripts/answer_adapter.py`（输入/输出两侧的适配，非上游改动）

这两处是**我们这一侧的调用约定**，不是对 vLLM 的修改，但它们直接决定精度结果，必须写清：

| 文件 | 作用 | 为什么必须 |
| --- | --- | --- |
| `scripts/image_canonical.py` | 用上游 `qwen_vl_utils.fetch_image` 预缩放，PNG 无损上传 | 官方管线对图像做**二次重采样**；直接上传原图与服务端只做一次缩放，像素最大差 **0.47**（归一化 ±1），翻转 grounding 首个 token |
| `scripts/answer_adapter.py` | 把 `<bbox>x1 y1 x2 y2</bbox>` 改写成 `[x1, y1, x2, y2]` | 官方正则只认方括号形式；适配后**四个整数原样搬运**，不取整/裁剪/重排，原始文本单独保留 |

### 3.4 升级步骤

`run.sh` 启动前会自动完成以下步骤（重复执行时跳过已装的包）；手动复现时按顺序执行：

```bash
# 1) 新增依赖（隔离前缀，不动公共环境）
pip install -q --no-deps --target "$WORK_DIR/envs/pylibs" -r requirements-extra.txt

# 2) 适配改动：本节点**不写入公共 site-packages**（任务红线），改由 sitecustomize 垫片注入
#    （patches/ 下的 diff 仅用于提交上游仓库）
export ROBO_VLLM_TIE_SHIM=1
export ROBO_MODEL_PATH="$MODEL_PATH"
export PYTHONPATH="$WORK_DIR/envs/vllm_shim:$PYTHONPATH"

# 3) 运行时
source /opt/dtk/env.sh

# 4) 组装 GEMM 算子调优库到工作目录（幂等；公共 /opt/dtk 库只读不动）
bash scripts/make_tuned_rocblas_lib.sh
```

### 3.5 性能调优方案（最优版）

首提基线是 `--enforce-eager`（eager 模式）。在同卡型、同框架、同负载下做了
E1–E9 共 9 组单变量实验后，采纳了其中 2 项改动，构成当前提交配置：

| 开关 | 位置 | 作用 |
| --- | --- | --- |
| 去掉 `--enforce-eager` | `env.sh` 的 `EXTRA_ARGS` | 启用 torch.compile + graph capture（2026-10-31 节点同卡实测（换卡后未重跑 graph 基线，属跨卡存档），4096/并发1 首提 eager 21.74 → graph 基线 22.38 tok/s，+2.9%；单项收益有限，主体收益来自 GEMM 调优库与 aiter FA） |
| `ROCBLAS_TENSILE_LIBPATH` | `env.sh` → `$WORK_DIR/rocblas_tuned/library_gpu4` | **GEMM 算子调优库**：用厂商 `auto_select_tools` 在本机实测 shape 上重新选优的 Tensile 选优表 |
| `VLLM_ATTENTION_BACKEND=ROCM_AITER_FA` | `env.sh` | 长上下文 decode 走 aiter FA kernel（10-30 同轮 E7 gate 隔离测得：16384/并发1 +17.9%；本节点未单独隔离该项） |

#### 3.5.0 换卡与「同卡 A/B」口径（2026-11-01 起强制）

SCNet 容器被平台回收重建后**物理卡会更换**（实测同配置跨卡可差约 30%）。
因此本提交包把「同卡 A/B」写进了工具链，而不是只写在报告里：

| 机制 | 位置 | 作用 |
|---|---|---|
| 卡指纹采集 | `scripts/card_identity.sh` | 取 GPU unique id（取不到则退化为 主机名+boot_id），写进 `outputs/CARD_ID.txt` |
| 指纹落盘 | `scripts/run_full_bench.sh` | 每次压测把 `card_id=` 写进 `<run-dir>/run_config.txt` |
| 同卡判定 | `scripts/bench_set.py` | 只把 `card_id` 与当前节点一致的目录算作同卡；没有 `card_id` 的老目录**一律只作存档**，绝不混进同卡 A/B |
| 三套同卡压测 | `scripts/run_same_card_bench.sh` | 同一张卡上依次跑 eager → graph 基线 → 最优版 |

配套的三套原始数据目录：`bench/tuned/`（最优版）、`bench/same_card_eager/`（首提基线 eager）、
`bench/same_card_baseline/`（graph 基线）。

> 结论口径：**只有同一张卡上的两个数才能相减。** 跨卡绝对值只作过程存档。

#### 3.5.1 GEMM 算子调优库：来源、组装与生效证据

- **来源**：2026-10-30 在本机用厂商 `auto_select_tools` 采集真实业务 shape（rocBLAS layer3
  trace，218 个唯一 bf16 GEMM size）后做选优，产出与 stock 不同的 Tensile 选优表。
  算子级对比：107 个 size 中 104 个变快、1 个持平、2 个微降（≤0.2%），中位 **+19.1%**，
  LM head（并发1 形状）+83%。机制是 decode 形状（N=1~64）由通用 MMAC kernel
  换成 GEMV 专用 kernel。
- **与 stock 的确切差异**：只有 4 个文件被改写、2 个新增，其余逐字节相同。
  这 6 个文件已随仓库提交在 `rocblas_tuned/library_gpu4/`：

  | 文件 | SHA256 | 状态 |
  | --- | --- | --- |
  | `GemmGenerality_AI_Eco.dat` | `ad891f317f5c097776e014d738f17e84d2e4af7dfc7ad352f01f0d60d28025f3` | 改写 |
  | `GemmGeneralityAdd_Eco.dat` | `23e30c02a186d0c9a0cc851c2874752f1fc3f7fdd89ba77bf2cd5c5806324f58` | 改写 |
  | `GemmGenerality_SolutionName_AI_Eco.txt` | `5509bf83b23f961c17b9a60021a2be070f431835b8920f34cedeac079697b1c3` | 改写 |
  | `GemmGeneralityAdd_SolutionName_Eco.txt` | `8b5b348f95a93ab7f852a816389cc2e34f48806a96dd02b5a4842c035082ccd9` | 改写 |
  | `solutionIndex_AI_Eco.txt` | `6ec0a9d84da70e5a2eeb0fa1dffe06c9f70feaf8defdc1d8203d9c09aa2d736c` | 新增 |
  | `solutionIndexAdd_Eco.txt` | `3639767ab465c78e729e56850ce0ccbe2229d84f1dfaa7bb059bc8c62b43da79` | 新增 |

- **组装**：`scripts/make_tuned_rocblas_lib.sh` 把 stock 库
  `/opt/dtk/lib/rocblas/library_gpu4` **复制**到工作目录，再覆盖上述 6 个文件，
  并逐个校验 SHA256；不匹配即退出非零，**不允许带病启动**。
  公共 `/opt/dtk` 库全程只读、未被修改（stock `GemmGenerality_AI_Eco.dat`
  哈希仍为 `be0f15afdd77d40cd3f1cdcf400b84842d1b358342e9c59e4b489a22fc4248cf`）。
- **生效证据**：`ROCBLAS_DEBUG_GENERALITY=1` 实测 kernel 切换
  （(4096,1,4096) 从 kernel 288 切到 712/GEMVMT256；(4096,1,12288) 与
  (4096,64,12288) 均切换到新 kernel）；隔离启动的 inotify 记录到候选目录被实际打开
  `GemmGenerality_AI_Eco.dat`、`GemmGeneralityAdd_Eco.dat` 及对应 SolutionName TXT。
- **回退**：不设 `ROCBLAS_TENSILE_LIBPATH` 即回到 stock 库（见第 4.3 节）。

#### 3.5.2 被否决的候选（保留失败证据，不进提交配置）

| 实验 | 变量 | 结果 |
| --- | --- | --- |
| E6 | `--gpu-memory-utilization` 0.85 → 0.90 | 并发 8 档门槛 -2.8%，**否决** |
| E8 | `VLLM_ROCM_USE_AITER=1`（aiter 总开关） | 启动期 ViT `mm_encoder_attention` flash op 崩溃，**否决** |
| E9 | `compile_mm_encoder=true` | 两档均在噪声内（-1.3%），**否决** |
| E2 | `--max-num-batched-tokens` 8192 / 16384 | 无取值同时满足「吞吐 >3% + TTFT 下降 + 零失败」，回退 4096 |

#### 3.5.3 调优对输出的影响

调优只改**算子实现与执行路径**，不改权重、精度、提示词与采样语义：
6 个固定样例输出可解析率 6/6，坐标漂移 ≤3/1000（与后端间既有漂移同量级）。
官方 ≥98% 一致率仍为 `BLOCKED`（缺官方 GPU 参考输出与判分器，见第 6.1 节），
**不以此宣称精度达标**。

## 4. 服务启动脚本

`env.sh` 中与启动相关的配置：

```bash
export MODEL_PATH="/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV"
export SERVED_NAME="robobrain2.5-8b-nv"
export FRAMEWORK="vllm"
export PORT=30000
export TP=1
export EP=1
export PP=1
export MAX_LEN=32768
export MEM_FRAC=0.85
export NNODES=1
export NODE_RANK=0
export LOAD_TIMEOUT=1800
export USE_TUNED=1                      # 1=最优版（graph+调优库+aiter FA）；0=回退 eager 首提基线
export TUNED_ROCBLAS_DIR="$WORK_DIR/rocblas_tuned/library_gpu4"
export ROCBLAS_TENSILE_LIBPATH="$TUNED_ROCBLAS_DIR"
export VLLM_ATTENTION_BACKEND=ROCM_AITER_FA
export EXTRA_ARGS="--no-enable-prefix-caching --max-num-batched-tokens 4096 --max-num-seqs 64 --limit-mm-per-prompt={\"image\":8}"
```

启动：

```bash
bash run.sh
```

`run.sh` 展开后实际执行的完整命令（取自 `server.log` 首行）：

```bash
python3 scripts/serve_vllm_patched.py \
  --model /public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV \
  --served-model-name robobrain2.5-8b-nv \
  --host 127.0.0.1 --port 30000 \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --max-model-len 32768 --gpu-memory-utilization 0.85 \
  --dtype bfloat16 --trust-remote-code \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 4096 --max-num-seqs 64 \
  --limit-mm-per-prompt '{"image":8}'
```

> **与官方 `vllm serve` 的唯一差别**：入口是 `scripts/serve_vllm_patched.py`，
> 它在启动 vLLM 之前注入 `sitecustomize` 垫片（第 3.3 节），其余参数完全一致。
> **相对首提基线的差异**：去掉了 `--enforce-eager`（启用 torch.compile + graph capture），
> 并新增两个环境变量 `ROCBLAS_TENSILE_LIBPATH` / `VLLM_ATTENTION_BACKEND`（见第 3.5 节）。

重点参数说明：

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `--tensor-parallel-size` | `1` | 本 Pod 只分配到 1 张卡；官方脚手架默认 4，**性能对比为单卡口径** |
| `--max-model-len` | `32768` | 官方 `env.sh` 口径：须覆盖 16K 输入（随机 token 文本重编码后 ~17.3K）+ 视觉 2,042 + 输出 1,024 ≈ 20.3K；实测开到该值**不减少** KV cache（20480 与 32768 下均为 253,312 token） |
| `--gpu-memory-utilization` | `0.85` | 与官方 `env.sh` 的 `MEM_FRAC` 一致；实测 KV cache = 253,312 token（eager 基线）/ 276,480 token（0.90 档，未采纳） |
| `--enforce-eager` | **关**（最优版） | 首提基线因容器 cgroup 上限 115 GiB 而开启 eager（2026-10-27 实测被 OOM killer 杀掉并导致 Pod 重建）。最优版改为启用 graph：启动期由 `run.sh` 的 **70 GiB 内存看门狗**兜底，实测启动稳定（ready_after 170~240 s），收益见第 7 节 |
| 环境变量 `ROCBLAS_TENSILE_LIBPATH` | `$WORK_DIR/rocblas_tuned/library_gpu4` | **GEMM 算子调优库**（第 3.5 节）。`run.sh` 启动前用 `scripts/make_tuned_rocblas_lib.sh` 把 stock 库复制到该目录并覆盖 6 个差异文件，逐个校验 SHA256；公共 `/opt/dtk` 库保持只读、不被改动 |
| 环境变量 `VLLM_ATTENTION_BACKEND` | `ROCM_AITER_FA` | 长上下文 decode 走 aiter FA kernel（10-30 同轮 E7 gate 隔离测得：16384/并发1 +17.9%；本节点未单独隔离该项）。**不要**设 `VLLM_ROCM_USE_AITER=1`：本镜像下会在 ViT `mm_encoder_attention` flash op 崩溃（`reports/23_tuning_progress_20261030.md` §2 E8） |
| `--no-enable-prefix-caching` | 开 | 压测时避免相同 prompt 的 prefill 成本被缓存命中抹掉（实测开启时命中率 64%） |
| `--limit-mm-per-prompt` | `{"image": 8}` | 保留多图能力（赛事要求「单图/多图均正常」） |
| `--max-num-batched-tokens` | `4096` | 降低 profiling 与调度峰值 |
| 环境变量 `PYTHONPATH` | `$WORK_DIR/envs/pylibs` | 客户端图像规范化依赖 `qwen_vl_utils` |

### 4.3 回退方法（回到 eager 首提基线）

调优实验与压测会**杀掉并重启**服务（用不同端口做单变量 A/B）。
一键回到「首提基线配置」只需一个环境变量 —— `run.sh` 是幂等的：

```bash
cd /root/private_data/workspace/robo/submission

# 1) 停掉任何残留的实验实例（只杀本任务的进程）
pkill -f "scripts/serve_vllm_patched.py" || true
pkill -f "VLLM::EngineCore" || true
sleep 8

# 2) 用首提基线配置重新拉起（eager / 32768 / 30000 / mem-frac 0.85 / stock 库）
USE_TUNED=0 setsid nohup bash run.sh > logs/run_stdout_final.log 2>&1 &
for i in $(seq 1 90); do
  curl -sf --noproxy '*' "http://127.0.0.1:30000/health" >/dev/null && break
  sleep 10
done

# 3) 核对确实回到基线配置（应与 bench/run_config.txt 逐字一致）
grep -m1 '^\[run\]' server.log
```

`USE_TUNED=0` 的效果：`env.sh` 不再导出 `ROCBLAS_TENSILE_LIBPATH` /
`VLLM_ATTENTION_BACKEND`，并把 `--enforce-eager` 加回 `EXTRA_ARGS`
（见 `env.sh` 末尾的 `ROBO_BASE_ARGS` 分支）。

期望输出（基线配置）：

```text
[run] python3 .../serve_vllm_patched.py --model ... --served-model-name robobrain2.5-8b-nv
      --host 127.0.0.1 --port 30000 --tensor-parallel-size 1 --pipeline-parallel-size 1
      --dtype bfloat16 --trust-remote-code --gpu-memory-utilization 0.85
      --max-model-len 32768 --enforce-eager --no-enable-prefix-caching
      --max-num-batched-tokens 4096 --max-num-seqs 64 --limit-mm-per-prompt={"image":8}
```

只回退其中一项也可以（例如保留 graph 但换回 stock 库）：

```bash
env -u ROCBLAS_TENSILE_LIBPATH -u VLLM_ATTENTION_BACKEND bash tools/start_bench_service.sh graph 30000
```

> Pod 重启 / SSH 断连后的完整恢复流程见 `reports/00_RUNBOOK.md` §2；
> 15 条实测故障排查见同文件 §4。

## 5. Smoke 测试

### 5.0 固定样例与校验值

`samples/` 下的 5 张图片取自官方 upstream 仓库 `https://github.com/FlagOpen/RoboBrain2.5.git`
@ `af98c932aac9`（`assets/demo/*.jpg`），**本地文件与 upstream 记录的 sha256 逐位一致**；
提示词模板与解析正则逐字复制自 upstream `inference.py:110-124`（diff 后无差异）。

| 文件 | 内容 |
| --- | --- |
| `samples/MANIFEST.md` | 每个样例的 sha256、字节数、upstream 路径与校验结论 |
| `samples/upstream_provenance.json` | upstream commit、`inference.py` 一致性判定、逐图校验值 |
| `samples/tasks_official.json` | 四类官方任务的固定样例（general / grounding / pointing / trajectory） |
| `samples/tasks_multi_image.json` | 多图输入样例 |

坐标/轨迹叠加图见 `outputs/annotated/`（白底叠加 bbox / point / trajectory）。

### 5.1 Smoke

```bash
bash smoke.sh
```

`smoke.sh` 依次执行：健康检查 → 模型列表 → 单轮图文 → 多轮图文 → 流式 →
四类任务 + 多图 + 异常输入（最后一项由 `scripts/smoke_service.py` 完成，输出 `smoke_results.json`（运行产物，写入工作目录，不随仓库提交））。

验证结果（`smoke.log` 关键部分；`smoke_results.json` 为运行产物，写入工作目录）：

- `/health` 返回 `200`
- `/v1/models` 返回 `robobrain2.5-8b-nv`（`max_model_len=32768`）
- 单轮图文回复：「The image shows a white ceramic mug filled with a dark liquid, likely coffee or tea, placed on a rustic wooden surface. …」
- 四类任务全部可被官方正则解析：
  grounding `[0, 309, 620, 989]`；affordance `(904, 600)`；pointing `(582, 725)`；
  navigation `(401, 376)`；trajectory 3 点含深度
- 多图（2 图）general 回复 2223 字符，两图均参与
- 流式：逐 token 返回，以 `data: [DONE]` 结束；**流式拼接结果与非流式逐字一致**
- 多轮图文：第二轮追问正常作答
- 异常输入：损坏图片 → `400`；畸形 data URL → `400`；不可达图片 URL → 返回空内容而非伪正常结果

## 6. 精度测评

> 赛事对精度项的要求是「与官方 GPU 基线一致率 ≥98% 或劣化 ≤2%」。本节点因**官方基线文件、
> 评测数据集与判分器均未下发**而无法完成该项，因此本节分两部分：
> **6.1 赛事官方精度（`BLOCKED`，如实标注）** 与 **6.2 公开上游基准评测（本轮实际做的工作）**。
> 后者**不能替代**前者。

### 6.1 赛事官方精度（一致率 ≥98% / 劣化 ≤2%）：`BLOCKED`

**本项状态：`BLOCKED`。** `verify.py` 按官方基线文件
`baselines/BAAI__RoboBrain2.5-8B-NV.json` 写死的配置调用服务跑评测；该基线文件
（以及官方评测数据集、官方判分器）**尚未下发**，节点也无外网可取。

按验收标准第 2 节：缺官方数据或判分器时精度项必须标 `BLOCKED`，
**不得用自选 demo 或自建 GPU 结果替代官方 98% 验收证据**。
`verify_result.json` 原文只记录 `status=BLOCKED` 与原因，另附一项**明确标注为内部指标**的
「同卡不同后端逐坐标一致性」（2/6 逐坐标一致）——**不能替代官方 GPU 参考一致率**。

```bash
python verify.py --selfcheck            # 确认判分逻辑未被改动（已通过）
source env.sh && python verify.py       # 跑评测 → verify_result.json（当前 exit 2 = BLOCKED）
```

判分口径要点（官方材料到位后 `verify.py` 自动走完整路径，无需改代码）：

| 项 | 口径 |
| --- | --- |
| 判分方式 | `agreement`（坐标一致率 ≥98%）/ `numeric`（取 `\boxed{}` 数值）/ `choice`（A–D 选项）；另支持基线文件给定的外部 `judge` |
| 劣化判定 | 绝对分差，百分制 ≤2 pt、千分制 ≤20 pt；同时记录相对百分比供对照 |
| 计分分母 | 解析失败与请求失败**都留在分母**、记为未命中（早期版本 `continue` 掉失败样本，等于靠丢样本做高一致率，已修正） |
| 缺参考输出 | 单列 `without_reference` 计数并在 `notes` 披露，不让分母被悄悄缩小 |
| 不会误报 PASS | 基线缺失 / 缺 `metrics` / 指标要素不全 → 一律 `BLOCKED`、`exit 2`；只有全部可跑才真正调服务逐样本评测 |

分母口径有可执行测试兜底：`tools/test_verify_denominator.py`（离线，构造 4 道题断言
`scored=4`、`errors=1`、一致率 25% 而非剔除失败样本后的 50%），已实跑通过。
图像请求走 `scripts/image_canonical.py` 规范化，与官方 upstream `inference.py` 取图路径一致
（否则像素不同会翻转输出，见 `reports/09_vllm_adaptation.md` §3）。

### 6.2 公开上游基准评测（10 个公开基准 · 汇总一张表）

**这 10 个评测集都不是赛事官方验收数据集。** 本节给出的是公开上游基准上的任务质量证据，
**不能替代 §6.1 的官方 98% 一致率门槛**。覆盖论文（arXiv 2601.14352）Table 2（2D 空间 5 项）、
Table 3（3D 空间 5 项，其中 Robo-Dopamine 为可选时序扩展，需独立 GRM 模型，本轮 `NOT_RUN`）。

**口径边界**（重要）：

* 对外上报的数字**只有主口径**（下表「本次值」列），与官方实现逐字一致（或属下表已披露的替代口径）；
* 任何**在评测集上按分数挑出来**的口径（§6.2.3 上采样、§6.2.4 提示词）**只作归因/诊断**，
  不替代主口径、不与论文对照、不作验收依据（《精度测试验收要求》§4.2：
  「不得根据某题 mask 或分数选择缩放方法」）；
* 任何**改动评测数据、标签、判分器、聚合规则**的动作本轮**一件都没做**
  （可逐项核验，见 `reports/16_public_bench_results.md` §5.15.1）。

**精度总表（主口径 · 一张表汇总；K100 非 eager 服务 / 单卡 BF16 / greedy）**：

| 基准 | 题数 | 本次值 (K100) | A800 (NVIDIA) | 论文 NV | 判分口径 |
|---|---|---|---|---|---|
| CV-Bench 综合分 | 2638 | **88.12** | 88.00 / 88.04 | 94.58 | 官方 FlagEvalMM `get_result` 零改动 |
| RefSpatial 默认范围均值 | 200 | **63.00** | — | 60.50 | 官方 RoboRefer CLI |
| CrossPoint-Bench | 1000 | **75.20** | — | 75.40 | 官方 `calc_score.py`（上游评测器，不在本仓库）+ `relative_1000` 零改动 |
| EmbSpatial-Bench | 3640 | **77.77** | — | 75.58 | ⚠️ 自建 MC（官方未发布代码，已披露） |
| RoboSpatial-Home | 350 | **62.0** | 62.86 | 73.03 | 官方 `eval_pregenerated_results` 零改动 |
| Q-Spatial-Bench | 101 | **62.7 ± 2.5%**（3 次均值；单次 60.4~65.4） | 对齐均值 61.9%（2 次）——**跨平台无显著差异** | 73.53 | 官方公式复刻 + 披露兜底抽取 |
| MSMU-Bench | 987 | 定量 47.0~76.4%（δ1.25~1.95）/ 定性 63.8% | — | 64.17 | 官方 `judge_prompt`（judge=本服务，greedy） |
| TraceSpatial 3D（最大可行覆盖） | 98/99 | Start 78.6~80.6 / End 63.3~66.3 / Succ 46.9~49.0 | — | 83 / 63 / 44 | 官方判分（4 条样本官方代码不终止，§6.2.5） |
| ShareRobot-T | 200 | DFD **0.181** | — | 0.1164 | FlagEvalMM **PR#84** 零改动 |
| ShareRobot-A | 200 | mIoU 0.019（GT 原始分辨率与缩略图参考系错配，不可靠） | — | —（论文未报） | 同上 |
| VABench-V | 300 | DFD **0.413** | — | 0.1281 | ⚠️ 官方判分未发布，PR84 同款几何指标（披露替代） |
| Robo-Dopamine | — | `NOT_RUN`（可选扩展，需独立 GRM 模型） | — | — | — |

<!-- PUBLIC_BENCH_TABLE:BEGIN -->


**主指标一览（主口径，对外上报口径）**：

| 基准 | 主指标 | 分母 / 口径 | 论文参考 | 本次主口径 | 差值 |
|---|---|---|---|---|---|
| CV-Bench | **综合分 CV_score** | 2638 / `canonical_png` | 94.58 | **88.12** | **−6.46 pt** |
| CV-Bench | Acc(ADE20K) / Acc(COCO) | 2638 / `canonical_png` | 论文只给综合分 | 77.88 / 85.96 | — |
| CV-Bench | Acc_2D / Acc_3D | 2638 / `canonical_png` | 论文只给综合分 | 81.92 / 94.33 | — |
| RefSpatial | Location / Placement | 100 / `upstream_scaled` | 论文未公开子集切分 | 68.00 / 58.00 | — |
| RefSpatial | **默认范围均值**（Location+Placement） | 200 / `upstream_scaled` | 60.50 | **63.00** | **+2.50 pt** |
| RefSpatial | Unseen（重叠子集，单列，不计入默认均值） | 77 / `upstream_scaled` | — | 48.05 | — |
| TraceSpatial | **3D Start / End / Overall** | 96/100 / `official` | 83 / 63 / 44 | **78.13 / 63.54 / 46.88** | −4.88 / +0.54 / +2.88 pt |
| TraceSpatial | No-Collision Rate | 96/100 / `official` | — | 81.25 | — |
| TraceSpatial | 3D Start / End 距离 (m) | 96/100 / `official` | — | 0.1268 / 0.1965 | — |
| TraceSpatial | 2D Start-in-Mask / End-in-BBox | 100/100 / `official` | — | 73.00 / 52.00 | — |
| TraceSpatial | 2D DFD | 97/100（计入路径均值）/ `official` | — | 0.2055 | — |

> 上表由 `tools/inject_accuracy_table.py` 从官方判分器产出的原始 JSON 现场生成，
> **不在本文手写**；完整 50+ 行全指标 × 全次口径明细随包提交为 `bench_public/summary_table.md`，
> 逐指标口径边界与分母构成见 `reports/16_public_bench_results.md` §5。

**稳定性与可复现性**：CV-Bench 固定样例 × 2 次重复推理原始输出**一致 12/12、变化率 0.0**；
RefSpatial 变化率 0.5833、TraceSpatial 0.75（长链推理固有噪声），三者终态全 OK；
离线重判分复现（`bench_public/rescore.py`）比较指标数 **12**、不一致 **0** ✅。

<!-- /PUBLIC_BENCH_TABLE -->


> ⚠️ TraceSpatial 3D 数字以上方总表 + §6.2.5 的**最大可行覆盖口径**为准
> （98/99 分母；自动生成表中的 96 分母为兜底口径历史值，根因见 §6.2.5 勘误）。

#### 6.2.1 官方 3D 判分的三个**会让报告出错**的陷阱

| 陷阱 | 现象 | 处理 |
|---|---|---|
| open3d 版本 | 0.19/0.20 改了 `ScalableTSDFVolume` 体素语义 → TSDF 恒抽不出点 → 碰撞检查恒 False → `No-Collision Rate` 恒 1.0、`Overall Success` **偏乐观** | 隔离前缀 `envs/pylibs_o3d_v2` 固定 open3d 0.18.0，官方 `summarize_acc.py`（上游评测器，不在本仓库）**一字未改** |
| 官方静默丢弃无效样本 | 归一化后含 NaN / 尺寸无效的样本不打印标记就丢掉；实测 100 片中 3 片为空 dict，分母应为 97 而非 100（会让 2D `dfd` 从 0.2055 偏成 0.1993，差 **3.0%**） | `tools/remerge_ts_3d.py` 用已落盘 `per_chunk` 重算并逐指标对拍 |
| 深景病态样本 | 样本 56/78/82/98 单条官方判分 **>65 min 仍未结束**；官方判分器自身 `[Skip]` 计数为 **0** —— 是**我们**主动排除 | 排除索引写进产物 `excluded_sample_indices` 字段，**可审计**；3D 子集分母 96/100 |

#### 6.2.2 CV-Bench `Count` 缺口：成因已定位（可证实）

论文 94.58 与实测 88.12 差 6.46 pt，**缺口 100% 落在 `Count` 子任务**（788 题 72.34%）。
逐项排除 8 组因子（判分链路 / 提示词 / token 预算 / 抽图压缩 / thinking / 图像编码协议等）后，
对失分样例做**误差分布 + 人工目视核对**（导出 15 张图逐张看）：

| 误差（预测个数 − 真值个数） | −3 | −2 | −1 | **0** | +1 | +2 |
|---|---|---|---|---|---|---|
| 题数 | 6 | 37 | 111 | **570** | 56 | 5 |

* **|误差| ≤ 2 占 98.9%**；769/788 纯字母输出、**0 例数字型输出** → 模型在**真实地数**，不是格式错；
* 失分拆成两类：**真实感知漏检**（小目标）与**实例标注口径不一致**
  （肉眼 1 盏灯但 ADE20K 标 3 个实例；肉眼 5 匹斑马但 COCO 只标 4 匹）；
* **第二类无法用任何推理侧参数修复** → 剩余杠杆在**权重/训练数据**，不在推理栈。

#### 6.2.3 输入分辨率因子：**只作归因，不作提分**（非官方口径，单列）

模型原生 `preprocessor_config.json` 为 `size = {longest_edge: 16777216, shortest_edge: 65536}`
→ 256×256 的图只有 **64 个视觉 token**。等比上采样（LANCZOS，不缩小原图，总像素 ≤3,000,000）
可加密 ViT patch 网格。只跑 `Count` 788 题、图像与判分口径不变，唯一变量 = 缩放策略：

<!-- UPSCALE_TABLE:BEGIN -->


| 缩放策略 | Acc(ADE20K) | Acc(COCO) | Acc_2D | Δ vs canonical |
|---|---|---|---|---|
| `canonical_png`（同会话对照） | 63.74 | 78.25 | 71.00 | — |
| **`up2_png`** | **65.79** | **82.06** | **73.93** | **+2.93** |
| **`up2_jpeg95`** | **65.79** | **82.74** | **74.27** | **+3.27** |

<!-- /UPSCALE_TABLE -->


* **机制**：`shortest_edge=65536` → 小图只有 64 个视觉 token；2× 上采样让小目标占到更多 token，
  与「洗碗机漏检」这类失分机制一致。逐题配对符号检验 p = 1.27e-06，
  同配置两次运行的噪声底只有 0.29 pt（不显著）⇒ 增益是真的。
* **全量诊断**（`up2_png`，2638 题，47.3 min，2638/2638 终态 `ok`）：

<!-- UPSCALE_FULL_TABLE:BEGIN -->


| 指标 | 主口径 `canonical_png` | `up2_png`（诊断口径） | Δ |
|---|---|---|---|
| Acc(ADE20K) | 77.88 | 78.99 | **+1.11** |
| Acc(COCO) | 85.96 | 88.45 | **+2.49** |
| Acc_2D | 81.92 | 83.72 | **+1.80** |
| Acc(Omni3D 3D) | 94.33 | 93.83 | **−0.50** |
| **CV_score（官方综合分）** | **88.12** | **88.78** | **+0.66** |

<!-- /UPSCALE_FULL_TABLE -->


3D 分支反而 −0.50 pt；官方综合分净 **+0.66 pt**，只有 6.46 pt 缺口的**约 1/10**
⇒ **6.46 pt 缺口不可能由输入分辨率解释**，必须回到权重/训练数据一侧解决。

⚠️⚠️ **合规性声明（本节数字的正确用法，必读）**

1. **上采样不是官方协议**：上游 `RoboBrain2.5/inference.py` 只把长宽取 28 的倍数、**不放大**；
   官方 `FlagEvalMM encode_image` 只缩小 / 补白边。
2. **它没有篡改 benchmark**：题目、GT 标注（SHA256 逐字相同）、官方判分器（零 patch）、
   官方聚合公式、提示词、坐标与阈值**全部未改**；唯一变动是送进模型的图像编码，
   且放大后仍在模型声明的 `longest_edge=16,777,216` 能力范围内。
3. **但它是「在评测集上按分数挑出来的协议」**（《精度测试验收要求》§4.2）
   ⇒ 本口径数值**带选择偏差**：**不得**替代主口径 **88.12**、**不得**与论文 94.58 并列、
   **不得**作为精度结论或官方 98% 一致率门槛的证据。
4. **它的合法用途只有一个：归因** —— 定位到「输入分辨率 → 视觉 token 密度」是 `Count` 缺口的机制，
   说明剩余缺口在推理栈上已无可动之处。做成合规的精度提升需要「独立验证集选协议 →
   留出集验证 → 冻结协议再跑测试集」，本轮**没有留出集**，如实标注**未完成**。
5. **主口径 `canonical_png` 本身不是挑出来的**：8 组协议矩阵中 `raw_jpeg95` 71.67、
   `long1500_jpeg95` 71.62 都比主口径 71.00 **高 0.6~0.7 pt** —— 主口径是先验按
   「与上游 `qwen_vl_utils` 逐字一致」选定的，且选中的是 8 组里**偏保守**的一个
   ⇒ **对外上报的 88.12 不含协议选择偏差**。

#### 6.2.4 提示词口径与运行间噪声底

两条官方提示词口径都跑过（`Count` 788 题，图像协议固定 `canonical_png`）：

| 提示词口径 | Acc(ADE20K) | Acc(COCO) | Acc_2D |
|---|---|---|---|
| **`flagevalmm`（主口径，与官方逐字一致）** | **64.33** | 78.25 | **71.29** |
| `dataset`（数据集自带 prompt） | 61.70 | 78.25 | 69.97 |

`dataset` 口径下模型 767/788 输出 `(C) 1` 形式，官方判分器对此**比严格字母归一化更宽松**
（多认 8 题），但即便如此在 ADE20K 上**仍差 2.63 pt**，远超噪声底 ⇒ 「dataset 口径更差」是真的。

**噪声底**：逐字同配置两次运行（788 题）原始输出 10/788 不同、判分层面 6/788 翻转、
官方 `acc_2d` 差 **0.29 pt**（符号检验 p=0.6875，不显著）⇒ **噪声底 ≈ 0.3 pt**；
本报告 `Count` 小于 ~0.6 pt 的分差一律视为噪声。

**评测耗时分级**（避免卡在慢评测上）：台账由 `tools/build_eval_timing.py` 自动生成。

| 档位 | 预估墙钟 | 处置 | 本轮实例 |
|---|---|---|---|
| L1 | < 5 min | 随时跑 | RefSpatial/TraceSpatial 推理、误差分布统计、合并口径重算 |
| L2 | 5~20 min | 主力档 | CV-Bench `Count` 788 题各组、TraceSpatial 3D 兜底判分（941 s） |
| L3 | 20~60 min | **设硬上限** | CV-Bench 全量 2638 题（2638/2638 终态 `ok`）、8 组协议矩阵 |
| L4 | > 60 min | **不跑**或先排除病态样本 | 3D 整集判分（含深景病态样本）、`up4` 上采样 |

#### 6.2.5 TraceSpatial 3D「病态样本」根因勘误与最大可行覆盖

> **勘误**：早期版本把这 4 条深景样本（56/78/82/98）判分超时归因于
> "DBSCAN 在大点云上近似二次复杂度"，**该归因错误**（实测 DBSCAN/目标点云链路 0.0 s 完成）。
> 真实根因（py-spy 栈 + 端到端复现双重证实）：模型对这些样本输出**恒定退化轨迹** →
> 官方距离均匀插值产生 NaN → 官方 3D 路径（无 NaN 过滤）调用递归版
> `discrete_frechet_distance`，其判据对 NaN 永假 → **指数级递归、永不终止**。
> 100/100 全量官方判分对任何变体组合都不可达（16 组合预筛：5 挂 / 2 Skip / 9 可完成）。

**最大可行覆盖口径**（96 条兜底 + 可完成样本补跑，排除索引与根因写入产物
`outputs/bench_public_ts3d_best/`）：

| 组合 | 分母 answers/eval | 3D Start | 3D End | Overall | 排除 |
|---|---|---|---|---|---|
| 基线 eager / official | 98/98 | 78.57 | 63.27 | 46.94 | 56, 78 |
| 候选 graph / official | 98/98 | 80.61 | 66.33 | 48.98 | 56, 78 |
| 基线 eager / upstream | 99/97 | 61.86 | 56.70 | 36.36 | 82 |
| 候选 graph / upstream | 99/97 | 61.86 | 61.86 | 38.38 | 98 |

与论文 83/63/44 对照：official 变体 End≈63、Success≥44，结论方向与 96 口径一致。
完整证据链见 `reports/20_extended_bench_results.md` §6。

#### 6.2.6 跨硬件对照（A800）与偏差归因

**跨硬件对照**（`reports/21_a800_extbench_crossval.md`）：RoboSpatial A800 62.86 vs K100 62.0
（Δ0.86 pt，同一保守偏置）—— 缺口在 bench 配置侧（官方 harness 提供深度输入）。
Q-Spatial 经双侧同配置重复定案：**K100 三次均值 62.7±2.5%、A800 两次均值 61.9%，
跨平台无显著差异**；该基准为千 token 长链推理，单次分数须标注不确定度。

**偏差归因一览**（全部非算子侧；算子保真另有 CV-Bench A800↔DCU 99.6% 逐题一致旁证，
见 `reports/19_a800_cross_validation.md`）：

| 偏差 | 归因 |
|---|---|
| CV-Bench −6.46 pt | `Count` 子任务：小目标漏检 + 实例标注口径（15 例人工目视核实） |
| RoboSpatial −11 pt | 模型保守偏置 + 官方跑法疑用深度输入（已实证 +13 pt） |
| Q-Spatial −13 pt | 提示引导/估计质量（模板变体差 21.8 pt；跨硬件边界翻转 +8.9 pt） |
| ShareRobot-T / VABench-V | 初版坐标适配层括号缺陷虚高（0.181/0.413 为修复后值） |

**复现**（工作目录 `/root/private_data/workspace/robo` 下执行，脚本随包提交）：
`bash tools/start_bench_service.sh eager 30000` 起服务 → `python3 bench_public/manifest.py`
→ `run_cv_bench.py` / `run_refspatial.py` / `run_tracespatial.py` → `python3 bench_public/rescore.py`
离线重判分；扩展基准用 `tools/run_new_benches_1.sh` → `run_new_benches_2.sh` → `master_bench_queue.sh`。

## 7. 性能参考

性能测试脚本 `bench.sh` 使用 SGLang 自带压测客户端（对 vLLM 服务通用），
随机 token 输入、固定输入输出长度，负载如下：

| 输入 | 输出 | 并发 |
| --- | --- | --- |
| 4K（4096） | 1K（1024） | 1 / 8 / 64 |
| 16K（16384） | 1K（1024） | 1 / 8 / 64 |

每档连测两次，第一次预热，**取第二次数据**；请求数为并发的 2 倍（至少 8 个）。

```bash
bash bench.sh                           # 文本模型
BENCH_MODE=image bash bench.sh          # 多模态理解模型，每个请求带 1 张图
```

单档命令（`bench.sh` 内部对 6 档 × 2 次循环执行）：

```bash
python -m sglang.bench_serving \
  --backend vllm --host 127.0.0.1 --port 30000 \
  --model robobrain2.5-8b-nv --tokenizer "$MODEL_PATH" \
  --dataset-name image --image-count 1 --image-resolution 1080p \
  --random-input-len 4096 --random-output-len 1024 --random-range-ratio 1 \
  --max-concurrency 8 --num-prompts 16 --seed 1 \
  --output-file bench/round2.jsonl
```

> **与官方 `bench.sh` 的差别只有两处，且都不改变负载语义**：
> 1. 本镜像没有 `sglang.benchmark.serving`（新版模块名），
>    `bench.sh` 按官方脚本自带的 fallback 使用旧版 `sglang.bench_serving`；
> 2. image 数据集对后端做硬白名单（`["sglang","sglang-native","sglang-oai-chat"]`），
>    `--backend vllm` 会直接抛 `ValueError`，本节点用 `scripts/bench_serving_vllm_image.py`
>    **在进程内**把数据集构造的后端映射为 `sglang-oai-chat`（两者都是「服务端自己套 chat template」，
>    语义一致），**不修改公共 site-packages**。
>
> **输出长度（重要）**：官方脚本不传任何 ignore-eos 开关，客户端默认 `ignore_eos=True`
> （`payload["ignore_eos"] = not args.disable_ignore_eos`），服务端因此忽略 EOS，
> **每个请求实际生成满 1024 token**，满足验收标准 §7.1「固定 1K 输出压测」。
> 实际输出 token 数记录在 `bench/*.jsonl` 的 `total_output_tokens`
> 与 `total_output_tokens_retokenized`（对输出文本重新分词的真实长度）。
> **两个字段口径不同**：前者是计划长度，后者是文本重新分词后的长度；
> 两者不等**不代表提前 EOS**（已用服务端引擎计数器逐档核实，见第 8 节踩坑第 10 条）。
>
> **上下文容量（重要）**：`--random-input-len` 只计**文本** token，请求还带 1 张 1080p 图
> （约 2,040 视觉 token），因此服务端实际 `prompt_tokens` ≈ 文本 + 2,040。
> 官方 `env.sh` 的 `MAX_LEN` 为 32768，本节点早期版本按单卡余量设成 20480，
> 结果 16384 档中「文本重编码后 >18,440 token」的请求会因
> **文本 + 视觉 + 输出 > max_model_len** 被直接拒绝；现已改回官方口径 **32768**。
> 逐档的实际输入/输出 token 数已记录在 `bench/*.jsonl`（`total_input_text_tokens`、
> `total_input_vision_tokens`、`total_output_tokens`）与各档日志的 `#Input tokens` 中。

第二次结果（`bench/summary.md` 原文）：

<!-- BENCH_SUMMARY:BEGIN -->

| random_input_len | random_output_len | max_concurrency | completed | duration | request_throughput | output_throughput | total_throughput | mean_ttft_ms | p99_ttft_ms | mean_tpot_ms | p99_tpot_ms | mean_e2e_latency_ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 1024 | 1 | 8 | 285.29 | 0.03 | 28.71 | 206.98 | 1411.01 | 1427.87 | 33.48 | 33.67 | 35660.08 |
| 4096 | 1024 | 8 | 16 | 144.12 | 0.11 | 113.68 | 820.13 | 10346.68 | 10402.20 | 60.30 | 60.37 | 72028.57 |
| 4096 | 1024 | 64 | 128 | 931.55 | 0.14 | 140.70 | 1014.65 | 114687.63 | 490857.75 | 304.85 | 661.73 | 426545.41 |
| 16384 | 1024 | 1 | 8 | 429.72 | 0.02 | 19.06 | 378.74 | 6738.19 | 6878.12 | 45.92 | 50.01 | 53709.66 |
| 16384 | 1024 | 8 | 16 | 361.86 | 0.04 | 45.28 | 898.14 | 26063.13 | 54884.90 | 150.95 | 166.22 | 180484.79 |
| 16384 | 1024 | 64 | 128 | 13480.76 | 0.01 | 9.72 | 192.86 | 4062229.71 | 6026783.45 | 1395.95 | 1826.13 | 5490284.66 |
<!-- BENCH_SUMMARY:END -->


<!-- TUNING_COMPARISON:BEGIN -->

### 7.1 与基线的同口径对比

**同卡 A/B：首提基线 eager → 最优版**（当前节点（同一张卡），同一负载、同一客户端；赛事口径的正式对比）

| 输入 | 并发 | 首提基线 eager tok/s | 最优版 tok/s | 吞吐提升 | TTFT 变化 | TPOT 变化 | 基线轮/候选轮 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4096 | 1 | 21.77 | 28.75 | +32.07% | -3.4% | -25.0% | 2/2 |
| 4096 | 8 | 101.65 | 113.68 | +11.84% | -3.5% | -11.7% | 2/1 |
| 4096 | 64 | 133.86 | 140.70 | +5.11% | -4.6% | -5.2% | 2/1 |
| 16384 | 1 | 16.50 | 19.43 | +17.80% | -2.9% | -16.6% | 2/2 |
| 16384 | 8 | 44.16 | 45.28 | +2.54% | -1.3% | -2.7% | 2/1 |
| 16384 | 64 | 9.35 | 9.72 | +4.03% | -3.7% | -4.0% | 1/1 |

**同卡 A/B：graph 基线 → 最优版**（2026-10-31 节点（历史同节点，卡指纹未记录），只开 graph、不带任何调优开关的对照；用于把收益拆成「graph」与「调优库 + aiter FA」两部分）

| 输入 | 并发 | graph 基线（2026-10-31 节点存档） tok/s | 最优版（2026-10-31 节点存档） tok/s | 吞吐提升 | TTFT 变化 | TPOT 变化 | 基线轮/候选轮 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4096 | 1 | 22.38 | 28.73 | +28.37% | -4.2% | -22.7% | 1/2 |
| 4096 | 8 | 104.84 | 115.66 | +10.32% | -10.0% | -9.2% | 1/2 |
| 16384 | 1 | 17.09 | 19.10 | +11.77% | -4.1% | -11.4% | 2/2 |

> 首提基线在 **2026-10-27 节点** 的原始六档记录保留在 `bench/round1.jsonl` /
> `bench/round2.jsonl`（官方口径基准，见第 8 节的节点差异说明）；
> 最优版的原始数据在 `bench/tuned/`。逐档对比表见 `reports/tuning_comparison.md`，
> 完整交付报告见 `reports/25_tuning_delivery_20261031.md`。
> 容器重建会换物理卡，跨卡绝对值不可比，上表只含**同一张卡**上的数据。
<!-- TUNING_COMPARISON:END -->


> **读表要点（单卡口径）**
> - `output_throughput` 是**输出** token 吞吐；`total_throughput` 另含输入 token。
> - **并发 64 档没有按 64 并发跑满**：单卡 KV cache 只有 253,312 token
>   （`num_gpu_blocks=3958` × `block_size=64`）。4096 档每路需 7,374 token
>   （文本 ~4,308 + 视觉 2,042 + 输出 1,024），实测服务侧常驻 **50 路**
>   （p50=50 / p90=56 / max=59），其余排队；
>   16384 档每路需 20,316 token，实测常驻 **15 路**（p50=15 / p90=16，
>   起跑瞬间可到 46），其余约 47 路排队。
>   因此这两档的 `mean_ttft_ms` 被排队拉高（70 s / 数千秒），**不是模型变慢**。
> - 官方脚手架默认 TP=4，本节点只有 1 张卡，**性能基线是单卡口径**；
>   与官方基线比对时必须带上卡数与真实并发度。
> - 两轮结果基本一致（4096/c1 两轮都是 25.15 tok/s），说明测量稳定；
>   官方口径要求取第二轮，上表即第二轮。
> - **未降级**：没有为「跑过」而减少并发、减少图片或缩短输出；
>   失败档位与排队情况在 `bench/summary_detailed.md` 与各档 `.log` 中如实保留。

- 原始数据：`bench/round1.jsonl`、`bench/round2.jsonl`、各档日志 `bench/round*_*.log`
- 辅助产物：`bench/summary.md`（官方 13 列格式，README 上表即取自它）、
  `bench/summary_detailed.md`（增补失败数与输入 token 文本/视觉拆分）、
  `bench/run_config.txt`（本次压测的服务命令行与环境快照）
- **踩坑留证**：`bench_pitfall_ignore_eos/` 保留了「误加 `--disable-ignore-eos`」
  那一版（输出吞吐虚高约 4×）的原始日志与 jsonl；`bench_verify/` 是修正前后
  「每请求是否真的生成满 1024 token」的对照验证。
  这两个目录**不是**性能数据，仅供复核口径。
- 本结果作为该件「模型 × 卡型 × 框架 × 精度」的性能基准；后续优化在同卡型、同框架、同负载下对比
- **口径提示**：`--random-input-len` 指的是**文本** token 数，请求还带 1 张图，
  因此服务端实际 `prompt_tokens` 大于该值（含视觉 token），已逐条记录在原始 jsonl 中
- **单卡口径**：官方脚手架默认 TP=4，本节点只有 1 张卡；并发 64 档受 KV cache 容量限制，
  实际并发度低于请求并发度，详见第 8 节

## 8. 注意事项

- **不支持的负载**：16K 输入 × 64 并发档受 KV cache 容量限制。
  实测 KV cache = 253,312 token（`num_gpu_blocks=3958` × `block_size=64`），
  16384 输入档单请求实测占用 **~20,316 token**（文本 17,250 + 视觉 2,042 + 输出 1,024），
  理论并发上限约 12；**实测服务侧真实常驻并发 15 路**（p50=15 / p90=16，起跑瞬间可到 46），
  其余 47 路排队。该档指标按**实际并发度**解读，未按请求的 64 并发宣称。
  原始数据见 `bench/round*_16384_1024_c64.log`，逐档并发分布见
  `reports/10a_server_counter_crosscheck.md`
- **修改的模型参数**：**无**。`config.json`、`generation_config.json`、权重、分词器均未改动。
  官方 `generation_config.json` 自带 `do_sample=true / temperature=0.7 / top_p=0.8 / top_k=20`，
  本次评测按官方 `inference.py` 的做法使用 greedy（请求显式传 `temperature=0`）
- **已知限制**：
  - 精度项 `BLOCKED`（缺官方基线文件、评测集与判分器）
  - **官方解析正则的覆盖边界**（验收标准 §4.3 要求单独检查）：
    轨迹深度写成整数（`[(1,2,3)]`）或科学计数法（`1e-3`）、坐标写成浮点（`901.5`）、
    pointing 缺括号、grounding 少一个坐标 —— 这些**官方正则同样不匹配**，
    本节点**保持官方口径不改**，但把原文、解析值与失败原因逐条留证
    （`outputs/phase4_parse_coverage/parse_coverage.md`，20 条用例中 8 条暴露缺口）；
    解析失败**计入官方计分分母、不剔除**。vLLM 分析模式的 `<bbox>x y x y</bbox>`
    由 `scripts/answer_adapter.py` 做纯格式桥后交给官方正则解析。
  - SGLang 在本节点不可用：镜像内 `aiter` 残缺构建（13 个符号未定义），
    5 个后端全崩；**无法单独计 SGLang 一件**
  - 多图上限 8 张（`--limit-mm-per-prompt '{"image": 8}'`）
  - **非 eager（torch.compile + graph capture）已作为当前提交配置的一部分**
    （`env.sh` 的 `USE_TUNED=1`，`EXTRA_ARGS` 不再带 `--enforce-eager`）。
    实测可启动（ready 260 s、cgroup 峰值 ~17.75 GiB，远低于 115 GiB 上限），
    `run.sh` 仍保留 70 GiB 内存看门狗作为保护。
    **已知风险（如实标注）**：非 eager 下输出数值会变化（与 transformers 基线逐坐标一致率
    3/6 → 2/6，最大坐标偏差 7/1000 → 30/1000），而官方 ≥98% 精度门槛因缺官方基线为
    `BLOCKED`，**无法验证该变化无害**。详见 `reports/11_tuning_experiments.md` §2
    与 `reports/25_tuning_delivery_20261031.md`。
    （历史上 2026-10-27 01:02 首次无看门狗的尝试曾触顶 115 GiB → OOM killer 杀容器 → Pod 重建，
    所以任何非 eager 尝试都必须带看门狗。）
  - **不宣称精度达标**：验收标准 §8.2 要求「同口径提升 ≥10%」**且**「优化后精度门槛通过」
    两条同时成立；本节点只满足前者（且只按**同一张卡**的 A/B 计算，见 §3.5.0），
    后者因缺官方材料为 `BLOCKED`。因此精度一栏保持 `BLOCKED`，
    **不以性能结果代替精度结论**。核算见 `reports/05_final.md` §4.6、
    `reports/08_gap_vs_official_baseline.md` 风险 #13。
  - **换卡（2026-11-01）**：容器被平台回收重建后**物理卡会更换**，跨卡绝对值不可比
    （实测同配置跨卡可差约 30%）。本包的压测工具链已把「同一张卡」做成硬约束：
    每次压测记录 `card_id`，报告只对同卡数据做 A/B，跨卡数据降级为存档。
    详见 §3.5.0 与 `reports/25_tuning_delivery_20261031.md` §6。
- **踩坑与规避**：
  1. `AttributeError: 'Qwen3VLTextConfig' object has no attribute 'tie_word_embeddings'`
     → vLLM 从子 config 读字段，transformers 5.3.0 只在顶层复合 config 保留
     → `sitecustomize` 垫片 / `patches/0001-vllm-qwen3-tie-word-embeddings.patch`
  2. 垫片在父进程生效但子进程仍报同一错
     → vLLM 用 `VLLM_WORKER_MULTIPROC_METHOD=spawn` 起 EngineCore，monkey-patch 不被继承
     → 垫片必须放 `sitecustomize.py`
  3. 非 eager 启动把容器内存顶到 115 GiB 上限 → OOM killer 杀容器 → **Pod 重建、节点 5 小时不可达**
     → 首跑一律 `--enforce-eager`；要试非 eager 必须带内存看门狗
     （`scripts/run_non_eager_ab.sh`，超阈值先杀进程，避免 kernel OOM）
  4. `ImportError: libgalaxyhip.so.5` → 未加载 DTK 运行时 → `source /opt/dtk/env.sh`
     （且**不能**在 `set -u` 下 source：`/opt/dtk/env.sh` 引用了未定义变量）
  5. serving 的 grounding 输出官方正则解析不了 → 模型在"分析模式"下输出 `<bbox>x y x y</bbox>`
     → `scripts/answer_adapter.py` 格式桥
  6. serving 的 grounding 输出**与直跑风格完全不同** → 图像二次重采样导致像素不同（最大差 0.47）
     → `scripts/image_canonical.py` 客户端规范化（对齐后最大差 0.000000）
  7. 压测 TTFT 异常低 → vLLM 默认开 prefix caching，相同 prompt 的 prefill 被缓存命中（实测 64%）
     → `--no-enable-prefix-caching` + 每请求不同 filler 文本
  8. 16K 档请求被服务拒收 → `--max-model-len 20480` 装不下「文本重编码后 ~17.3K + 视觉 2,040 + 输出 1,024」
     → `MAX_LEN` 改回官方 `env.sh` 口径 **32768**（模型自身 `max_position_embeddings=262144`，未擅改声明）
9. 压测输出吞吐虚高约 4× → 客户端把 `--disable-ignore-eos` 的语义理解反了：
     `payload["ignore_eos"] = not args.disable_ignore_eos`，
     官方脚本**不传**该开关即 `ignore_eos=True`（忽略 EOS，输出满 1024）；
     误传该开关反而允许提前 EOS，而客户端的 `total_output_tokens` 统计的是**计划长度**
     （`output_lens.append(output_len)`），不会反映实际生成量
     → 已删掉该开关，恢复官方默认；证据保留在 `bench_pitfall_ignore_eos/`
     （同一档位：`total_output_tokens=8192` 而 `retokenized=2069`）
  10. 想用 `total_output_tokens_retokenized` 去判断「是否提前 EOS」→ **会误判**：
     该列是**已解码文本重新分词后的长度**，不是生成量。16384 档两者相差最大
     （131,072 vs 101,416，差 22.6%），看起来像提前 EOS，实际是**随机 token 提示词
     让模型输出高度重复**，同一段文本被 BPE 压成了更少的 token
     → 正确判法是读 **vLLM 引擎自己的计数器**（`Avg generation throughput` 累加）：
     实测引擎生成量 = 计划量的 **95.6%–99.7%**（16384/并发 64 档达 **99.7%**，
     130,616 vs 131,072），**确认不存在提前 EOS**，官方 `output_throughput` 列可直接填报
     → 核算脚本 `tools/analyze_server_throughput.py`，逐档表
     `reports/10a_server_counter_crosscheck.md`，
     报告 `reports/10_phase5_bench.md` §3.7 / §3.8
  11. 收尾流水线的「等压测结束」超时预算 = 启动时刻 + 4 h，**早于**六档压测实际结束
     （最后一档 16384/并发 64 单轮就要 ~2.2 h）→ 流水线会带着 5 档数据提前进入调优实验，
     而调优实验的 A/B 脚本会 `pkill` 服务进程，**直接杀掉正在被压测的服务器**，最后一档数据作废
     → 等待条件改为「压测进程退出 **且** `round2.jsonl` 已有 6 条」，超时预算提到 10 h 且
     超时直接退出；所有会杀服务的实验脚本都加了「压测在跑就拒绝启动」的安全闸。
     这条属于流程性坑，与模型无关，但一旦发生会**静默毁掉整轮数据**
     （见 `reports/08_gap_vs_official_baseline.md` 风险 #12）
- **其他**：首次启动（eager）约 200 s；容器内存峰值 11.2 GB；
  `MAX_LEN=32768`（官方 `env.sh` 口径）与 `MEM_FRAC 0.85` 共同决定 KV cache 容量 253,312 token
  （`max_model_len` 只影响单请求上限，不改变 KV 分配量：实测 20480 与 32768 下均为 253,312 token）。
