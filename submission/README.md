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
- 提交仓库：【待填：托管平台可访问链接】；本次交付代码 revision = `0658ed25dcf32eaa66422e1e5fbdce16682fcd11`（短 `0658ed2`，逐文件校验值见 `MANIFEST.sha256`，详见 `.revision`）

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
git clone 【URL】 && cd 【repo】
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
| 区域 | 【待填】 |
| Notebook 规格 | CPU 6 核 / 内存 115 GiB（cgroup 上限）/ 【系统盘、文件存储待填】 |
| 加速卡与显存 | `K100-AI`（`K500SM_AI`）×**1**；64 GB |
| 开发镜像 | `jupyter-sglang5.12-ubuntu22.04-dtk26.04-py3.10-qwen3.6:v1.0` |
| 模型获取方式 | AI 社区模型库（只读路径） |
| 服务映射 | 后端 `127.0.0.1:30000` → 代理【待填】→ 外部【待填】（未使用可写「无」） |
| 平台证据 | 【待补截图：Notebook 创建页、环境自检、模型管理、服务映射】 |

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
```

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
export EXTRA_ARGS="--enforce-eager --no-enable-prefix-caching --max-num-batched-tokens 4096 --max-num-seqs 64 --limit-mm-per-prompt {\"image\": 8}"
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
  --enforce-eager --no-enable-prefix-caching \
  --max-num-batched-tokens 4096 --max-num-seqs 64 \
  --limit-mm-per-prompt '{"image": 8}'
```

> **与官方 `vllm serve` 的唯一差别**：入口是 `scripts/serve_vllm_patched.py`，
> 它在启动 vLLM 之前注入 `sitecustomize` 垫片（第 3.3 节），其余参数完全一致。

重点参数说明：

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `--tensor-parallel-size` | `1` | 本 Pod 只分配到 1 张卡；官方脚手架默认 4，**性能对比为单卡口径** |
| `--max-model-len` | `32768` | 官方 `env.sh` 口径：须覆盖 16K 输入（随机 token 文本重编码后 ~17.3K）+ 视觉 2,042 + 输出 1,024 ≈ 20.3K；实测开到该值**不减少** KV cache（20480 与 32768 下均为 253,312 token） |
| `--gpu-memory-utilization` | `0.85` | 与官方 `env.sh` 的 `MEM_FRAC` 一致；实测 KV cache = 253,312 token |
| `--enforce-eager` | 开 | **容器 cgroup 上限 115 GiB**：非 eager 启动会在 inductor 编译期触顶，被 OOM killer 杀掉并导致 Pod 重建（实测记录见第 8 节）。代价是失去编译/graph 加速，本节点无法规避 |
| `--no-enable-prefix-caching` | 开 | 压测时避免相同 prompt 的 prefill 成本被缓存命中抹掉（实测开启时命中率 64%） |
| `--limit-mm-per-prompt` | `{"image": 8}` | 保留多图能力（赛事要求「单图/多图均正常」） |
| `--max-num-batched-tokens` | `4096` | 降低 profiling 与调度峰值 |
| 环境变量 `PYTHONPATH` | `$WORK_DIR/envs/pylibs` | 客户端图像规范化依赖 `qwen_vl_utils` |

### 4.3 恢复基线服务（回退方法）

调优实验（E1/E2/E3）会**杀掉并重启**服务（用不同端口做单变量 A/B）。
实验跑完后恢复「首提基线配置」只需一条命令 —— `run.sh` 是幂等的：

```bash
cd /root/private_data/workspace/robo/submission
source ./env.sh

# 1) 停掉任何残留的实验实例（只杀本任务的进程）
pkill -f "scripts/serve_vllm_patched.py" || true
pkill -f "VLLM::EngineCore" || true
sleep 8

# 2) 用首提基线配置重新拉起（eager / 32768 / 30000 / mem-frac 0.85）
setsid nohup bash run.sh > logs/run_stdout_final.log 2>&1 &
for i in $(seq 1 90); do
  curl -sf --noproxy '*' "http://127.0.0.1:30000/health" >/dev/null && break
  sleep 10
done

# 3) 核对确实回到基线配置（应与 bench/run_config.txt 逐字一致）
grep -m1 '^\[run\]' server.log
```

期望输出（基线配置）：

```text
[run] python3 .../serve_vllm_patched.py --model ... --served-model-name robobrain2.5-8b-nv
      --host 127.0.0.1 --port 30000 --tensor-parallel-size 1 --pipeline-parallel-size 1
      --dtype bfloat16 --trust-remote-code --gpu-memory-utilization 0.85
      --max-model-len 32768 --enforce-eager --no-enable-prefix-caching
      --max-num-batched-tokens 4096 --max-num-seqs 64 --limit-mm-per-prompt={"image":8}
```

> Pod 重启 / SSH 断连后的完整恢复流程见 `../reports/00_RUNBOOK.md` §2；
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

坐标/轨迹叠加图见 `../outputs/annotated/`（白底叠加 bbox / point / trajectory）。

### 5.1 Smoke

```bash
bash smoke.sh
```

`smoke.sh` 依次执行：健康检查 → 模型列表 → 单轮图文 → 多轮图文 → 流式 →
四类任务 + 多图 + 异常输入（最后一项由 `scripts/smoke_service.py` 完成，输出 `smoke_results.json`）。

验证结果（`smoke.log` / `smoke_results.json` 关键部分）：

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

**本项状态：`BLOCKED`。**

`verify.py` 按官方基线文件 `baselines/BAAI__RoboBrain2.5-8B-NV.json` 写死的配置调用服务跑评测。
该基线文件（以及官方评测数据集、官方判分器）**尚未下发**，节点也无外网可取。

按验收标准第 2 节：缺少官方数据或判分器时，精度项必须标记为 `BLOCKED`，
**不得用自行选择的 demo 或自建 GPU 结果替代官方 98% 验收证据**。
因此 `verify_result.json` 中只记录 `status=BLOCKED` 与原因，另附一项**明确标注为内部指标**的
「同卡不同后端逐坐标一致性」——它不能替代官方 GPU 参考一致率。

```bash
python verify.py --selfcheck            # 先确认判分逻辑未被改动（已通过）
source env.sh && python verify.py       # 跑评测，结果写入 verify_result.json
```

判分逻辑自检（`verify.py --selfcheck` 原文）：

```text
[selfcheck] ok   grounding   'The box is [0, 309, 620, 989].' -> {'bbox': [0, 309, 620, 989]}
[selfcheck] ok   grounding   'box <bbox>0 298 622 998</bbox>' -> {'bbox': [0, 298, 622, 998]}
[selfcheck] ok   pointing    'the answer is [(901, 600)]' -> {'points': [{'x': 901, 'y': 600}]}
[selfcheck] ok   trajectory  '[(294, 525, 0.577), (405, 350, 0.607)]' -> {'points': [[294, 525, 0.577], [405, 350, 0.607]]}
[selfcheck] degradation 98 vs 100 = 2.00%
[selfcheck] ok   point degradation 100->98 = 2.00 pt
[selfcheck] ok   agreement judge (exact)
[selfcheck] ok   numeric + choice judges
[selfcheck] PASSED
```

> **重要：`verify.py` 不会在没有真正跑评测的情况下报 PASS。**
> 早期版本的「基线存在」分支只是把基线文件里的数值抄进结果并置 `passed=true`，
> 官方材料到位后会产生**没有证据的 PASS**（触及赛事红线）。现已改为：
>
> | 情况 | 行为 |
> |---|---|
> | 基线文件缺失 | `status=BLOCKED`，`exit 2` |
> | 基线存在但缺 `metrics` 段 | `status=BLOCKED`，`exit 2` |
> | 某个指标的子集 / 参考输出 / 判分器缺失 | 该指标 `BLOCKED`（写明缺哪一样），整体不判 PASS，`exit 2` |
> | 指标全部可跑 | **真正调用服务**逐样本评测，按官方口径算劣化后逐项 PASS/FAIL，`exit 0/1` |
>
> **计分分母的口径**（验收标准 §4.3「解析失败不能剔除出官方计分分母」）：
>
> - 模型输出**解析失败**（正则不匹配）→ 判为**未命中**，**留在分母里**；
> - 请求**失败**（HTTP 错误 / 异常）→ 同样**计入分母并记为未命中**，
>   错误原文逐条记进 `verify_evidence.json`；
>   （早期版本把失败的样本直接 `continue` 掉，等于靠丢样本把一致率做高，已修正）
> - **缺参考输出**的题目无法判分 → 单独计数为 `without_reference`，
>   并在结果 `notes` 里显式披露，不让分母被悄悄缩小；
> - 结果里同时给出 `scored` / `requested` / `errors` / `denominator_basis`，
>   评审可直接核对分母构成。
>
> 上述口径有一条**可执行测试**兜底：`../tools/test_verify_denominator.py`
> （离线，不调用服务）构造 4 道题 —— ①答对 ②请求失败(HTTP 500) ③输出不可解析 ④答错但可解析，
> 断言 `scored=4`、`errors=1`、一致率 **25%**（1/4）而不是剔除失败样本后的 50%，
> 且状态为 FAIL。该测试已实跑通过。
>
> 判分方式按官方口径分三类内置实现，外加可选的外部判分器
> （基线文件里给 `judge` 字段指向 `baselines/judge.py`，其 `judge(response, reference)` 返回布尔值）：
>
> | 类型 | 触发条件 | 判分方式 |
> |---|---|---|
> | `agreement` | T02 具身/坐标一致率 | 官方正则解析响应与参考输出，坐标在 `tolerance_px` 内视为一致；一致率 ≥ `threshold`（默认 98%）判 PASS |
> | `numeric` | gsm8k 之类 | 提取 `\boxed{}`（或 `####`、末位数字）后数值比较 |
> | `choice` | mmlu 之类 | 提取 A–D 选项字母比较 |
>
> 劣化口径按官方原文「较基线劣化 ≤2%，**百分制即 2 pt**」取**绝对分差**，
> 千分制（如 OCRBench）取 20 pt；同时记录相对百分比供对照。
> 图像请求走 `scripts/image_canonical.py` 规范化，与官方 upstream `inference.py`
> 的取图路径一致（否则像素不同会翻转输出，见 `reports/09_vllm_adaptation.md` §3）。
> 逐样本证据写入 `verify_evidence.json`。

> 官方基线文件到位后，`verify.py` 会自动走「按冻结配置跑评测 → 与官方 GPU 基线对比 →
> 劣化 ≤2% 判 PASS」的路径，无需改代码。

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
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | 1024 | 1 | 8 | 325.78 | 0.02 | 25.15 | 181.28 | 802.97 | 834.86 | 39.02 | 39.32 | 40720.70 |
| 4096 | 1024 | 8 | 16 | 113.06 | 0.14 | 144.91 | 1046.99 | 5748.54 | 5810.17 | 49.61 | 50.30 | 56497.34 |
| 4096 | 1024 | 64 | 128 | 651.67 | 0.20 | 201.13 | 1450.34 | 71638.63 | 322252.82 | 219.93 | 462.72 | 296628.01 |
| 16384 | 1024 | 1 | 8 | 375.90 | 0.02 | 21.79 | 432.26 | 3648.41 | 3701.71 | 42.36 | 42.76 | 46986.40 |
| 16384 | 1024 | 8 | 16 | 256.60 | 0.06 | 63.85 | 1267.53 | 13901.87 | 29278.75 | 111.49 | 119.08 | 127953.45 |
| 16384 | 1024 | 64 | 128 | 7694.30 | 0.02 | 17.03 | 337.93 | 2309809.63 | 3478357.63 | 797.18 | 1000.47 | 3125328.93 |
<!-- BENCH_SUMMARY:END -->

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
  `../reports/10a_server_counter_crosscheck.md`
- **修改的模型参数**：**无**。`config.json`、`generation_config.json`、权重、分词器均未改动。
  官方 `generation_config.json` 自带 `do_sample=true / temperature=0.7 / top_p=0.8 / top_k=20`，
  本次评测按官方 `inference.py` 的做法使用 greedy（请求显式传 `temperature=0`）
- **已知限制**：
  - 精度项 `BLOCKED`（缺官方基线文件、评测集与判分器）
  - **官方解析正则的覆盖边界**（验收标准 §4.3 要求单独检查）：
    轨迹深度写成整数（`[(1,2,3)]`）或科学计数法（`1e-3`）、坐标写成浮点（`901.5`）、
    pointing 缺括号、grounding 少一个坐标 —— 这些**官方正则同样不匹配**，
    本节点**保持官方口径不改**，但把原文、解析值与失败原因逐条留证
    （`../outputs/phase4_parse_coverage/parse_coverage.md`，20 条用例中 8 条暴露缺口）；
    解析失败**计入官方计分分母、不剔除**。vLLM 分析模式的 `<bbox>x y x y</bbox>`
    由 `scripts/answer_adapter.py` 做纯格式桥后交给官方正则解析。
  - SGLang 在本节点不可用：镜像内 `aiter` 残缺构建（13 个符号未定义），
    5 个后端全崩；**无法单独计 SGLang 一件**
  - 多图上限 8 张（`--limit-mm-per-prompt '{"image": 8}'`）
  - **非 eager（torch.compile + graph capture）在带 70 GiB 内存看门狗时本容器可以启动**
    （实测 ready 260 s、cgroup 峰值 ~17.75 GiB，远低于 115 GiB 上限），
    但**不作为提交配置**：实测收益只落在并发 1 档（4096/并发 1 输出吞吐 32.81 vs 25.15 tok/s，+30.5%），
    官方口径核心的并发 8 档只有 +0.2%（落在同档两轮波动 <2.3% 内）；
    且非 eager 下输出数值发生变化（与 transformers 基线逐坐标一致率 3/6 → 2/6，
    最大坐标偏差 7/1000 → 30/1000），而官方 ≥98% 精度门槛因缺官方基线为 `BLOCKED`，
    **无法验证该变化无害**。详见 `../reports/11_tuning_experiments.md` §2。
    （历史上 2026-10-27 01:02 首次无看门狗的尝试曾触顶 115 GiB → OOM killer 杀容器 → Pod 重建，
    所以任何非 eager 尝试都必须带看门狗。）
  - **不宣称性能优化达标**：验收标准 §8.2 要求「同口径提升 ≥10%」**且**「优化后精度门槛通过」
    两条同时成立；本节点只满足前者在并发 1 档的一部分，后者因缺官方材料无法满足，
    因此提交配置保持首提基线。核算见 `../reports/05_final.md` §4.6、
    `../reports/08_gap_vs_official_baseline.md` 风险 #13。
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
     → 核算脚本 `../tools/analyze_server_throughput.py`，逐档表
     `../reports/10a_server_counter_crosscheck.md`，
     报告 `../reports/10_phase5_bench.md` §3.7 / §3.8
  11. 收尾流水线的「等压测结束」超时预算 = 启动时刻 + 4 h，**早于**六档压测实际结束
     （最后一档 16384/并发 64 单轮就要 ~2.2 h）→ 流水线会带着 5 档数据提前进入调优实验，
     而调优实验的 A/B 脚本会 `pkill` 服务进程，**直接杀掉正在被压测的服务器**，最后一档数据作废
     → 等待条件改为「压测进程退出 **且** `round2.jsonl` 已有 6 条」，超时预算提到 10 h 且
     超时直接退出；所有会杀服务的实验脚本都加了「压测在跑就拒绝启动」的安全闸。
     这条属于流程性坑，与模型无关，但一旦发生会**静默毁掉整轮数据**
     （见 `../reports/08_gap_vs_official_baseline.md` 风险 #12）
- **其他**：首次启动（eager）约 200 s；容器内存峰值 11.2 GB；
  `MAX_LEN=32768`（官方 `env.sh` 口径）与 `MEM_FRAC 0.85` 共同决定 KV cache 容量 253,312 token
  （`max_model_len` 只影响单请求上限，不改变 KV 分配量：实测 20480 与 32768 下均为 253,312 token）。
