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
- 提交仓库：代码仓库 `https://github.com/danishZoe/RoboBrain2.5_8b_K100AI`；托管平台可访问链接【待填】；本次交付代码 revision = `4cded9f3faf1d7cfe7ab29b0aff602620c4babc2`（短 `4cded9f`，逐文件校验值见 `MANIFEST.sha256`，详见 `.revision`）

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
git clone https://github.com/danishZoe/RoboBrain2.5_8b_K100AI.git \
  && cd RoboBrain2.5_8b_K100AI
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

### 6.1 赛事官方精度（一致率 ≥98% / 劣化 ≤2%）：`BLOCKED`

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

### 6.2 公开上游基准评测（CV-Bench / RefSpatial-Bench / TraceSpatial-Bench）

**这三个评测集都不是赛事官方验收数据集。** 本节给出的是**公开上游基准**上的
测试有效性与任务质量证据，**不能替代 §6.1 的赛事官方 98% 一致率门槛**。

> **本节所有数字的口径边界**（尤其 §6.2.3 的上采样口径）：
> * **对外上报的数字只有主口径**（CV-Bench `canonical_png` = 88.12、
>   RefSpatial、TraceSpatial，及 §6.3 扩展基准的各主口径），它们与官方实现
>   逐字一致（或属 §6.3 表内已披露的替代口径）；
> * 任何**在评测集上按分数挑出来**的口径（§6.2.3 上采样、§6.2.4 提示词），
>   都只作**归因/诊断**用途，**不替代主口径、不与论文对照、不作验收依据**
>   ——《精度测试验收要求》§4.2：「不得根据某题 mask 或分数选择缩放方法」；
> * 任何**改动评测数据、标签、判分器、聚合规则**的动作本轮**一件都没做**
>   （可逐项核验，见 `reports/16` §5.15.1）。

评测方式：**一律调用各官方评测仓库自带的判分脚本，不改判分逻辑**：

| 基准 | 数据量 | 官方判分入口 | 结论 |
|---|---|---|---|
| CV-Bench | 2638 | `FlagEvalMM/tasks/cv_bench/evaluate.py::get_result` | 见下表 |
| RefSpatial-Bench | 277（Location 100 + Placement 100 + Unseen 77） | `RoboRefer/Evaluation/summarize_acc.py` | 见下表 |
| TraceSpatial-Bench | 100 | `RoboTracer/Evaluation/summarize_acc.py` | 见下表 |

<!-- PUBLIC_BENCH_TABLE:BEGIN -->

#### 6.2.0 精度总表（主口径 · 对外上报口径）

> 本节所有数字由 `tools/inject_accuracy_table.py` 从**官方判分器产出的原始 JSON**
> 现场生成（`outputs/bench_public/{cvbench,refspatial,tracespatial}/summary.json`），
> **不在本文手写**；改口径请改脚本后重跑，不要直接编辑本块。

**表 A：主指标一览（主口径，一屏看完）**

| 基准 | 主指标 | 方向 | 分母 / 口径 | 论文参考 | 本次主口径 | 差值 |
|---|---|---|---|---|---|---|
| CV-Bench | **综合分 CV_score** | ↑ | 2638 / `canonical_png` | 94.58 | **88.12** | **−6.46 pt** |
| CV-Bench | Acc(ADE20K) | ↑ | 2638 / `canonical_png` | 论文只给综合分 94.58 | 77.88 | — |
| CV-Bench | Acc(COCO) | ↑ | 2638 / `canonical_png` | 论文只给综合分 94.58 | 85.96 | — |
| CV-Bench | Acc(Omni3D 3D) | ↑ | 2638 / `canonical_png` | 论文只给综合分 94.58 | 94.33 | — |
| CV-Bench | Acc_2D | ↑ | 2638 / `canonical_png` | 论文只给综合分 94.58 | 81.92 | — |
| CV-Bench | Acc_3D | ↑ | 2638 / `canonical_png` | 论文只给综合分 94.58 | 94.33 | — |
| RefSpatial-Bench | Location | ↑ | 100 / `upstream_scaled` | 论文未公开子集切分 | **68.00** | — |
| RefSpatial-Bench | Placement | ↑ | 100 / `upstream_scaled` | 论文未公开子集切分 | **58.00** | — |
| RefSpatial-Bench | **默认范围均值**（Location+Placement） | ↑ | mean(Location, Placement) / `upstream_scaled` | 60.50（聚合口径未公开） | **63.00** | **+2.50 pt** |
| RefSpatial-Bench | Unseen（重叠子集，单列，不计入默认均值） | ↑ | 77 / `upstream_scaled` | — | 48.05 | — |
| TraceSpatial-Bench | **3D Start Success** | ↑ | 96/100（我们排除 4 条深景病态） / `official` | 83 | **78.13** | **−4.88 pt** |
| TraceSpatial-Bench | **3D End Success** | ↑ | 96/100（我们排除 4 条深景病态） / `official` | 63 | **63.54** | **+0.54 pt** |
| TraceSpatial-Bench | **Overall Success** | ↑ | 96/100（我们排除 4 条深景病态） / `official` | 44 | **46.88** | **+2.88 pt** |
| TraceSpatial-Bench | No-Collision Rate | ↑ | 96/100（我们排除 4 条深景病态） / `official` | — | 81.25 | — |
| TraceSpatial-Bench | 3D Start 距离 (m) | ↓ | 96/100（我们排除 4 条深景病态） / `official` | — | 0.1268 | — |
| TraceSpatial-Bench | 3D End 距离 (m) | ↓ | 96/100（我们排除 4 条深景病态） / `official` | — | 0.1965 | — |
| TraceSpatial-Bench | 2D Start-in-Mask | ↑ | 100/100 / `official` | — | 73.00 | — |
| TraceSpatial-Bench | 2D End-in-BBox | ↑ | 100/100 / `official` | — | 52.00 | — |
| TraceSpatial-Bench | 2D DFD | ↓ | 97/100（计入路径均值） / `official` | — | 0.2055 | — |

**表 B：本节各表的口径性质 —— 哪些数字可以上报**

| 小节 / 表 | 内容 | 口径性质 | 能否作为上报数字 |
|---|---|---|---|
| §6.2.0 表 A | 三个公开基准的**主口径主指标** | 主口径（与上游逐字一致） | ✅ 可以 |
| §6.2.0 表 D | 全部指标 × 全部次口径 | 主口径 + 次口径并列 | 仅「主口径」行可以 |
| §6.2.0 表 C | 同配置重复推理、离线重判分 | 稳定性 / 可复现性 | ✅ 可以（作为证据） |
| §6.2.2 | `Count` 缺口成因（误差分布 + 目视核对） | 归因 | ❌ 仅归因 |
| §6.2.3 | 上采样（`Count` 788 题 / 全量 2638 题） | **在评测集上按分数选出的协议** | ❌ 禁止（验收要求 §4.2） |
| §6.2.4 | 提示词口径、运行间噪声底、评测耗时台账 | 归因 + 台账 | ❌ 仅归因 |

**表 C：稳定性与可复现性**

| 检查 | 结果 |
|---|---|
| CV-Bench 固定样例 × 2 次重复推理 | 原始输出一致 12/12，变化数 0，变化率 0.0，终态全 OK ✅ |
| RefSpatial-Bench 固定样例 × 2 次重复推理 | 原始输出一致 5/12，变化数 7，变化率 0.5833，终态全 OK ✅ |
| TraceSpatial-Bench 固定样例 × 2 次重复推理 | 原始输出一致 3/12，变化数 9，变化率 0.75，终态全 OK ✅ |
| 离线重判分复现（`rescore.py`） | 比较指标数 **12**，不一致 **0**，可复现 **✅** |

**表 D：全量指标明细**（官方判分器逐字产出，含全部次口径；
同一张表也随包提交为 `bench_public/summary_table.md`）

| 基准/指标 | 计划题数 | 有效返回数 | 实际计分数 | 请求/解析失败数 | 参考值 | 本次值 | 差值/单位 | 允许下降 | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| RefSpatial / Location / upstream_scaled（默认范围） | 100 | 100 | 100 | 0 | 60.50%（论文，聚合口径未公开） | 68.00% | 68.00%（官方 CLI） | 未定义 | PASS（公开基准） |
| RefSpatial / Placement / upstream_scaled（默认范围） | 100 | 100 | 100 | 0 | 60.50%（论文，聚合口径未公开） | 58.00% | 58.00%（官方 CLI） | 未定义 | PASS（公开基准） |
| RefSpatial / Unseen / upstream_scaled（重叠子集（单列，不计入默认总分）） | 77 | 77 | 77 | 0 | 60.50%（论文，聚合口径未公开） | 48.05% | 48.05%（官方 CLI） | 未定义 | PASS（公开基准） |
| RefSpatial / Location / upstream_raw（默认范围） | 100 | 100 | 100 | 0 | 60.50%（论文，聚合口径未公开） | 0.00% | 0.00%（官方 CLI） | 未定义 | PASS（公开基准） |
| RefSpatial / Placement / upstream_raw（默认范围） | 100 | 100 | 100 | 0 | 60.50%（论文，聚合口径未公开） | 1.00% | 1.00%（官方 CLI） | 未定义 | PASS（公开基准） |
| RefSpatial / Unseen / upstream_raw（重叠子集（单列，不计入默认总分）） | 77 | 77 | 77 | 0 | 60.50%（论文，聚合口径未公开） | 1.30% | 1.30%（官方 CLI） | 未定义 | PASS（公开基准） |
| RefSpatial / Location / official_scaled（默认范围） | 100 | 100 | 100 | 0 | 60.50%（论文，聚合口径未公开） | 66.00% | 66.00%（官方 CLI） | 未定义 | PASS（公开基准） |
| RefSpatial / Placement / official_scaled（默认范围） | 100 | 100 | 100 | 0 | 60.50%（论文，聚合口径未公开） | 58.00% | 58.00%（官方 CLI） | 未定义 | PASS（公开基准） |
| RefSpatial / Unseen / official_scaled（重叠子集（单列，不计入默认总分）） | 77 | 77 | 77 | 0 | 60.50%（论文，聚合口径未公开） | 45.45% | 45.45%（官方 CLI） | 未定义 | PASS（公开基准） |
| RefSpatial / Location / official_raw（默认范围） | 100 | 100 | 100 | 0 | 60.50%（论文，聚合口径未公开） | 0.00% | 0.00%（官方 CLI） | 未定义 | PASS（公开基准） |
| RefSpatial / Placement / official_raw（默认范围） | 100 | 100 | 100 | 0 | 60.50%（论文，聚合口径未公开） | 1.00% | 1.00%（官方 CLI） | 未定义 | PASS（公开基准） |
| RefSpatial / Unseen / official_raw（重叠子集（单列，不计入默认总分）） | 77 | 77 | 77 | 0 | 60.50%（论文，聚合口径未公开） | 1.30% | 1.30%（官方 CLI） | 未定义 | PASS（公开基准） |
| TraceSpatial / 2D / official / 3D Start | 100 | 100 | 100 | 0 | 83%（论文） | 0.00% | 0.00%；同集官方参照 RoboTracer_Intrinsics_Depth=75% | 未定义 | PASS（公开基准） |
| TraceSpatial / 2D / official / 3D End | 100 | 100 | 100 | 0 | 63%（论文） | 0.00% | 0.00%；同集官方参照 RoboTracer_Intrinsics_Depth=57% | 未定义 | PASS（公开基准） |
| TraceSpatial / 2D / official / Success | 100 | 100 | 100 | 0 | 44%（论文） | 0.00% | 0.00%；同集官方参照 RoboTracer_Intrinsics_Depth=52% | 未定义 | PASS（公开基准） |
| TraceSpatial / 2D / official / 2D Start-in-Mask | 100 | 100 | 100 | 0 | — | 73.00% | 73.00%；同集官方参照 RoboTracer_Intrinsics_Depth=58% | 未定义 | PASS（公开基准） |
| TraceSpatial / 2D / official / 2D End-in-BBox | 100 | 100 | 100 | 0 | — | 52.00% | 52.00%；同集官方参照 RoboTracer_Intrinsics_Depth=43% | 未定义 | PASS（公开基准） |
| TraceSpatial / 2D / official / 2D DFD | 100 | 100 | 100 | 0 | 越低越好；RoboTracer_Intrinsics_Depth 见报告 | 0.2055 | 0.2055 | 未定义 | PASS（公开基准） |
| TraceSpatial / 2D / official / 3D DFD | 100 | 100 | 100 | 0 | 越低越好；RoboTracer_Intrinsics_Depth 见报告 | 0.0000 | 0.0000 | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / official / 3D Start | 100 | 100 | 96 | 0 | 83%（论文） | 78.13% | 78.13%；同集官方参照 RoboTracer_Intrinsics_Depth=75% | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / official / 3D End | 100 | 100 | 96 | 0 | 63%（论文） | 63.54% | 63.54%；同集官方参照 RoboTracer_Intrinsics_Depth=57% | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / official / Success | 100 | 100 | 96 | 0 | 44%（论文） | 46.88% | 46.88%；同集官方参照 RoboTracer_Intrinsics_Depth=52% | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / official / 2D Start-in-Mask | 100 | 100 | 96 | 0 | — | 72.92% | 72.92%；同集官方参照 RoboTracer_Intrinsics_Depth=58% | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / official / 2D End-in-BBox | 100 | 100 | 96 | 0 | — | 51.04% | 51.04%；同集官方参照 RoboTracer_Intrinsics_Depth=43% | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / official / No-Collision Rate | 100 | 100 | 96 | 0 | — | 81.25% | 81.25% | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / official / 3D Start 距离(m) | 100 | 100 | 96 | 0 | 越低越好；RoboTracer_Intrinsics_Depth 见报告 | 0.1268 | 0.1268 | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / official / 3D End 距离(m) | 100 | 100 | 96 | 0 | 越低越好；RoboTracer_Intrinsics_Depth 见报告 | 0.1965 | 0.1965 | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / official / 2D DFD | 100 | 100 | 96 | 0 | 越低越好；RoboTracer_Intrinsics_Depth 见报告 | 0.2057 | 0.2057 | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / official / 3D DFD | 100 | 100 | 96 | 0 | 越低越好；RoboTracer_Intrinsics_Depth 见报告 | 0.4151 | 0.4151 | 未定义 | PASS（公开基准） |
| TraceSpatial / 2D / upstream / 3D Start | 100 | 100 | 100 | 0 | 83%（论文） | 0.00% | 0.00%；同集官方参照 RoboTracer_Intrinsics_Depth=75% | 未定义 | PASS（公开基准） |
| TraceSpatial / 2D / upstream / 3D End | 100 | 100 | 100 | 0 | 63%（论文） | 0.00% | 0.00%；同集官方参照 RoboTracer_Intrinsics_Depth=57% | 未定义 | PASS（公开基准） |
| TraceSpatial / 2D / upstream / Success | 100 | 100 | 100 | 0 | 44%（论文） | 0.00% | 0.00%；同集官方参照 RoboTracer_Intrinsics_Depth=52% | 未定义 | PASS（公开基准） |
| TraceSpatial / 2D / upstream / 2D Start-in-Mask | 100 | 100 | 100 | 0 | — | 58.00% | 58.00%；同集官方参照 RoboTracer_Intrinsics_Depth=58% | 未定义 | PASS（公开基准） |
| TraceSpatial / 2D / upstream / 2D End-in-BBox | 100 | 100 | 100 | 0 | — | 52.00% | 52.00%；同集官方参照 RoboTracer_Intrinsics_Depth=43% | 未定义 | PASS（公开基准） |
| TraceSpatial / 2D / upstream / 2D DFD | 100 | 100 | 100 | 0 | 越低越好；RoboTracer_Intrinsics_Depth 见报告 | 0.2682 | 0.2682 | 未定义 | PASS（公开基准） |
| TraceSpatial / 2D / upstream / 3D DFD | 100 | 100 | 100 | 0 | 越低越好；RoboTracer_Intrinsics_Depth 见报告 | 0.0000 | 0.0000 | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / upstream / 3D Start | 100 | 98 | 95 | 2 | 83%（论文） | 63.16% | 63.16%；同集官方参照 RoboTracer_Intrinsics_Depth=75% | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / upstream / 3D End | 100 | 98 | 95 | 2 | 63%（论文） | 56.84% | 56.84%；同集官方参照 RoboTracer_Intrinsics_Depth=57% | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / upstream / Success | 100 | 98 | 95 | 2 | 44%（论文） | 37.50% | 37.50%；同集官方参照 RoboTracer_Intrinsics_Depth=52% | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / upstream / 2D Start-in-Mask | 100 | 98 | 95 | 2 | — | 61.05% | 61.05%；同集官方参照 RoboTracer_Intrinsics_Depth=58% | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / upstream / 2D End-in-BBox | 100 | 98 | 95 | 2 | — | 46.32% | 46.32%；同集官方参照 RoboTracer_Intrinsics_Depth=43% | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / upstream / No-Collision Rate | 100 | 98 | 95 | 2 | — | 83.16% | 83.16% | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / upstream / 3D Start 距离(m) | 100 | 98 | 95 | 2 | 越低越好；RoboTracer_Intrinsics_Depth 见报告 | 0.2912 | 0.2912 | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / upstream / 3D End 距离(m) | 100 | 98 | 95 | 2 | 越低越好；RoboTracer_Intrinsics_Depth 见报告 | 0.2430 | 0.2430 | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / upstream / 2D DFD | 100 | 98 | 95 | 2 | 越低越好；RoboTracer_Intrinsics_Depth 见报告 | 0.2582 | 0.2582 | 未定义 | PASS（公开基准） |
| TraceSpatial / 3D / upstream / 3D DFD | 100 | 98 | 95 | 2 | 越低越好；RoboTracer_Intrinsics_Depth 见报告 | 0.5374 | 0.5374 | 未定义 | PASS（公开基准） |
| CV-Bench / Acc(ADE20K) | 2638 | 2638 | 2638 | 0 | 94.58%（论文 Table 2） | 77.88% | 77.88%（官方 FlagEvalMM get_result） | 未定义 | PASS（公开基准） |
| CV-Bench / Acc(COCO) | 2638 | 2638 | 2638 | 0 | 94.58%（论文 Table 2） | 85.96% | 85.96%（官方 FlagEvalMM get_result） | 未定义 | PASS（公开基准） |
| CV-Bench / Acc(Omni3D) | 2638 | 2638 | 2638 | 0 | 94.58%（论文 Table 2） | 94.33% | 94.33%（官方 FlagEvalMM get_result） | 未定义 | PASS（公开基准） |
| CV-Bench / Acc_2D | 2638 | 2638 | 2638 | 0 | 94.58%（论文 Table 2） | 81.92% | 81.92%（官方 FlagEvalMM get_result） | 未定义 | PASS（公开基准） |
| CV-Bench / Acc_3D | 2638 | 2638 | 2638 | 0 | 94.58%（论文 Table 2） | 94.33% | 94.33%（官方 FlagEvalMM get_result） | 未定义 | PASS（公开基准） |
| CV-Bench / CV-Bench 综合分 | 2638 | 2638 | 2638 | 0 | 94.58%（论文 Table 2） | 88.12% | 88.12%（官方 FlagEvalMM get_result） | 未定义 | PASS（公开基准） |

> 表 D 的「实际计分数」是**该指标实际计入的分母**，三种来源必须分清：
> * **TraceSpatial 3D 子集 96/100**：这 4 条是**我们主动排除**的深景病态样本（索引 56/78/82/98，官方判分单条 >65 min 仍未结束，见 §6.2.1）——
  **官方判分器没有排除任何样本**，它自己的 `[Skip]` 数为 0（`upstream` 变体 1）；
  被排除的索引写在产物 `excluded_sample_indices` 字段里，可审计；
> * **2D 子集 2D DFD 计入路径均值 97/100**：这 3 条是**官方脚本自己**因归一化后含 NaN / 尺寸无效而静默丢弃的；
> * **RefSpatial `Unseen` 77 条**是**重叠子集**，不计入默认范围均值。


<!-- /PUBLIC_BENCH_TABLE -->

复现命令：

```bash
cd /root/private_data/workspace/robo
export PYTHONPATH="$PWD/envs/pylibs_core:$PWD/third_party/upstream_eval/FlagEvalMM:$PWD/scripts:${PYTHONPATH:-}"
source /opt/dtk/env.sh          # 解决 libgalaxyhip.so.5

bash tools/start_bench_service.sh eager 30000     # 首提基线配置
python3 submission/bench_public/manifest.py --out-dir outputs/bench_public/manifest
python3 submission/bench_public/run_refspatial.py  --url http://127.0.0.1:30000/v1/chat/completions \
        --run-id baseline-eager-refspatial  --out-dir outputs/bench_public/refspatial
python3 submission/bench_public/run_cv_bench.py    --url http://127.0.0.1:30000/v1/chat/completions \
        --run-id baseline-eager-cvbench    --out-dir outputs/bench_public/cvbench
python3 submission/bench_public/run_tracespatial.py --url http://127.0.0.1:30000/v1/chat/completions \
        --run-id baseline-eager-tracespatial --out-dir outputs/bench_public/tracespatial
python3 submission/bench_public/stability.py --url http://127.0.0.1:30000/v1/chat/completions \
        --out-dir outputs/bench_public/stability --n 12
python3 submission/bench_public/rescore.py  --bench-root outputs/bench_public   # 离线重判分
python3 submission/bench_public/build_summary_table.py --bench-root outputs/bench_public \
        --out outputs/bench_public/summary_table.md
```

> 📦 **本包自包含性**：上面 `submission/bench_public/*.py` **就在本包内**（`bench_public/` 目录）；
> `tools/*.sh|py` 是工作区 `/root/private_data/workspace/robo/tools/` 下同名脚本的**镜像**，
> 本包内 `tools/` 目录已包含全部被本文引用的脚本，可直接取用。
> `envs/`、`third_party/`、`outputs/` 属于运行环境与产物，**不在本包内**，按 §6.2 的命令在
> 工作区 `/root/private_data/workspace/robo` 下执行。

> ⚠️ **官方 TraceSpatial 3D 判分的两个硬依赖（2026-10-28 实测）**：
>
> 1. **必须用 open3d 0.18.0**。0.19/0.20 改了 `ScalableTSDFVolume` 的体素语义，
>    官方脚本给的 `voxel_length=0.02, sdf_trunc=0.08` 在新语义下 **TSDF 恒抽不出点**
>    → 官方脚本抛 `ValueError: TSDF extracted no points` →
>    碰撞检查被官方脚本自己捕获成 `collision_flag=False` →
>    **`No-Collision Rate` 恒 1.0、`Overall Success` 偏乐观**。
>    修复方式：隔离前缀 `envs/pylibs_o3d_v2`（open3d 0.18.0 + numpy 1.26.4 + sklearn 桩件），
>    **官方 `summarize_acc.py` 一字未改**，用 `ROBO_O3D_PREFIX` 切换判分子进程的 `PYTHONPATH`。
> 2. **官方判分是单核的**。修复后实测单样本 ~7~30 s（16 路并发时），
>    整集 100 条单进程 ≈ 12 分钟。用 `tools/run_ts_score_3d.sh`
>    （逐样本分片并行 + cgroup 内存闸门）跑完 4 个组合 × 100 题 ≈ 30 分钟；
>    **分片合并与整集单进程是否等价，由 `tools/verify_3d_merge.sh` 逐指标对拍证明**。

```bash
# 3D 判分（含 open3d 前缀切换）
export ROBO_O3D_PREFIX=$PWD/envs/pylibs_o3d_v2
bash tools/run_ts_score_3d.sh 16 45
python3 submission/bench_public/merge_ts_scores.py --bench-root outputs/bench_public
bash tools/verify_3d_merge.sh
```

#### 6.2.1 官方 3D 判分的两个**会让报告出错**的陷阱（2026-10-28 实测）

> 1. **必须用 open3d 0.18.0**（见上）。0.19/0.20 的 `ScalableTSDFVolume` 改了体素语义，
>    官方给的 `voxel_length=0.02, sdf_trunc=0.08` 在新语义下 **TSDF 恒抽不出点** →
>    碰撞检查恒 False → `No-Collision Rate` 恒 1.0、`Overall Success` **偏乐观**。
> 2. **官方对「无效样本」是静默丢弃的**：归一化后含 NaN / 尺寸无效的样本、
>    以及 `dfd > 100` 的样本都不打印任何标记就丢掉。唯一可观测信号是
>    **有效样本数为 0 时打印空 dict `{}`**。逐样本分片时一个分片恰好一个样本，
>    于是合并口径可精确还原为「**空 dict 的分片分母不加**」。
>    实测 100 片里有 3 片是空 dict：分母应为 97，误记成 100 会让 2D `dfd`
>    偏 **3.0%**（0.2055 → 0.1993）。
>    防线：`tools/remerge_ts_3d.py`（用已落盘 `per_chunk` 重算并逐指标对比）。
> 3. **深景病态样本 —— 由「我们」主动排除，不是官方判分器排除**：样本 56 / 78 / 82 / 98
>    单条官方判分 **>65 min 仍未结束**（open3d 修复后 TSDF 真生效 → 在大目标点云上做
>    `cluster_dbscan(eps=0.02, min_points=10)`，近似二次复杂度）。
>    官方判分器**本身没有排除任何样本**，它自己的 `[Skip]` 机制
>    （无内参 / 解析失败 / 无深度 / 反投影失败）在本次 official 变体上计数为 **0**。
>    这 4 条是**我们**在判分前用 `tools/run_ts_score_3d_96.sh` 主动排除的（`--exclude`），
>    排除索引写进产物的 `excluded_sample_indices` 字段，**可审计**；
>    其余 96 条单条 15~45 s，16 路并发整集 **941 s** 跑完。
>    ⇒ **3D 子集分母是 96/100，属「我们主动排除」而非「官方判定无效」**，
>    这一点在 §6.2.0 表 D 的注释里也写明了。
>    对拍脚本取**前 50 条干净样本**，阶段 A / B 取同一区间。

#### 6.2.2 CV-Bench `Count` 缺口：成因已定位（可证实）

论文 Table 2 的 CV-Bench 94.58 与我们实测的 88.12 差 6.46 pt，
**缺口 100% 落在 `Count` 子任务**（788 题 72.34%）。逐项排除后（判分链路 / 提示词复刻 /
token 预算 / 抽图压缩 / thinking / 图像编码协议 8 组因子），我们对失分样例做了
**误差分布 + 人工目视核对**（导出 15 张图逐张看，`outputs/bench_public/cvbench_count_samples/`）：

| 误差（预测个数 − 真值个数） | −3 | −2 | −1 | **0** | +1 | +2 |
|---|---|---|---|---|---|---|
| 题数 | 6 | 37 | 111 | **570** | 56 | 5 |

* **|误差| ≤ 2 占 98.9%**；抽取形态 769/788 纯字母、**0 例数字型输出**；
  字母分布与真值几乎重合 → 模型在**真实地数**，不是格式错；
* 目视核对把失分拆成两类：
  * **真实感知漏检**（如"洗碗机"样例：GT=1 正确，模型答 0）；
  * **实例标注口径不一致**（"吊灯"样例：肉眼只有 1 盏，ADE20K 标了 3 个实例；
    "斑马"样例：肉眼 5 匹，COCO 只标 4 匹，模型答 5 在人类视角下是对的）。
* **第二类无法用任何推理侧参数修复** —— 它取决于模型计数语义是否与标注约定对齐，
  而论文 NV 权重的训练数据未公开。剩余杠杆在**权重/训练数据**，不在推理栈。

#### 6.2.3 输入分辨率因子：**只作归因，不作提分**（非官方口径，单列）

模型原生 `preprocessor_config.json` 为 `size = {longest_edge: 16777216, shortest_edge: 65536}`
→ 256×256 的图只有 **64 个视觉 token**。把图**等比上采样**（LANCZOS，不缩小原图，
总像素 ≤3,000,000）会加密 ViT 的 patch 网格，小目标能占到更多 token。
只跑 `Count` 788 题、图像与判分口径不变，唯一变量 = 缩放策略：

<!-- UPSCALE_TABLE:BEGIN -->

结果（`Count` 788 题，同一会话，唯一变量 = 缩放策略；判分 = 官方 FlagEvalMM `get_result`）：

| 缩放策略 | Acc(ADE20K) | Acc(COCO) | Acc_2D | Δ vs canonical |
|---|---|---|---|---|
| `canonical_png`（同会话对照） | 63.74 | 78.25 | 71.00 | — |
| **`up2_png`** | **65.79** | **82.06** | **73.93** | **+2.93** |
| **`up2_jpeg95`** | **65.79** | **82.74** | **74.27** | **+3.27** |

<!-- /UPSCALE_TABLE -->

* `canonical_png` 对照复现出与另一次实验**完全一致**的 63.74 / 78.25
  → 本轮对 `common.py` 的改动是纯增量的、没动基线路径，且 CV-Bench 可复现。
* **机制**：模型原生 `preprocessor_config.json` 的 `shortest_edge=65536`
  → 256×256 的图只有 **64 个视觉 token**；2× 上采样（LANCZOS、不缩小原图、
  总像素 ≤3,000,000）加密 ViT 的 patch 网格，让小目标占到更多 token，
  与"洗碗机漏检"这类失分（见 §6.2.2）机制一致。
* **显著性**：逐题配对符号检验，错→对 61 / 对→错 18，**p = 1.27e-06**；
  同配置两次运行的噪声底只有 0.29 pt（p=0.6875，不显著）⇒ 增益是真的。

⚠️⚠️ **合规性声明（本节数字的正确用法，必读）**

1. **上采样不是官方协议**：上游 `RoboBrain2.5/inference.py` 走
   `qwen_vl_utils.process_vision_info`（只把长宽取 28 的倍数，**不放大**），
   官方 `FlagEvalMM encode_image` 也只按 `max_long_side` **缩小**、按 `min_short_side` **补白边**。
2. **它没有篡改 benchmark**：题目、GT 标注（SHA256 逐字相同）、官方判分器
   （`get_result` 零 patch）、官方聚合公式、提示词、坐标与阈值**全部未改**；
   唯一变动是**送进模型的图像编码**，而这本来就属于推理栈，
   且放大后仍在模型声明的 `longest_edge=16,777,216` 能力范围内。
3. **但它是「在评测集上按分数挑出来的协议」**：《精度测试验收要求》§4.2
   明确写「**不得根据某题 mask 或分数选择缩放方法**」。
   ⇒ 因此 **本口径的数值带选择偏差，不是无偏的准确率估计**：
   * **不得**替代主口径（`canonical_png` = **88.12**）作为上报数字；
   * **不得**与论文的 94.58 并列比较或作为"更接近论文"的证据；
   * **不得**作为精度结论或赛事官方 98% 一致率门槛的证据
     （官方参考输出是用官方图像处理产生的，改分辨率会直接掉一致率）。
4. **它的合法用途只有一个：归因**。这组实验把「提示词 / 图像编码格式 / 解码参数」
   三个因子全部排除，**定位到「输入分辨率 → 视觉 token 密度」是 `Count` 缺口的机制**，
   从而说明剩余缺口（真实小目标漏检 + 实例标注口径不一致）**在推理栈上已无可动之处**，
   必须回到权重/训练数据解决。
5. **要做成合规的精度提升，正确路径是**：在**独立验证集**上选协议 →
   在**留出集**上验证 → 冻结协议后再跑测试集。
   本轮**没有留出集**，所以这件事**未完成**，如实标注。
6. **主口径 `canonical_png` 本身不是挑出来的**（这是最该被追问的一点，
   §5.5 的 8 组协议矩阵给出硬证据）：`raw_jpeg95` 71.67、`long1500_jpeg95` 71.62
   都比主口径 `canonical_png` 71.00 **高 0.6~0.7 pt**。
   主口径是**先验按「与上游 `qwen_vl_utils` 逐字一致」**选定的，
   不是按分数选的，而且选中的是 8 组里**偏保守**的一个
   ⇒ **对外上报的 88.12 不含协议选择偏差。**

**（全量诊断结果）**。为回答「输入分辨率这个因子在全量上能拿回多少」，
又跑了一次**全量 2638 题**（`up2_png`、并发 48、**2837.5 s / 47.3 min**，在 59 min 硬上限内完成，
2638/2638 终态 `ok`）：

<!-- UPSCALE_FULL_TABLE:BEGIN -->

| 指标 | 主口径 `canonical_png` | `up2_png`（诊断口径） | Δ |
|---|---|---|---|
| Acc(ADE20K) | 77.88 | 78.99 | **+1.11** |
| Acc(COCO) | 85.96 | 88.45 | **+2.49** |
| Acc_2D | 81.92 | 83.72 | **+1.80** |
| Acc(Omni3D 3D) | 94.33 | 93.83 | **−0.50** |
| **CV_score（官方综合分）** | **88.12** | **88.78** | **+0.66** |

<!-- /UPSCALE_FULL_TABLE -->

* 2D 分支 +1.80 pt（含 `Relation`，把 `Count` 的 +2.9~3.3 pt 摊薄）；
  **3D 分支反而 −0.50 pt**（上采样对 3D 空间感知无益）；
* 官方综合分净 **+0.66 pt**，只有「与论文 94.58 的差距（6.46 pt）」的**约 1/10**
  ⇒ **6.46 pt 缺口不可能由输入分辨率解释**，必须回到权重/训练数据一侧解决。
* 该表同样按 §6.2.3 的口径声明处理：**诊断口径，不替代主口径 88.12**。

```bash
# 第二口径复现（只跑 Count 子任务；输出仅用于归因，不用于上报）
bash tools/exp_cvbench_upscale.sh
# 配对显著性检验 + 答案格式审计
python3 tools/paired_sign_test.py
python3 tools/audit_prompt_format.py
```

#### 6.2.4 提示词口径、运行间噪声底与评测耗时分级

**（1）提示词口径**。CV-Bench 数据集自带 `prompt` 字段（选项写成 `(A) 3`、
结尾 "Select from the following choices."），与 FlagEvalMM 构造的
（`A. 3` + "Answer with the option's letter from the given choices directly."）**逐字不同**。
两条官方口径都跑过（`Count` 788 题，图像协议固定 `canonical_png`）：

| 提示词口径 | Acc(ADE20K) | Acc(COCO) | Acc_2D |
|---|---|---|---|
| **`flagevalmm`（主口径，与官方逐字一致）** | **64.33** | 78.25 | **71.29** |
| `dataset`（数据集自带 prompt） | 61.70 | 78.25 | 69.97 |

**答案格式审计**（排除"抽取伪影"可能）：`dataset` 口径下模型 767/788 输出 `(C) 1` 形式，
官方判分器对这种格式**比严格字母归一化更宽松**（多认 8 题），
但即便如此 `dataset` 口径在 ADE20K 上**仍差 2.63 pt**（按严格归一化口径差 4.97 pt），
远超噪声底 ⇒ **「dataset 口径更差」是真的**。
⚠️ 反面提醒：自建抽取器若不做格式归一化，在该口径下会得到 **0/788 全错**的假象。

**（2）运行间噪声底与配对显著性检验**。逐字**同配置**两次运行（788 题）：
原始输出 10/788（1.27%）不同、判分层面 6/788 翻转、官方 `acc_2d` 差 **0.29 pt**
（符号检验 p=0.6875，不显著）⇒ **噪声底 ≈ 0.3 pt**。
据此，本报告里 `Count` **小于 ~0.6 pt 的分差一律视为噪声**；
上采样 +2.93/+3.27 pt 的 p 值为 1.3e-06 / 4.7e-07，**远在噪声之上**。

**（3）评测耗时分级与台账**（用户要求"优先做快的评测、不要老卡在慢评测上"）：

| 档位 | 预估墙钟 | 处置 | 本轮实例 |
|---|---|---|---|
| L1 | < 5 min | 随时跑 | RefSpatial/TraceSpatial 推理、误差分布统计、合并口径重算 |
| L2 | 5~20 min | 主力档 | CV-Bench `Count` 788 题各组、TraceSpatial 3D 兜底判分（941 s） |
| L3 | 20~60 min | **设硬上限** | CV-Bench 全量 2638 题、8 组协议矩阵 |
| L4 | > 60 min | **不跑**或先排除病态样本 | 3D 整集判分（含深景病态样本）、`up4` 上采样 |

* 3D 整集判分卡住 > 1 h → **主动停止**，改用「排除 56/78/82/98 + 96 条兜底口径」，
  **941 s** 跑完 4 个组合；
* `up4` 单组 > 23 min 无结果 → **主动停止**；
* 全量 `up2`（L3）→ 并发 16→48，并设 **59 min 硬上限**（`tools/watch_up2_full.sh` 到点自动止损）。
* 台账由脚本自动生成：`outputs/bench_public/eval_timing.md` / `.json`
  （`tools/build_eval_timing.py`）。
  ⚠️ 统计陷阱：`scores_3dchunk_*.json` 的 `wall_s_total` 是**各分片墙钟之和**（并发累加，
  并发 16 时约为真实墙钟的 4 倍），**必须用 `wall_s_all_jobs`**。

```bash
python3 tools/build_eval_timing.py     # 重新生成耗时台账
```

#### 6.2.5 TraceSpatial 3D「病态样本」根因勘误与最大可行覆盖（2026-10-29 更新）

> **勘误**：§6.2/L17 曾把 4 条深景样本（56/78/82/98）判分超时归因于
> "DBSCAN 在大点云上近似二次复杂度"，**该归因错误**（实测 DBSCAN/目标点云
> 链路 0.0s 完成）。真实根因（py-spy 栈 + 端到端复现双重证实）：
> 模型对这几条样本输出**恒定退化轨迹** → 官方距离均匀插值产生 NaN →
> 官方 3D 路径（无 NaN 过滤，2D 路径有）调用递归版 `discrete_frechet_distance`，
> 其备忘录判据对 NaN 永假 → **指数级递归、永不终止**。100/100 全量官方判分
> 对任何变体组合都不可达（16 组合预筛：5 挂 / 2 Skip / 9 可完成）。
> 完整证据链见 `reports/20` §6。

**TraceSpatial 3D 最大可行覆盖口径**（96 条兜底 + 可完成样本补跑，逐分片
`sample_index` 标注、排除索引与根因写入产物 `outputs/bench_public_ts3d_best/`）：

| 组合 | 分母 answers/eval | 3D Start | 3D End | Overall | 排除 |
|---|---|---|---|---|---|
| 基线 eager / official | 98/98 | 78.57 | 63.27 | 46.94 | 56, 78 |
| 候选 graph / official | 98/98 | 80.61 | 66.33 | 48.98 | 56, 78 |
| 基线 eager / upstream | 99/97 | 61.86 | 56.70 | 36.36 | 82 |
| 候选 graph / upstream | 99/97 | 61.86 | 61.86 | 38.38 | 98 |

与论文 83/63/44 对照：official 变体 End≈63、Success≥44，结论方向与 96 口径一致。

### 6.3 扩展公开基准（论文 Table 2/3 补齐 · 2026-10-29）

> 明细、协议披露、bug 修复复盘与诊断实验见 **`reports/20`**；
> **A800 交叉验证见 `reports/21`**。论文分仅作同量级对照（协议未公开）；
> Robo-Dopamine 为可选视频扩展（需独立 GRM 模型），本轮 `NOT_RUN`。

| 基准 | 题数 | K100 (DCU) | A800 (NVIDIA) | 论文 NV | 判分代码 |
|---|---|---|---|---|---|
| CrossPoint-Bench | 1000 | **75.20** | — | 75.40 | 官方 `calc_score.py` + `relative_1000` 零改动 |
| EmbSpatial-Bench | 3640 | **77.77** | — | 75.58 | ⚠️ 自建 MC（官方未发布代码，已披露） |
| RoboSpatial-Home | 350 | **62.0** | 62.86 | 73.03 | 官方 `eval_pregenerated_results` 零改动 |
| Q-Spatial-Bench | 101 | **60.4%** | 69.3% | 73.53 | 官方公式复刻 + 披露兜底抽取 |
| MSMU-Bench | 987 | 定量 47.0~76.4%（δ1.25~1.95）/ 定性 63.8% | — | 64.17 | 官方 `judge_prompt`（judge=本服务，greedy） |
| ShareRobot-T | 200 | DFD **0.181** | — | 0.1164 | FlagEvalMM **PR#84** 零改动 |
| ShareRobot-A | 200 | mIoU 0.019（GT 原始分辨率与缩略图参考系错配，不可靠） | — | —（论文未报） | 同上 |
| VABench-V | 300 | DFD **0.413** | — | 0.1281 | ⚠️ 官方判分未发布，PR84 同款几何指标（披露替代） |

**跨硬件对照（`reports/21`）**：RoboSpatial A800 62.86 vs K100 62.0（Δ0.86pt，
同一保守偏置）——缺口在 bench 配置侧（官方 harness 提供深度输入；喂深度后
compat 41.0→54.3%）。Q-Spatial A800 69.3% vs K100 60.4%（+8.9pt）：逐题配对
分析显示净差几乎全来自 ratio≈2 判定边界翻转（19/101 题在线界附近），属长链
数值估计对 bf16 微分歧与版本差的高敏感性，**非系统性算子错误**；A800 距论文
仍差 4.2pt，缺口主因仍是提示引导/模型估计质量（官方模板间变体实验：裸题 38.6%
vs steps 模板 60.4%）。

**偏差归因一览**（全部非算子侧；算子保真另有 CV-Bench A800↔DCU 99.6% 逐题
一致旁证，`reports/19_a800_cross_validation.md`）：

| 偏差 | 归因 |
|---|---|
| CV-Bench −6.46pt | `Count` 子任务：小目标漏检 + 实例标注口径（15 例人工目视核实） |
| RoboSpatial −11pt | 模型保守偏置 + 官方跑法疑用深度输入（已实证 +13pt） |
| Q-Spatial −13pt | 提示引导/估计质量（模板变体差 21.8pt；跨硬件边界翻转 +8.9pt） |
| ShareRobot-T / VABench-V | 初版坐标适配层括号缺陷虚高（0.181/0.413 为修复后值） |

```bash
# 复现（K100 远端，非 eager 服务 graph 30000 起服务后）：
bash tools/run_new_benches_1.sh     # EmbSpatial + CrossPoint
bash tools/run_new_benches_2.sh     # RoboSpatial（等 1）
bash tools/master_bench_queue.sh    # Q-Spatial + MSMU + ShareRobot + VABench（等 2）
# runner：submission/bench_public/run_{embspatial,crosspoint,robospatial,qspatial,msmu,sharerobot,vabench}.py
```

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
| 16384 | 1024 | 64 | - | - | - | - | - | - | - | - | - | - |
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
