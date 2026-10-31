# 第二/四阶段：适配说明与运行手册

适用节点：`crdnotebook-2114695384420274177-scnjz2s4uc-68757`（K8s 容器，天数智芯 **K500SM_AI**，DTK 26.04）
工作目录：`/root/private_data/workspace/robo`

---

## 1. 适配结论（一句话）

**RoboBrain2.5-8B-NV 已在 K100AI 节点上以官方调用方式跑通**：模型与处理器正常加载，
K500SM_AI 参与实际推理，官方 demo 图 + 官方提示词下 **6/6 样例输出可被官方正则解析**，
且 grounding / pointing / trajectory 三类空间任务的标注图经目视校验 **全部落在正确目标上**。

## 2. 改了什么（全部在工作目录内，公共环境零改动）

| 类别 | 改动 | 原因 |
|---|---|---|
| 新增依赖 | `qwen_vl_utils==0.0.14` → `envs/pylibs` | 官方 `inference.py` 依赖它做 `process_vision_info` |
| 新增依赖 | `matplotlib==3.10.7` + `fonttools/kiwisolver/contourpy/cycler` → `envs/pylibs` | Robo-Dopamine 的奖励曲线可视化 |
| 新增依赖 | `pyparsing==3.2.3` → `envs/pylibs` | 镜像自带 2.4.7，matplotlib 3.10 要求 ≥3.0 |
| 依赖安装方式 | `pip install --no-deps --no-index --target envs/pylibs` | **不触碰系统 site-packages**，避免破坏 torch/vllm/sglang |
| 引入源码 | `third_party/RoboBrain2.5/`、`third_party/Robo-Dopamine/` | 官方提示词模板 / 样例图 / GRM 流水线的权威来源 |
| 环境变量 | `env.sh` 把 HF / Torch / Triton / vLLM / SGLang / XDG / pip 缓存与 `TMPDIR` 全部重定向到 `cache/` `tmp/` | 满足"缓存不落在默认位置"的要求 |

**未改动**：模型权重、模型 config、公共 Python 环境、驱动 / DTK / hyhal、调度资源。
官方 `requirements.txt`（NVIDIA 栈：torch 2.5.1 / vllm 0.7.3 / `nvidia-*-cu12`）**未采用**，
因为与本机 DTK 栈冲突。

## 3. 目录结构

```
/root/private_data/workspace/robo/
├── env.sh                    # 统一环境（DTK/hyhal + 缓存重定向 + PYTHONPATH）
├── scripts/
│   ├── env_check.sh          # 第一阶段环境与模型检查（只读采集）
│   ├── make_samples.py       # 生成离线合成冒烟样例
│   ├── robo_infer.py         # 官方 inference.py 的离线移植（提示词/解析/标注均逐字对齐）
│   ├── minimal_inference.py  # 第二阶段验收脚本（设备参与 + 两次运行一致性）
│   ├── run_tasks.py          # 四类任务功能验证 + 标注图
│   ├── benchmark.py          # 第三阶段基线测量（TTFT / 吞吐 / 峰值显存）
│   ├── tune.py               # 第四阶段受控实验（单变量）
│   ├── benchmark_batch.py    # 第四阶段批处理扩展测量（含答案与基线比对）
│   ├── batch_infer.py        # 批量推理（吞吐提升的实际用法，含标注输出）
│   ├── compare_experiments.py# 实验对比总表生成
│   ├── probe_attn_backend.sh # SGLang attention 后端兼容性探测
│   ├── probe_one_request.py  # 单请求探针
│   ├── serve_sglang.sh       # SGLang 服务化（本节点不可用，见 06 报告）
│   ├── serve_vllm.sh / run_vllm_bench.sh # vLLM 服务化（本节点无可用环境，未验证）
│   └── run_sglang_bench.sh   # SGLang 启动 + 同口径测量
├── samples/                  # 固定样例（官方 demo 图 + 合成图 + 两份 tasks.json）
├── configs/                  # 基线与最佳配置
├── logs/                     # 全部运行日志
├── reports/                  # 环境 / 适配 / 基线 / 调优报告
├── outputs/                  # 推理输出、标注图、测量原始数据
├── envs/                     # 离线 wheel + 隔离安装目录 pylibs/
├── cache/  tmp/              # 所有缓存与临时文件
└── third_party/              # 官方仓库只读镜像
```

## 4. 一条可复现的推理命令

```bash
# 1) 进入远端
ssh dcu_1026

# 2) 加载环境（DTK 运行时 + 缓存重定向）
cd /root/private_data/workspace/robo && source ./env.sh

# 3) 通用 VQA（图像 + 文本）
python3 scripts/robo_infer.py \
  --task general \
  --prompt "What is shown in this image?" \
  --image samples/images/official_affordance.jpg \
  --max-new-tokens 128 --do-sample 0 \
  --out-dir outputs/manual --json-out outputs/manual/general.json

# 4) 视觉定位（输出归一化 bbox）
python3 scripts/robo_infer.py \
  --task grounding \
  --prompt "the person wearing a red hat" \
  --image samples/images/official_grounding.jpg \
  --max-new-tokens 64 --do-sample 0 --plot 1 \
  --out-dir outputs/manual

# 5) 可操作性 / 指代定位（输出归一化点）
python3 scripts/robo_infer.py \
  --task pointing \
  --prompt "the affordance area for holding the cup" \
  --image samples/images/official_affordance.jpg \
  --max-new-tokens 64 --do-sample 0 --plot 1 \
  --out-dir outputs/manual

# 6) 3D 轨迹预测（输出 (x, y, depth) 序列）
python3 scripts/robo_infer.py \
  --task trajectory \
  --prompt "reach for the banana on the plate" \
  --image samples/images/official_trajectory.jpg \
  --max-new-tokens 192 --do-sample 0 --plot 1 \
  --out-dir outputs/manual
```

预期输出（已实测，贪心解码可复现）：

| task | 实测输出 |
|---|---|
| general | `The image shows a white ceramic mug filled with a dark liquid, ...` |
| grounding | `[0, 309, 620, 989]` |
| pointing（杯子把手） | `[(901, 600)]` |
| pointing（两杯之间） | `[(582, 725)]` |
| pointing（马桶） | `[(400, 376)]` |
| trajectory | `[(294, 525, 0.577), (405, 350, 0.607), (525, 300, 0.720)]` |

坐标均为 **0~1000 归一化值**，与上游 `draw_on_image` 的 `rel_to_abs` 约定一致。

## 4.1 批量推理（吞吐提升的实际用法）

SGLang 在本节点的 DTK 内核栈下**不可用**（见 `reports/06_sglang_compatibility.md`），
因此吞吐提升改由已验证路径的批处理获得：

```bash
python3 scripts/batch_infer.py \
  --tasks samples/tasks_official.json --image-dir samples/images \
  --batch-size 4 --plot 1 \
  --out-dir outputs/batch_bs4
```

实测（固定样例，同口径）：

| batch | 聚合 tok/s | 相对 bs=1 | 单请求时延 | 峰值显存 |
|---|---|---|---|---|
| 1 | 11.50 | ×1.00 | 6.87 s | 16.55 GiB |
| 2 | 19.03 | ×1.66 | 4.15 s | 16.64 GiB |
| **4** | **31.24** | **×2.72** | **2.53 s** | **17.07 GiB** |
| 8 | 30.34 | ×2.64 | 2.60 s | 20.42 GiB |

> ⚠️ **批处理必须用 left padding**（脚本已设 `processor.tokenizer.padding_side = "left"`）。
> 用默认的 right padding 会让模型从 pad 位置继续生成，输出被 `Human:` / 一长串 `0` 污染，
> 且生成 token 数虚高（实测 bs=4 生成 512 而非正确的 316）。
> 这类问题**只看"输出非空"或"平均吞吐"是发现不了的**，必须逐条比对答案。

## 5. 完整复现流程

```bash
cd /root/private_data/workspace/robo && source ./env.sh

bash scripts/env_check.sh                    # ① 环境检查 -> logs/env_check_*.log
python3 scripts/make_samples.py              # ② 生成合成样例（官方图已随 third_party 就位）
python3 scripts/minimal_inference.py \
    --image samples/images/official_affordance.jpg \
    --prompt "What is shown in this image?" \
    --max-new-tokens 64 --repeat 2 \
    --out-dir outputs/phase2                  # ③ 第二阶段验收 -> VERDICT PASS
python3 scripts/run_tasks.py \
    --tasks samples/tasks_official.json \
    --out-dir outputs/tasks_official          # ④ 四类任务功能验证 + 标注图
python3 scripts/benchmark.py \
    --tasks samples/tasks_official.json --repeat 3 \
    --out-dir outputs/phase3_baseline         # ⑤ 第三阶段基线
python3 scripts/tune.py --tag <name> --<单变量> \
    --out-dir outputs/phase4                  # ⑥ 第四阶段受控实验
python3 scripts/benchmark_batch.py \
    --batch-sizes 1 2 4 8 --repeat 3 \
    --out-dir outputs/phase4_batch            # ⑦ 第四阶段批处理扩展
python3 scripts/compare_experiments.py \
    --baseline outputs/phase3_baseline/baseline_raw.json \
    --dirs outputs/phase4 --out outputs/phase4/comparison.md   # ⑧ 实验对比总表
```

## 5.1 恢复基线的方法

第三阶段基线由 `outputs/phase3_baseline/baseline_raw.json` + `baseline_summary.md` 固定：

```bash
python3 scripts/benchmark.py --tasks samples/tasks_official.json \
    --image-dir samples/images --repeat 3 --out-dir outputs/phase3_baseline
```

第四阶段实验另存于 `outputs/phase4/<tag>.json` 与 `outputs/phase4_batch/`，
**基线脚本与配置未被覆盖**。

## 6. 服务化

> **现状（2026-10-27 更新）**：SGLang 在本节点**不可用**（5 个 attention 后端 + 2 种 model impl
> 全部崩溃，根因是镜像内 `aiter` 残缺构建与 torch HIP SDPA 内核异常，详见
> `reports/06_sglang_compatibility.md`）。**官方 T02 接受「SGLang 或 vLLM」，两者分别计件**，
> 因此当前走 **vLLM** 这条路。
>
> vLLM 启动必须用带兼容垫片的启动器 `scripts/serve_vllm_patched.py`，
> 直接跑 `scripts/serve_vllm.sh` / `vllm serve` 会在加载权重时崩：
> `AttributeError: 'Qwen3VLTextConfig' object has no attribute 'tie_word_embeddings'`。

```bash
# vLLM（官方认可的指定框架之一；必须用垫片启动器）
python3 scripts/serve_vllm_patched.py \
  --model "$MODEL_PATH" --served-model-name robobrain2.5-8b-nv \
  --host 127.0.0.1 --port 8000 --tensor-parallel-size 1 --dtype bfloat16 \
  --gpu-memory-utilization 0.85 --max-model-len 8192 \
  --limit-mm-per-prompt '{"image": 8}'
# 另开一个终端：探针 + 与 transformers 基线逐坐标比对
python3 scripts/probe_http_request.py --backend vllm --port 8000 \
  --tasks samples/tasks_official.json --image-dir samples/images \
  --repeat 2 --out-dir outputs/probe_vllm
python3 scripts/compare_backends.py \
  --baseline outputs/phase3_baseline/baseline_raw.json \
  --candidate outputs/probe_vllm/probe_result.json \
  --out outputs/probe_vllm/agreement.md

# SGLang（本节点不可用，保留脚本以备其他镜像/节点复用）
bash scripts/serve_sglang.sh --port 30000 --mem-frac 0.85 --ctx 8192
```

> **本版本 sglang 的参数名与上游文档不同**，已在 `serve_sglang.sh` 注释中标注：
> 多模态图像数量上限是 `--limit-mm-data-per-request`（**不是** `--limit-mm-per-prompt`）；
> radix cache（前缀缓存）**默认开启**，对应开关是 `--disable-radix-cache`，
> 本版本**不存在** `--enable-prefix-caching`。

## 7. 兼容性改动清单（相对官方仓库）

| 官方写法 | 本机写法 | 说明 |
|---|---|---|
| `requirements.txt`: torch 2.5.1 + vllm 0.7.3 + `nvidia-*-cu12` | 镜像自带 torch 2.9.0 (dtk2604) + vllm 0.15.1 (dtk2604) + sglang 0.5.12 | NVIDIA 内核在 HCU 上不可用，必须用 DTK 版本 |
| `pip install qwen-vl-utils` | 本机离线 → `envs/pylibs`（`PYTHONPATH` 注入） | 节点无外网 |
| `transformers==4.57.0` | 镜像自带 5.3.0（含 `models.qwen3_vl`） | 已验证 `Qwen3VLForConditionalGeneration` 加载与生成正常 |
| `qwen_vl_utils.process_vision_info` | 同名函数（已离线装好） | 保持官方行为，未改写 |
| `cv2` 标注 | 同上（镜像自带 4.10.0） | 未改写 |
| `decord`（可选） | 缺失；视频走 `cv2.VideoCapture` | GRM 流水线不受影响 |
| `sglang.launch_server --limit-mm-per-prompt` | `--limit-mm-data-per-request` | 本版本参数名不同 |
| `sglang.launch_server --enable-prefix-caching` | 无此参数（radix cache 默认开） | 本版本默认行为不同 |
| `vllm serve` | `python3 scripts/serve_vllm_patched.py`（等价参数） | vLLM 0.15.1 从**子 config** 读 `tie_word_embeddings`，而 transformers 5.3.0 只在**顶层复合 config** 上保留该字段 → 加载权重即崩。垫片用 `sitecustomize` 生效，**不改 site-packages**；等价 diff 见 `patches/0001-vllm-qwen3-tie-word-embeddings.patch` |
| 顶层 `tie_word_embeddings: false` + 权重含 `lm_head.weight` | 垫片按权重索引判定为 **False**（未绑定，vLLM 必须加载 lm_head） | 语义由模型自身文件双重确认，非拍脑袋 |
| 单进程 monkey-patch | 必须放 `sitecustomize.py` | vLLM 用 `VLLM_WORKER_MULTIPROC_METHOD=spawn` 起 EngineCore，父进程补丁**不被子进程继承** |

## 8. 恢复基线的方法

第三阶段基线由 `outputs/phase3_baseline/baseline_raw.json` + `baseline_summary.md` 固定，
对应命令：

```bash
python3 scripts/benchmark.py --tasks samples/tasks_official.json \
    --image-dir samples/images --repeat 3 --out-dir outputs/phase3_baseline
```

任何调优改动若不达标，回到该命令的输出即可对比；基线脚本与配置**未被覆盖**，
第四阶段实验另存于 `outputs/phase4/<tag>.json`。

## 9. 已知限制

- 节点**无外网**：无法安装官方 requirements、无法拉取更多样例/评测集、无法访问 HF；
  一切新增依赖走"本机下载 wheel → scp → 离线安装"。
- 样例仅 6 条图像+文本，**视频/多图 GRM 流水线未纳入基线**（见报告第 7 节）。
- 官方 `requirements.txt` 未被满足，属**有意为之**（NVIDIA 栈与 DTK 冲突）。
- `hy-smi --showpids` 在容器内不可用，资源占用以 `torch.cuda.mem_get_info()` 为准。
- 任务提示词模板逐字取自官方 `inference.py`，**输出格式已验证**，但没有官方
  benchmark 分数可比对（无外网，无法拉取评测集）。
- **SGLang 在本节点不可用**（详见 `reports/06_sglang_compatibility.md`）；指定框架走 vLLM。
- **vLLM 需兼容垫片**才能加载本模型（`scripts/serve_vllm_patched.py` / `envs/vllm_shim/sitecustomize.py`）；
  提交到官方仓库时用 `patches/0001-vllm-qwen3-tie-word-embeddings.patch`。
- **精度项仍为 `BLOCKED`**：官方 GPU 参考输出、官方测试集、官方判分器均未取得，
  不能用自建 demo 顶替（详见 `reports/08_gap_vs_official_baseline.md`）。
- **vLLM 启动期内存风险**：编译 + graph 捕获是内存峰值最高的环节，
  notebook 配额约 6 核 / 115 GB，启动参数需按 `reports/00_RUNBOOK.md` §5.5 降内存配置。
