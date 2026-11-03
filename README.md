# RoboBrain2.5-8B-NV on K100-AI + vLLM 0.15.1

## 1. 简介

本项目完成 RoboBrain2.5-8B-NV 在单张 K100-AI、vLLM 0.15.1、BF16 环境的图文推理适配与性能测试。模型基于 Qwen3-VL，面向视觉理解、空间定位和轨迹等机器人任务。

权重由运行方挂载；代码从 GitLab 获取；运行环境使用 SCNet 基础镜像及本次整理的系统盘内容。当前实例已完成图文 smoke 和性能测试，**保存镜像后重新创建实例的复现尚未验证**。

- 模型：[SCNet AI 社区](https://www.scnet.cn/ui/aihub/models/sugon_scnet/RoboBrain2.5-8B-NV)
- 基础镜像：`appstore.scnet.cn:5000/aihub/dcu/zhr5535/jupyter-sglang5.12-ubuntu22.04-dtk26.04-py3.10-qwen3.6:v1.0`；本次保存的镜像地址和版本 **待填写**。
- 代码：[GitLab 仓库](https://developer.sourcefind.cn/codes/g_exhyd6h6/robobrain2.5_8b_k100ai)；默认使用 `main`，以实际提交版本为准。
- 上游：[RoboBrain2.5 模型说明](https://huggingface.co/BAAI/RoboBrain2.5-8B-NV)。

克隆并进入仓库：

```bash
git clone https://developer.sourcefind.cn/codes/g_exhyd6h6/robobrain2.5_8b_k100ai.git
cd robobrain2.5_8b_k100ai
```

主要入口：`env.sh` 配置环境，`run.sh` 启动服务，`smoke.sh` 验证图文，`verify.py` 执行精度检查，`bench.sh` 测试性能。原始日志在 `bench/`，公开评测细节在 `reports/`；完整历史说明见 [归档报告](docs/README-full-20261103.md)。

## 2. 验证环境

| 项目 | 已测环境 |
| --- | --- |
| 平台 / 卡 | SCNet Notebook；K100-AI × 1，运行时设备名 `K500SM_AI`，约 64 GiB |
| 基础环境 | DTK 26.04；Python 3.10.12；HIP 6.3.26113 |
| 主要软件 | PyTorch `2.9.0+das.opt1.dtk2604`、vLLM `0.15.1+das.opt1.alpha.dtk2604`、Transformers 5.3.0、Triton `3.5.1+das.opt1.dtk2604` |
| 附加依赖 | `qwen-vl-utils==0.0.14`，位于系统盘 `/opt/robo-image-v1/pylibs-clean/` |
| 运行参数 | BF16、TP=1、最大长度 32768、显存利用率 0.85、端口 30000 |
| 模型路径 | `/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV`，可通过 `MODEL_PATH` 指定 |
| 平台信息 | 区域、系统盘及文件存储规格、平台截图 **待补充**；未配置外部服务映射 |

环境原始记录见 [env.log](env.log)；历史详细说明及平台证据清单见 [归档报告](docs/README-full-20261103.md) 和 [platform/README.md](platform/README.md)。

## 3. 适配优化方案

- 沿用基础镜像的 PyTorch、vLLM、Transformers 和 DTK；使用隔离目录中的 `qwen-vl-utils`，未替换核心软件包。
- `envs/vllm_shim/sitecustomize.py` 为当前 vLLM 与 Transformers 的配置接口补充 `tie_word_embeddings=False`；对应差异见 [patch](patches/0001-vllm-qwen3-tie-word-embeddings.patch)，不写入系统 site-packages。
- 默认配置使用 Graph、`ROCM_AITER_FA` 和 rocBLAS 选优差异文件。`scripts/make_tuned_rocblas_lib.sh` 在工作目录组装并校验候选库，保留 stock 库；`USE_TUNED=0` 回退 eager 和 stock 路径。
- 图像规范化和空间答案解析分别见 `scripts/image_canonical.py`、`scripts/answer_adapter.py`。这些适配与相关质量限制见 [适配报告](reports/09_vllm_adaptation.md) 和 [调优报告](reports/25_tuning_delivery_20261031.md)。

## 4. 服务启动脚本

先检查设备分配和占用，挂载模型，克隆本仓库并进入代码目录。在本次系统盘环境中：

```bash
export WORK_DIR=/root/private_data/workspace/robo/repro_run
export MODEL_PATH=/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV
export VLLM_NO_USAGE_STATS=1
bash /opt/robo-image-v1/prepare.sh "$PWD" "$WORK_DIR"
bash run.sh
bash env_check.sh
```

服务使用 `scripts/serve_vllm_patched.py` 启动，配置为单卡、BF16、`--max-model-len 32768`、`--gpu-memory-utilization 0.85`、`--max-num-batched-tokens 4096`、`--max-num-seqs 64`，禁用 prefix caching。完整参数和日志由 `run.sh` 输出；详细步骤见 [REPRO_PORTABLE.md](REPRO_PORTABLE.md)。首次启动需要编译和 Graph 捕获。

## 5. Smoke 测试

```bash
bash smoke.sh
```

2026-11-03 当前实例使用干净拉取的分支代码和新工作目录，服务成功启动；健康检查、单图问答、多轮、流式、定位、可操作点、导航、轨迹和多图测试均有输出。示例图文问题的回答描述了木桌上的白色杯子；流式结果与非流式结果一致。异常输入检查会在服务日志产生预期的错误记录。

本次原始日志在镜像 `/opt/robo-image-v1/ref_out/validation_20261103/branch_smoke.log`；历史记录见 [smoke.log](smoke.log)。

## 6. 精度测评

公开精度测试脚本 `eval.sh` 按指定顺序运行**完整数据集**，默认运行 CV-Bench（2638 题）和 RefSpatial（Location + Placement，共 200 题）：

```bash
bash eval.sh cvbench refspatial   # 两个数据集依次运行
# 也可只运行其中一个：bash eval.sh cvbench 或 bash eval.sh refspatial
```

脚本在镜像内查找数据及上游判分代码，默认以代码目录为工作目录，原始预测、判分结果和日志写入代码目录的 `outputs/public_eval_*`；如需另选工作目录，可先设置 `WORK_DIR`。数据集按顺序执行，每个数据集内部默认并发 16；应先启动服务、停止压测。该脚本是仓库提供的公开基准运行入口，不是赛事方指定评测脚本。完整两项运行在新实例上尚未验证；TraceSpatial 等其他基准暂不包含在此入口。

`verify.py` 保留为仓库原有精度检查入口，当前没有可用的赛事官方精度基线，不能报告官方一致率或判定官方精度通过。以下是公开基准实测，用于说明任务质量，评测口径和分母见 [公开基准报告](reports/16_public_bench_results.md)：

| 公开基准 | 结果 | 口径说明 |
| --- | ---: | --- |
| CV-Bench | 88.12，2638 题 | FlagEvalMM 判分，`canonical_png` |
| RefSpatial | 63.00，200 题 | Location / Placement 默认范围均值 |
| CrossPoint-Bench | 75.20，1000 题 | 上游判分器，`relative_1000` |
| EmbSpatial-Bench | 77.77，3640 题 | 自建多选答案抽取 |
| RoboSpatial-Home | 62.0，350 题 | 上游判分 |
| Q-Spatial-Bench | 62.7%，101 题 | 3 次均值；单次 60.4%–65.4% |

原始预测和判分文件在镜像 `ref_out/accuracy/`，三套公开数据集在 `eval_data/benchmarks/`；来源索引为 `ref_out/README_traceability.md`。TraceSpatial 不同表的分母和口径尚需逐项核对，不在此合并为单一成绩。

## 7. 性能参考

使用仓库 `bench.sh`：每请求一张 1080p 图，输入文本长度 4K / 16K、输出预算 1K，并发 1 / 8 / 64；每档两轮，请求数为 `max(8, 2×并发)`。`scripts/bench_serving_vllm_image.py` 是针对镜像内 SGLang 客户端的 vLLM 图文兼容包装，**不是已确认的赛事官方评测脚本**。

```bash
BENCH_MODE=image bash bench.sh
```

同卡 eager 基线与 Graph + 选优库 + aiter FA 候选的输出吞吐如下；括号为基线 / 候选轮数。数值为选定轮次的均值或单轮值，**不是统一取第二轮**：

| 输入 | 并发 | 基线 tok/s | 候选 tok/s | 轮数 |
| --- | ---: | ---: | ---: | ---: |
| 4096 | 1 | 21.77 | 28.75 | 2 / 2 |
| 4096 | 8 | 101.65 | 113.68 | 2 / 1 |
| 4096 | 64 | 133.86 | 140.70 | 2 / 1 |
| 16384 | 1 | 16.50 | 19.43 | 2 / 2 |
| 16384 | 8 | 44.16 | 45.28 | 2 / 1 |
| 16384 | 64 | 9.35 | 9.72 | 1 / 1 |

以下为所选候选轮次的客户端 13 项指标（单位：时间 s / ms、吞吐 requests/s 或 tokens/s）；并非六档统一的第二轮：

| 输入 | 输出 | 并发 | 完成 | 时长 s | 请求/s | 输出 tok/s | 总 tok/s | TTFT 均值 ms | TTFT P99 ms | TPOT 均值 ms | TPOT P99 ms | 端到端均值 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 1024 | 1 | 8 | 285.29 | 0.03 | 28.71 | 206.98 | 1411.01 | 1427.87 | 33.48 | 33.67 | 35660.08 |
| 4096 | 1024 | 8 | 16 | 144.12 | 0.11 | 113.68 | 820.13 | 10346.68 | 10402.20 | 60.30 | 60.37 | 72028.57 |
| 4096 | 1024 | 64 | 128 | 931.55 | 0.14 | 140.70 | 1014.65 | 114687.63 | 490857.75 | 304.85 | 661.73 | 426545.41 |
| 16384 | 1024 | 1 | 8 | 429.72 | 0.02 | 19.06 | 378.74 | 6738.19 | 6878.12 | 45.92 | 50.01 | 53709.66 |
| 16384 | 1024 | 8 | 16 | 361.86 | 0.04 | 45.28 | 898.14 | 26063.13 | 54884.90 | 150.95 | 166.22 | 180484.79 |
| 16384 | 1024 | 64 | 128 | 13480.76 | 0.01 | 9.72 | 192.86 | 4062229.71 | 6026783.45 | 1395.95 | 1826.13 | 5490284.66 |

六档输出吞吐已从对应原始 JSONL 重算；输入/输出 token、TTFT、TPOT、端到端延迟、完整命令与各轮数据见 [调优对比](reports/tuning_comparison.md)、[基线原始文件](bench/same_card_eager/) 和 [候选原始文件](bench/tuned/)。`recovered_` 文件是会话转录恢复，不能当作客户端原始文件；16K/C64 的 9.72 对应 `fullbench_candidate_20261101_131934__round1_16384_1024_c64.jsonl`。

分支客户端已用六档、每轮一个请求和 32 token 输出预算完成链路检查；这不是正式并发性能测试，也不替代上表原始数据。正式压测结果写入工作目录新的 `outputs/bench_*`。

## 8. 注意事项

- C64 受单卡 KV cache 容量限制，请求可能排队，名义并发不等于实际同时执行的请求数；尾延迟与显存见原始日志。
- 新实例需提供模型挂载和可写工作目录；首次启动需编译。镜像尚未在保存后的新实例上验证。
- 压测和 smoke 的成功不等于完整精度验收。公开评测的依赖在新实例中的完整可用性仍需验证。
- 不修改模型权重、tokenizer 或模型配置；空间答案解析和图像预处理的兼容处理见第 3 节。
- 原始性能与公开评测结果保留在 `bench/` 和镜像 `ref_out/`；镜像目录绝对路径及赛事方正式评测脚本仍须与赛事方确认。
