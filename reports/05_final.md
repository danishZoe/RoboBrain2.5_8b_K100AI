# 最终报告：RoboBrain2.5-8B-NV 在 K100AI（天数智芯 K500SM_AI）上的适配

> 提交件：**RoboBrain2.5-8B-NV on K100-AI + vLLM 0.15.1**（精度 bf16，单卡 TP=1）
> 镜像：`jupyter-sglang5.12-ubuntu22.04-dtk26.04-py3.10-qwen3.6:v1.0`
> 提交包：`submission/`（`README.md` 为官方模板的正式报告，见 `submission/README.md`）
> 本文件是给评审/复跑者看的**总览与索引**。

---

## 1. 是否已在 K100AI 跑通

**是。** 已在本节点的 K100AI 卡（运行时实际名称 `K500SM_AI`，架构 `gfx928`，63.98 GiB）
上完成 **vLLM serving** 的端到端适配与验证。

| 判据 | 结果 | 证据 |
|---|---|---|
| 设备参与推理 | ✅ 不是 CPU 回退：`hy-smi` 显示 VRAM 由 0% 升到 ~30%，`/metrics` 的 `vllm:generation_tokens_total` 持续增长 | `submission/server.log`、`outputs/phase5_smoke/` |
| 模型与全部组件加载 | ✅ 权重 750 tensor / 2 分片完整；processor、分词器、视觉塔（depth 27）全部加载 | `submission/env.log` |
| 端到端有效输出 | ✅ 四类任务 + 多图 + 流式 + 多轮全部返回与任务相关的内容 | `submission/smoke.log`、`outputs/phase5_smoke/smoke_results.json` |
| 重复运行稳定 | ✅ 3 进程 × 4 次贪心 = 12/12 输出逐字一致；官方六档压测两轮零失败 | `reports/10_phase5_bench.md` |
| 服务可复现启动 | ✅ `bash run.sh` 幂等；实测 server up after 195 s | `submission/server.log` |

**框架选择说明**：官方 T02 接受「SGLang **或** vLLM」并分别计件。本节点 **SGLang 不可用**
（镜像内 `aiter` 残缺构建：13 个符号未定义，5 个后端 + 2 种 model impl 全部崩溃，详见
`reports/06_sglang_compatibility.md`），因此**只计 vLLM 一件**，未用 SGLang 名义冒领。

---

## 2. 验证的任务范围

### 2.1 已验证（服务级，`scripts/smoke_service.py` 12/12 通过）

| 编号 | 检查项 | 结果 |
|---|---|---|
| F01 | `GET /health` / `GET /v1/models` | 200 / 200，返回 `robobrain2.5-8b-nv` |
| F02 | general 图文描述 | 200，有效描述 |
| F03 | grounding（bbox） | 200，`[0, 309, 620, 989]` |
| F04 | affordance / pointing | 200，`(904, 600)` / `(582, 725)` |
| F05 | navigation | 200，`(401, 376)` |
| F06 | trajectory（3D waypoints 含深度） | 200，3 点 |
| F07 | 多图（2 图）general | 200，两图均参与 |
| F08a/b/c | 异常输入（损坏图 / 不可达 URL / 畸形 data URL） | 400 / 200 空内容 / 400，**未静默接受** |
| F09 | 流式 vs 非流式 | 拼接结果**逐字一致** |
| F10 | 多轮图文 | 200，正常作答 |

### 2.2 未验证 / 无法验证（如实标注）

| 项 | 状态 | 原因 |
|---|---|---|
| **精度项（官方 ≥98% 一致率）** | **BLOCKED** | 官方 GPU 参考输出、官方测试集、官方判分器**均未下发**，节点无外网。按验收标准第 2 节不得用自建 demo 顶替 |
| 视频 / 多视角 GRM 输入 | 未纳入 | 官方分题材料未下发（验收标准 §10.1 未纳入） |
| 深度图输入 | 未纳入 | 同上 |
| SGLang 计件 | 不可用 | 镜像内 `aiter` 残缺构建 |
| 官方 `bench.sh` / `env.sh` / `verify.py` 脚手架 | 未正式下发 | 本节点按 PDF 全文**等价实现**，已在 `reports/09` 记录两处偏离 |
| 多机 / TP=4 口径 | 未验证 | 本 Pod 仅分配 1 张卡 |

---

## 3. 一条可复现推理命令

```bash
# 0) 进入提交包并加载环境变量（含 DTK 运行时、隔离依赖、HF 缓存重定向）
cd /root/private_data/workspace/robo/submission
source ./env.sh

# 1) 起服务（装依赖 → 只读判重 patch → sitecustomize 垫片 → 拉起 vLLM → 轮询 /health）
bash run.sh
#    实测：server up after 195 s

# 2) 单请求图文推理（一条命令即可复现）
python3 scripts/make_data_url.py samples/images/official_affordance.jpg > /tmp/img.url
curl -sf http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"robobrain2.5-8b-nv\",\"temperature\":0,\"max_tokens\":768,
       \"messages\":[{\"role\":\"user\",\"content\":[
         {\"type\":\"image_url\",\"image_url\":{\"url\":\"$(cat /tmp/img.url)\"}},
         {\"type\":\"text\",\"text\":\"Please provide the bounding box coordinate of the region this sentence describes: the mug.\"}]}]}" \
  | python3 -m json.tool
```

四类任务的官方提示词模板与解析正则见 `submission/scripts/robo_infer.py` 与
`third_party/RoboBrain2.5/inference.py:110-124`（逐字复制，未改写）。

> **图像必须先规范化**：`scripts/image_canonical.py` 用上游 `qwen_vl_utils` 的取图路径
> 预缩放后再以 PNG 上传，使服务端的像素与官方 upstream **逐位一致**
> （`max|diff| = 0.000000`；5/5 样例图全部一致）。不规范化则像素最大差 0.47，
> 会翻转 grounding 的首个生成 token，官方正则直接判为不可解析。
> 根因与机制见 `reports/09_vllm_adaptation.md` §3。

---

## 4. 优化前后同口径指标

### 4.1 提交口径：官方六档负载压测（vLLM serving，image 数据集）

口径：单卡 K100AI / vLLM 0.15.1 / BF16 / `--enforce-eager` / `--max-model-len 32768` /
`--gpu-memory-utilization 0.85` / `--max-num-batched-tokens 4096` / `--max-num-seqs 64` /
`--no-enable-prefix-caching`；客户端为官方指定的 `sglang.bench_serving`，
每请求 1 张 1080p 图，`--seed 1`，请求数 `max(8, 2×并发)`，输出实际打满 1024 token。

<!-- FINAL_BENCH_TABLE -->
| random_input_len | random_output_len | max_concurrency | completed | duration | request_throughput | output_throughput | total_throughput | mean_ttft_ms | p99_ttft_ms | mean_tpot_ms | p99_tpot_ms | mean_e2e_latency_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | 1024 | 1 | 8 | 325.78 | 0.02 | 25.15 | 181.28 | 802.97 | 834.86 | 39.02 | 39.32 | 40720.70 |
| 4096 | 1024 | 8 | 16 | 113.06 | 0.14 | 144.91 | 1046.99 | 5748.54 | 5810.17 | 49.61 | 50.30 | 56497.34 |
| 4096 | 1024 | 64 | 128 | 651.67 | 0.20 | 201.13 | 1450.34 | 71638.63 | 322252.82 | 219.93 | 462.72 | 296628.01 |
| 16384 | 1024 | 1 | 8 | 375.90 | 0.02 | 21.79 | 432.26 | 3648.41 | 3701.71 | 42.36 | 42.76 | 46986.40 |
| 16384 | 1024 | 8 | 16 | 256.60 | 0.06 | 63.85 | 1267.53 | 13901.87 | 29278.75 | 111.49 | 119.08 | 127953.45 |
| 16384 | 1024 | 64 | 128 | 7694.30 | 0.02 | 17.03 | 337.93 | 2309809.63 | 3478357.63 | 797.18 | 1000.47 | 3125328.93 |

- 官方 13 列格式，原样取自 `submission/bench/summary.md`（记录轮 = 第 2 轮）。
- 两轮共完成 **608** 个请求，失败 **0** 个。
- 轮间最大偏差 **2.3%**（输出吞吐），重复性良好。
- **单卡口径**：本 Pod 只有 1 张卡（官方脚手架默认 TP=4）。
  并发 64 档受 KV cache（253,312 token）限制，实测服务侧真实常驻并发为 **50**（4096 档）/ **15**（16384 档），其余排队；该档 TTFT 反映排队，不是模型变慢。
- 输出 token 口径已用服务端引擎计数器独立核实（引擎生成量 = 计划量的 95.6%–99.7%，**不存在提前 EOS**），详见 `reports/10_phase5_bench.md` §3.7 / §3.8。
<!-- /FINAL_BENCH_TABLE -->

完整表格（含输入 token 的文本/视觉拆分、失败数、端到端时延）见
`reports/10_phase5_bench.md` §3.3 与 `submission/bench/summary_detailed.md`。

### 4.2 服务级优化实验（单变量）

<!-- FINAL_TUNING_TABLE -->
| 实验 | 变量 | 结果（实测） | 采纳 |
|---|---|---|---|
| E1 | 去掉 `--enforce-eager`（启用 torch.compile + CUDA graph） | 可启动；4096/1 档 32.81 tok/s（vs 官方基线 +30.5%）；4096/8 档 145.15 tok/s（vs 官方基线 +0.2%）；与 transformers 基线逐坐标完全一致 2/6 = 33.3% | **不采纳**（收益只落在并发 1 档，官方口径核心的并发 8/64 档无收益；且输出数值发生变化，官方 ≥98% 精度门槛因缺官方基线无法验证该变化无害） |
| E1-确认 | 非 eager 重跑（登记要求「重跑确认」） | 4096/1 档 32.84 tok/s（vs 官方基线 +30.6%）；4096/8 档 148.77 tok/s（vs 官方基线 +2.7%）；16384/1 档 25.86 tok/s（vs 官方基线 +18.7%）；16384/8 档 68.88 tok/s（vs 官方基线 +7.9%）；与服务级 12 项 Smoke 同批完成 | **不采纳**（确认实验与首测结论一致） |
| E2 | `--max-num-batched-tokens` 4096 / 8192 / 16384 | 8192 档 16384/1：吞吐 -0.2%、TTFT -1.9%；8192 档 16384/8：吞吐 +0.5%、TTFT +30.5%；16384 档 16384/1：吞吐 +0.2%、TTFT -4.1%；16384 档 16384/8：吞吐 +3.4%、TTFT +74.7% | **不采纳**（无取值同时满足「吞吐 >3% + TTFT 下降 + 无失败」，回退 4096） |
| E3 | 厂商 GEMM 算子调优 | 本节点无 `auto_select_tools` / shape 数据 / rocprof，**不具备执行条件**；trace 亦未产出 | 不适用 |
| E4 | 输出 token 计量口径核验 | 引擎生成量 = 计划量 95.6%–99.7%，**无提前 EOS**；官方吞吐列可直接填报 | **采纳** |

判定标准均在实验前登记（`reports/11_tuning_experiments.md` §2.2 / §3.3 / §5.2），完整结果与原始数据见该报告各节与 `outputs/phase5_ab_*`。
<!-- /FINAL_TUNING_TABLE -->

### 4.3 诊断口径：transformers 直跑（**非提交口径**，仅历史对照）

口径：6 样例 × 3 次、剔除首次 warm-up（稳态 n=12）/ `max_pixels=501760` / 贪心

| 指标 | 值 |
|---|---|
| 模型加载 | 62.12 s |
| TTFT | 均值 **0.4918 s** |
| 端到端时延 | 均值 **2.9681 s** |
| 解码吞吐 | 均值 **11.8357 tok/s** |
| 峰值显存 | 均值 **16.5497 GiB** |
| 批处理 bs=4 | 聚合 **31.236 tok/s（×2.72）**，峰值 17.068 GiB |

### 4.4 关键对比：服务化 vs 直跑（同卡、同样例、同贪心）

| 指标 | transformers 直跑 | vLLM serving | 变化 |
|---|---|---|---|
| 单请求解码吞吐 | 11.84 tok/s | **21–25 tok/s** | **≈2×** |
| grounding 单请求端到端 | 1.90 s | **0.9148 s** | **−52%** |
| 并发 8 吞吐 | — | **144.91–148.86 tok/s** | 并发扩展 **5.8×** |

<!-- CLEAN_REPRO:BEGIN -->

### 4.5 干净环境复现（验收标准 §7.2）

尚未执行（`logs/clean_repro/` 无复跑数据）。

<!-- CLEAN_REPRO:END -->

<!-- OPTGATE:BEGIN -->

### 4.6 验收标准 §8.2 优化门槛（≥10%）核算

§8.2 有两条硬要求：**同卡型、同框架、同负载下提升 ≥10%**；**优化后必须重跑官方精度门槛，精度未通过的性能结果不算优化验收成功**。
下表按 §8.2 的表格模板逐格填写，数字取自原始文件。

| 档位/指标 | 首次提交基准 | 优化结果 | 公式与变化 | 精度结果 | 原始证据 |
|---|---|---|---|---|---|
| 4096/1 输出吞吐 tok/s | 25.15 | 32.81 | `(32.81 / 25.15 - 1) × 100%` = **+30.5%**（≥10%） | 官方精度门槛 **BLOCKED**（缺官方 GPU 基线/判分器）；内部一致性：与 transformers 基线逐坐标 2/6 = 33.3% | `outputs/phase5_ab_noneager/noneager_20261027_194104/` + `submission/bench/round2.jsonl` |
| 4096/8 输出吞吐 tok/s | 144.91 | 145.15 | `(145.15 / 144.91 - 1) × 100%` = **+0.2%**（<10%） | 官方精度门槛 **BLOCKED**（缺官方 GPU 基线/判分器）；内部一致性：与 transformers 基线逐坐标 2/6 = 33.3% | `outputs/phase5_ab_noneager/noneager_20261027_194104/` + `submission/bench/round2.jsonl` |

**结论（逐条对照 §8.2）**：

1. **≥10% 门槛：只在并发 1 档成立，且不构成整件级提升。**
   最高提升出现在 4096/1 档（**+30.5%**），但官方口径的核心档（并发 8 / 64）没有同量级收益；§8.2 明确「不因单档提升就宣称整件已获官方认可」，因此本报告**不宣称整件达到 ≥10%**。
2. **精度复验：无法完成 → §8.2 的第二个条件不成立。**
   §8.2 要求「优化后重跑服务功能及官方精度门槛，精度未通过的性能结果不算优化验收成功」。本任务的官方精度门槛因缺官方 GPU 参考输出与判分器而 **BLOCKED**（见 `reports/08_gap_vs_official_baseline.md` 风险 #1），无法证明非 eager 配置下输出数值变化（内部逐坐标一致率 2/6 = 33.3%）是无害的。
3. **因此采纳决策：提交配置保持首提基线（`--enforce-eager`）。**
   零风险、可复现、与提交表完全一致；非 eager 的可启动性与收益/代价在 `reports/11_tuning_experiments.md` §2 如实登记，不隐藏、也不冒充已采纳的优化。

> 一句话：**本次交付没有可宣称达标的性能优化**。同口径基准已完整建立、实验方法与判定标准已实验前登记、单变量 A/B 已执行并留证；但 §8.2 的两个条件（≥10% 且精度复验通过）在本节点上只满足前者的一部分，而后者因缺官方材料无法满足 —— 如实标注，不凑数。

<!-- /OPTGATE -->

---

## 5. 兼容性改动清单（相对官方仓库）

| # | 改动 | 为什么 | 是否影响精度 |
|---|---|---|---|
| 1 | `patches/0001-vllm-qwen3-tie-word-embeddings.patch`（等价实现：`envs/vllm_shim/sitecustomize.py`） | vLLM 0.15.1 从**子 config** 读 `tie_word_embeddings`，transformers 5.3.0 只在**顶层复合 config** 保留 → `AttributeError` 致命失败 | **否**。取值由模型 `config.json`（`false`）+ 权重索引含 `lm_head.weight` 双向推导，不写死 |
| 2 | `requirements-extra.txt`：`qwen-vl-utils==0.0.14` + 6 个连带依赖，`--no-deps --target $WORK_DIR/envs/pylibs` | 官方 `inference.py` 的 `process_vision_info()` 依赖它；镜像未自带 | 否（只影响取图路径，使其与官方一致） |
| 3 | `scripts/image_canonical.py` | 服务端只做一次缩放，官方 upstream 做两次；像素不同会翻转输出 | **是**（正是为了对齐官方才引入） |
| 4 | `scripts/answer_adapter.py` | vLLM 输出 `<bbox>x y x y</bbox>`，官方正则只认 `[x1,y1,x2,y2]`；纯格式桥，四个整数原样搬运 | 否（不改数值） |
| 5 | `scripts/bench_serving_vllm_image.py` | 官方压测客户端的 image 数据集对后端做硬白名单，`--backend vllm` 直接抛错 | 否（进程内映射，不改公共 site-packages） |
| 6 | `run.sh` 不写公共 site-packages（只读判重） | 任务红线：公共 Python 环境只读 | 否 |
| 7 | `env.sh` `MAX_LEN=32768` | 官方 `env.sh` 口径；20480 装不下「16.3K 文本 + 2,042 视觉 + 1,024 输出」 | 否（未改模型声明的 262,144） |

**未改动**：模型权重、`config.json`、`generation_config.json`、分词器、chat template、采样语义。

### 5.1 交付代码 revision

<!-- REVISION:BEGIN -->

| 项 | 值 |
|---|---|
| 交付代码 revision（git commit） | `0658ed25dcf32eaa66422e1e5fbdce16682fcd11` |
| 短 hash | `0658ed2` |
| 记录时间 | 2026-10-27T21:44:22+08:00 |
| 纳入仓库的文件数 | 304 |
| 提交包逐文件校验值 | `submission/MANIFEST.sha256` |
| 仓库内容范围 | `submission scripts tools configs samples patches reports platform` |

> 该 revision 由 `tools/record_revision.sh` 在交付包定稿后生成，仓库只含交付物（隔离依赖 `envs/pylibs/` 与运行产物 `outputs/`、`logs/` 不入仓库，可由提交包脚本重新生成）。PDF 要求代码与镜像分别提交到指定托管平台，**托管链接仍需用户上传后回填**；节点无外网，无法代替上传。


---

## 6. 目录与产物索引

| 路径 | 内容 |
|---|---|
| `submission/` | **提交包**：`README.md`（官方模板八章）、`env.sh`、`env_check.sh`、`run.sh`、`smoke.sh`、`verify.py`、`bench.sh`、`requirements-extra.txt`、`patches/`、`envs/vllm_shim/`、`scripts/`、`samples/`、`bench/`、`env.log`、`server.log`、`smoke.log`、`verify_result.json` |
| `reports/00_RUNBOOK.md` | 运行手册与故障恢复指南（含 15 条实测故障排查） |
| `reports/01`–`07` | 环境、适配、基线、调优、最终、**SGLang 不可用定案**、设备型号映射 |
| `reports/08_gap_vs_official_baseline.md` | 与官方基线的逐项差距评估（含 10 条口径风险） |
| `reports/09_vllm_adaptation.md` | vLLM 适配全过程：4 个阻塞 + grounding 根因闭环 |
| `reports/10_phase5_bench.md` | 阶段 5 冒烟与六档压测（含两处口径修正与交叉验证） |
| `reports/11_tuning_experiments.md` | 服务级单变量调优实验 E1–E3 |
| `outputs/` | 原始测量数据（baseline / phase4 / phase5_* / 批量 / A/B） |
| `scripts/` | 环境检查、最小推理、探针、压测、图像规范化、格式桥、A/B 脚本 |
| `tools/` | 汇总、提交包自检、产物刷新、流水线等辅助工具 |
| `patches/`、`envs/` | 等价 diff 与隔离依赖/垫片 |

---

## 7. 尚存限制

1. **精度项 `BLOCKED`** —— 唯一能关闭它的材料（官方 GPU 基线文件、官方测试集、官方判分器）
   只能由用户向平台索取；节点无外网。`verify.py` 已改成**真跑评测**，
   缺哪一项就标 `BLOCKED`，绝不会在没有证据时报 PASS。
2. **单卡口径** —— 官方脚手架默认 TP=4，本 Pod 只有 1 张卡。
   性能对比必须显式标注卡数；并发 64 档受 KV cache（253,312 token）限制，
   实测服务侧真实常驻并发 **50**（4096 档）/ **15**（16384 档，p50=15 / p90=16），
   不是请求的 64，其余排队。详见 `reports/10_phase5_bench.md` §3.6。
3. **`--enforce-eager` 的代价** —— 见 `reports/11_tuning_experiments.md` §2；
   非 eager 在 115 GiB cgroup 上限下曾触发 OOM（Pod 重建、节点不可达 5 h），
   A/B 带内存看门狗执行，结论以实测为准。
4. **厂商 GEMM 算子调优在本节点不具备执行条件** —— 无 `auto_select_tools`、无随包 shape 数据、
   无 `rocprof`；只采到 rocBLAS layer3 trace 作诊断（`reports/11` §4）。
5. **`platform/` 截图与平台区域/代理地址待用户补** —— 节点访问不到平台控制台。
6. **官方脚手架未下发** —— `bench.sh` / `env.sh` / `env_check.sh` / `run.sh` / `verify.py`
   均为按 PDF 全文的等价实现；官方脚手架到位后需在其上重跑一次并比对偏差（PDF 规定 >20% 打回）。
7. **图像分辨率口径** —— 官方 `bench.sh` 的 `IMAGE_RES` 注明「以分题材料为准」，
   当前用 1080p；拿到官方题集后需重新确认。
8. **代码/镜像提交链接待回填** —— PDF 要求代码与镜像分别提交到指定托管平台并记录
   可访问链接与 revision。节点无外网、也无法代替用户上传，
   `submission/README.md` §1 的「提交仓库」与 §2 的「区域 / 服务代理映射地址」仍是占位，
   上传后需回填这两处（`platform/README.md` 说明了需要哪几张截图）。
