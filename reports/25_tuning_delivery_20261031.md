# 25 · 最优版交付报告（2026-10-31）

> 口径：单卡 K100AI（运行时 `K500SM_AI` / gfx928 / 63.98 GiB）/
> vLLM `0.15.1+das.opt1.alpha.dtk2604` / BF16 / TP=1。
> 压测：官方 `bench_serving` 客户端（image 数据集，1×1080p 图/请求，
> `--random-output-len 1024`、`--seed 1`、请求数 `max(8, 2×并发)`）。
>
> 本报告是「提交配置从首提 eager 基线切到最优版」这一轮交付的**唯一事实来源**：
> 配置、收益、证据、回退方法、未覆盖项都在这里。

---

## 1. 结论

| 项 | 结论 |
|---|---|
| 提交配置 | **最优版** = graph(非 eager) + GEMM 算子调优库 + `ROCM_AITER_FA` |
| 相对 graph 基线 | 4096/并发1 **+28.36%**、4096/并发8 **+10.32%**（2026-10-31 同节点同轮实测，见 §5.2） |
| 相对首提 eager 基线 | 4096/并发1 **+32.11%**、4096/并发8 **+13.24%**、4096/并发64 **+1.92%**（2026-10-31 同节点同轮实测，赛事口径正式对比，见 §5.2） |
| 精度 | 官方 ≥98% 一致率仍 **BLOCKED**（缺官方 GPU 参考输出与判分器），**不以性能结果代替精度结论** |
| 代码托管 | `ssh://git@developer.sourcefind.cn:2222/g_exhyd6h6/robobrain2.5_8b_k100ai.git`，`main` = 最优版，标签 `baseline-submission` = 首提基线 |

---

## 2. 最终配置

```bash
# env.sh（提交包内，唯一事实来源）
export USE_TUNED=1
export TUNED_ROCBLAS_DIR="$WORK_DIR/rocblas_tuned/library_gpu4"
export ROCBLAS_TENSILE_LIBPATH="$TUNED_ROCBLAS_DIR"
export VLLM_ATTENTION_BACKEND=ROCM_AITER_FA
export EXTRA_ARGS="--no-enable-prefix-caching --max-num-batched-tokens 4096 --max-num-seqs 64 --limit-mm-per-prompt={\"image\":8}"
# 其余：--tensor-parallel-size 1 --pipeline-parallel-size 1 --dtype bfloat16
#       --trust-remote-code --gpu-memory-utilization 0.85 --max-model-len 32768
```

相对首提基线只有三处差异：

| # | 差异 | 依据 |
|---|---|---|
| 1 | 去掉 `--enforce-eager` | graph（torch.compile + graph capture）：同节点 4096/并发1 首提 eager 21.75 → graph 基线 22.38 tok/s（+2.9%），单项收益有限；叠加调优库与 aiter FA 后到 28.73 tok/s，主体收益来自第 2、3 项 |
| 2 | `ROCBLAS_TENSILE_LIBPATH` → 工作目录内的调优库 | decode 形状 N=1~64 的 GEMM 换成 GEMV 专用 kernel（算子级 107 size 中位 +19.1%；10-30 单独隔离 E5：4096/并发1 +23.8%）；同节点 graph 基线 → 最优版 4096/并发1 +28.36%、4096/并发8 +10.32% |
| 3 | `VLLM_ATTENTION_BACKEND=ROCM_AITER_FA` | 16384 长上下文 decode 走 aiter FA kernel（10-30 同轮 E7 gate：16384/并发1 +17.9%，TPOT 34.66 → 28.85 ms）；本节点未单独隔离此项，未计入上表 |

**未改动**：模型权重、`config.json`、`generation_config.json`、分词器、chat template、
采样语义、图像预处理约定、`--max-num-batched-tokens 4096`、`--max-num-seqs 64`、
`--gpu-memory-utilization 0.85`、`--max-model-len 32768`。

---

## 3. 调优库的组装与生效证据

### 3.1 与 stock 的确切差异

候选库目录与 `/opt/dtk/lib/rocblas/library_gpu4` 逐文件比对：**4 个文件被改写、
2 个新增，其余 159 个逐字节相同**。这 6 个文件随仓库提交在
`submission/rocblas_tuned/library_gpu4/`：

| 文件 | SHA256 | 状态 | 大小 |
| --- | --- | --- | ---: |
| `GemmGenerality_AI_Eco.dat` | `ad891f317f5c097776e014d738f17e84d2e4af7dfc7ad352f01f0d60d28025f3` | 改写 | 5,804,448 |
| `GemmGeneralityAdd_Eco.dat` | `23e30c02a186d0c9a0cc851c2874752f1fc3f7fdd89ba77bf2cd5c5806324f58` | 改写 | 6,852,736 |
| `GemmGenerality_SolutionName_AI_Eco.txt` | `5509bf83b23f961c17b9a60021a2be070f431835b8920f34cedeac079697b1c3` | 改写 | 401,342 |
| `GemmGeneralityAdd_SolutionName_Eco.txt` | `8b5b348f95a93ab7f852a816389cc2e34f48806a96dd02b5a4842c035082ccd9` | 改写 | 839,342 |
| `solutionIndex_AI_Eco.txt` | `6ec0a9d84da70e5a2eeb0fa1dffe06c9f70feaf8defdc1d8203d9c09aa2d736c` | 新增 | 5,680,474 |
| `solutionIndexAdd_Eco.txt` | `3639767ab465c78e729e56850ce0ccbe2229d84f1dfaa7bb059bc8c62b43da79` | 新增 | 6,447,653 |

stock 对照：`/opt/dtk/lib/rocblas/library_gpu4/GemmGenerality_AI_Eco.dat`
= `be0f15afdd77d40cd3f1cdcf400b84842d1b358342e9c59e4b489a22fc4248cf`（**全程未被修改**）。

### 3.2 组装（可复现、幂等、失败即退出）

`submission/scripts/make_tuned_rocblas_lib.sh`：

1. 把 stock 库**复制**到 `$WORK_DIR/rocblas_tuned/library_gpu4`（公共 `/opt/dtk` 只读不动）；
2. 用仓库内 6 个差异文件覆盖；
3. 逐个校验 SHA256，不符即 `exit 1`——**不允许带病启动**；
4. 目标目录已存在且哈希一致时跳过复制（幂等）。

`run.sh` 在启动服务前调用它；组装失败则拒绝启动。

实测：组装结果与 10-30 验证过的候选目录 **163 个文件逐字节一致**（`diff -r` 无差异）。

### 3.3 生效证据

- **kernel 切换**：`ROCBLAS_DEBUG_GENERALITY=1` 实测 (4096,1,4096) 从 kernel 288
  切到 712/GEMVMT256；(4096,1,12288) 与 (4096,64,12288) 均切到新 kernel。
- **文件被实际打开**：隔离启动的 inotify 记录到候选目录被打开
  `GemmGenerality_AI_Eco.dat`、`GemmGeneralityAdd_Eco.dat` 及对应 SolutionName TXT，
  `queue_overflow=false`（原始记录 `outputs/prompt_retune/.../library_open_verification/open_events.json`）。
- **服务级收益**：同口径 A/B（见 §4）。

---

## 4. 完整压测（最优版）

> 产物目录：`outputs/fullbench_candidate_20261031_231152/`
> 驱动脚本：`scripts/run_full_bench.sh`；汇总：`scripts/summarize_full_bench.py`；
> 对比表：`reports/tuning_comparison.md`（脚本 `scripts/build_tuning_comparison.py`）。

<!-- FULL_BENCH_TABLE:BEGIN -->
**最优版（提交配置）六档实测结果**（官方 13 列格式，取记录轮）：

| random_input_len | random_output_len | max_concurrency | completed | duration | request_throughput | output_throughput | total_throughput | mean_ttft_ms | p99_ttft_ms | mean_tpot_ms | p99_tpot_ms | mean_e2e_latency_ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 1024 | 1 | 8 | 285.07 | 0.03 | 28.74 | 207.23 | 1448.69 | 1630.32 | 33.42 | 33.62 | 35632.50 |
| 4096 | 1024 | 8 | 16 | 141.29 | 0.11 | 115.96 | 836.86 | 9786.72 | 10444.80 | 59.46 | 66.23 | 70614.40 |
| 4096 | 1024 | 64 | 128 | 941.54 | 0.14 | 139.21 | — | 114388.36 | — | 309.35 | — | 430852.94 |
| 16384 | 1024 | 1 | 8 | 426.14 | 0.02 | 19.22 | 381.56 | 6717.82 | 6826.51 | 45.50 | 48.37 | 53266.30 |
| 16384 | 1024 | 8 | 16 | 363.36 | 0.04 | 45.09 | — | 25831.20 | — | 151.86 | — | 181184.52 |
| 16384 | 1024 | 64 | 128 | 13667.91 | 0.01 | 9.59 | — | 4134411.49 | 6101802.49 | 1416.31 | — | 5583300.15 |

> **缺口说明**：`—` 表示该字段在本次数据来源中未被留存，**不是 0**。以下档位存在未留存字段：

> - 4096/并发64：total_throughput, p99_ttft_ms, p99_tpot_ms
> - 16384/并发8：total_throughput, p99_ttft_ms, p99_tpot_ms
> - 16384/并发64：total_throughput, p99_tpot_ms
>
> 原因与影响见 §9；吞吐、TTFT、TPOT、时长、完成数、失败数等核心指标齐全。
各档轮次与波动：

| 输入 | 并发 | 轮数 | 输出吞吐 tok/s（各轮） | 轮间极差 | 失败 |
| --- | ---: | ---: | --- | ---: | ---: |
| 4096 | 1 | 2 | 28.72, 28.74 | 0.07% | 0 |
| 4096 | 8 | 2 | 115.36, 115.96 | 0.52% | 0 |
| 4096 | 64 | 2 | 140.36, 139.21 | 0.83% | 0 |
| 16384 | 1 | 2 | 18.98, 19.22 | 1.30% | 0 |
| 16384 | 8 | 2 | 44.96, 45.09 | 0.28% | 0 |
| 16384 | 64 | 1 | 9.59 | 0.00% | 0 |

> 原始数据：`outputs/fullbench_candidate_20261031_231152/`（每档 `round*.jsonl` 与同名 `.log`）；
> 驱动脚本 `scripts/run_full_bench.sh`，汇总 `scripts/summarize_full_bench.py`。
> 同一配置在 10-30 节点上的绝对值更高（该节点整体快约 30%），
> 跨节点绝对值不可直接比较，见 §6 节点差异警告。
<!-- FULL_BENCH_TABLE:END -->

---

## 5. 与基线的对比

### 5.1 三条基线的定义

| 基线 | 定义 | 用途 |
|---|---|---|
| 首提基线（eager） | `--enforce-eager` + stock 库 + 默认 attention | 赛事「首次提交性能实测」，`baseline-submission` 标签 |
| graph 基线 | 非 eager + stock 库 | 本轮调优的**同轮对照**（只开 graph，不带任何调优开关） |
| 最优版 | 非 eager + 调优库 + aiter FA | 当前提交配置 |

### 5.2 同卡 A/B：graph 基线 → 最优版（唯一可直接相减的两列）

<!-- SAME_NODE_AB:BEGIN -->
| 输入 | 并发 | graph 基线 tok/s | 最优版 tok/s | 吞吐提升 | TTFT 变化 | TPOT 变化 | 基线轮/候选轮 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4096 | 1 | 22.38 | 28.73 | +28.36% | -4.2% | -22.7% | 1/2 |
| 4096 | 8 | 104.84 | 115.66 | +10.32% | -10.0% | -9.2% | 1/2 |
<!-- SAME_NODE_AB:END -->

### 5.3 同卡 A/B：首提基线 eager → 最优版（赛事口径的正式对比）

> 首提基线 = `--enforce-eager` + stock rocBLAS 库 + 默认 attention，
> 与赛事「首次提交性能实测」口径完全一致；最优版只改了三处配置（见 §2）。
> 两者**必须来自同一张卡**才能相减。

<!-- SAME_NODE_AB_EAGER:BEGIN -->
（待回填）
<!-- SAME_NODE_AB_EAGER:END -->

---

## 6. 换卡与节点差异警告（重要）

2026-10-31 复测发现：**同一配置在当前节点比 10-30 慢约 30%**。

2026-11-01 05:08 起容器又被平台回收重建，**物理卡发生更换**。于是绝对值分三类，
**只有同一张卡内部的两个数才允许相减**：

| 类别 | 含义 | 可否相减 |
|---|---|---|
| 同卡 A/B | 同一张卡上 eager / graph 基线 / 最优版 三套 | ✅ 可以（§5.2、§5.3） |
| 历史同节点 | 2026-10-31 节点上测的三套（当时未记录卡指纹） | ✅ 可以，但必须标注为历史会话 |
| 跨卡存档 | 不同卡上的实验（含 10-30、10-27） | ❌ 不可相减，只作过程存档 |

判定方式：每次压测由 `scripts/run_full_bench.sh` 把**卡指纹**写进
`<run-dir>/run_config.txt` 的 `card_id=`；报告脚本 `scripts/bench_set.py`
只把 `card_id` 与当前节点 `outputs/CARD_ID.txt` 一致的目录算作同卡，
**没有 `card_id` 的老目录一律只作存档**，绝不混进同卡 A/B。

历史节点对照（仅供参考，不可与当前卡相减）：

| 配置 | 节点 | 4096/并发1 输出吞吐 |
| --- | --- | ---: |
| graph 基线（stock 库） | 10-30 | 33.03 / 33.09 tok/s |
| graph 基线（stock 库） | 10-31 | **22.38 tok/s** |
| 首提 eager 基线 | 10-27 | 25.15 tok/s |
| 首提 eager 基线（10-31 复现） | 10-31 | **21.73 / 21.76 tok/s** |

排查结论：服务参数一致、graph 捕获列表一致（19 mixed + 11 FULL）、
`/proc/<worker>/environ` 无残留调优开关 → **不是配置差异，是节点差异**。
因此：

- 跨节点的绝对值**不可直接比较**；
- 收益结论一律用**同卡同轮**的 A/B；
- 官方「>20% 复现偏差打回」这条要按同卡口径看，见 §8。

---

## 7. 代码库改动清单

| # | 改动 | 影响精度 | 说明 |
|---|---|---|---|
| 1 | `submission/env.sh`：新增 `USE_TUNED` 开关、`ROCBLAS_TENSILE_LIBPATH`、`VLLM_ATTENTION_BACKEND`；`EXTRA_ARGS` 去掉 `--enforce-eager`（`USE_TUNED=0` 时自动加回） | 否 | 唯一改变运行行为的文件 |
| 2 | `submission/scripts/make_tuned_rocblas_lib.sh`（新） | 否 | 组装并校验调优库 |
| 3 | `submission/rocblas_tuned/library_gpu4/`（新，6 个差异文件） | 否 | 调优产物随仓库提交，保证可复现 |
| 4 | `submission/run.sh`：启动前组装调优库；新增 graph 启动期 70 GiB 内存看门狗 | 否 | 看门狗是 10-27 OOM 事故的防护 |
| 5 | `submission/README.md`：新增 §3.5 调优方案、§4 配置与回退、§7 性能与对比 | 否 | 官方八章报告同步更新 |
| 6 | `reports/25_tuning_delivery_20261031.md`（新）、`reports/tuning_comparison.md`（新） | 否 | 本报告与对比表 |
| 7 | `scripts/run_full_bench.sh`、`scripts/summarize_full_bench.py`、`scripts/build_tuning_comparison.py`（新） | 否 | 完整压测与对比表生成 |
| 8 | `tools/check_submission.sh`：env.sh 变量检查允许缩进的 `export` | 否 | 自检脚本修正 |
| 9 | `.gitignore`：排除 `/rocblas_tuned/`（运行时组装目录）与 `core.*`（3 GB 级转储） | 否 | 防止误提交 |
| 10 | `platform/README.md`：补代码托管链接 | 否 | 镜像链接仍待用户补 |
| 11 | `scripts/card_identity.sh`（新）、`scripts/bench_set.py`（新） | 否 | 卡指纹采集与「同卡 / 跨卡」判定；换卡后报告不会再把不同卡的绝对值相减 |
| 12 | `scripts/run_full_bench.sh`：每次运行把 `card_id` 写进 `run_config.txt`；`--config eager` 支持首提基线口径 | 否 | 同卡三套压测可复核 |
| 13 | `scripts/start_aiterfa_service.sh`：改用交付包组装的调优库路径（`$WORK_DIR/rocblas_tuned/library_gpu4`）+ 6 文件 SHA256 校验 + 端口参数 | 否 | 原先指向的 `tools/auto_select_tools/...` 是调优现场目录（已被 `.gitignore` 排除），换容器后不一定存在，会让最优版**静默退回 stock 库** |
| 14 | `scripts/run_same_card_bench.sh`（新）：同一张卡上依次跑 eager → graph 基线 → 最优版 | 否 | 保证三套数据同卡可比 |
| 15 | `scripts/collect_tuned_bench_into_submission.sh`：纳入仓库，并扩展为归集三套同卡原始数据 | 否 | 原先该脚本只在节点侧临时目录，新容器收尾会找不到 |

**基线材料全部保留**：首提基线的 `submission/bench/round1.jsonl` / `round2.jsonl`、
`summary.md`、`summary_detailed.md` 未删除；GitLab 标签 `baseline-submission` 指向首提基线提交。

---

## 8. 回退方法

```bash
cd /root/private_data/workspace/robo/submission

# 一键回到首提基线（eager + stock 库 + 默认 attention）
pkill -f "scripts/serve_vllm_patched.py" || true
pkill -f "VLLM::EngineCore" || true
sleep 8
USE_TUNED=0 setsid nohup bash run.sh > logs/run_stdout_final.log 2>&1 &
```

只回退其中一项：

```bash
# 只回退 GEMM 调优库，保留 graph
env -u ROCBLAS_TENSILE_LIBPATH bash tools/start_bench_service.sh graph 30000
# 只回退 attention 后端，保留 graph + 调优库
env -u VLLM_ATTENTION_BACKEND bash tools/start_bench_service.sh graph 30000
```

---

## 9. 未覆盖项与限制（如实标注）

| 项 | 状态 | 原因 |
|---|---|---|
| 容器换卡 | **已处理** | 2026-11-01 05:08 起容器被平台回收重建，**物理卡更换**。旧卡的绝对值降级为「跨卡存档」，收益结论改用同卡 A/B（§6 口径表）；`scripts/card_identity.sh` + `scripts/bench_set.py` 负责判定 |
| 赛事官方 ≥98% 一致率 | **BLOCKED** | 官方 GPU 参考输出、官方测试集、官方判分器均未下发；节点无外网 |
| 公开基准（RefSpatial 等）复跑 | NOT_RUN（本轮） | 本轮任务范围不含精度测试 |
| 16384/c64 档轮次 | **单轮** | 该档单轮耗时 3 h 48 min（128 请求排队 69 min，TTFT 均值 4,134,411 ms），两轮需 7.5 h+；其余五档均为两轮。表格已标注轮数 |
| 16384/c64 档的同节点 graph 基线对照 | NOT_RUN | 该档无同节点 graph 基线对照：基线补测结果因节点中断未能取回，而该档基线单轮同样需 ~3.8 h |
| 本轮压测结果的取回方式 | **会话日志转录** | 2026-11-01 05:08 起 dcu_1026 容器不可达（网关端口可连但收不到 SSH banner，同网关其他容器正常），节点工作目录无法访问；最优版六档结果由压测客户端与汇总脚本的**原样输出**转录恢复（见 `reports/evidence/bench_recovered_20261101/`），因此**不是客户端原始 jsonl 文件**，其中 3 个档位的 `total_throughput` / `p99_ttft_ms` / `p99_tpot_ms` 未被留存，表中以 `—` 标注 |
| 同节点 graph 基线补测（16384/并发1、16384/并发8、4096/并发64） | NOT_RUN | 补测于 2026-11-01 04:36 启动（graph 基线服务 READY after 230 s），预期 05:05 完成，但节点在 05:08 断开，结果未能取回；同节点 graph 基线对照因此只覆盖 4096/并发1、4096/并发8 两档 |
| c64 以上并发摸高 | NOT_RUN | 受单卡 KV cache 容量限制（253,312 token），本轮只跑官方六档 |
| 长稳 | NOT_RUN | 用户暂缓 |
| HIP Trace / PMC 剖析 | PARTIAL | 10-30 只用 rocBLAS layer3 调用频次定位热点，未采 HIP Trace（`reports/23` §5 缺口清单） |
| 第二轮「干净」GEMM 搜索 | **未完成** | `tools/auto_select_tools_all` 的 InferenceTuner 只完成 1/11 任务即中断；按规范（`enable_bestkernel_csv=0`）的干净搜索尚未跑完，当前生效仍是 10-30 那份库 |
| 平台截图 / 区域 / 服务代理地址 / 镜像链接 | 待用户补 | 节点访问不到平台控制台（`platform/README.md`） |

---

## 10. 复现命令

```bash
# 0) 环境
cd /root/private_data/workspace/robo
source /opt/dtk/env.sh

# 1) 组装调优库（幂等）
bash submission/scripts/make_tuned_rocblas_lib.sh

# 2) 起服务（最优版）
cd submission && bash run.sh && cd ..

# 3) 完整六档压测（两轮）
bash scripts/run_full_bench.sh --config candidate \
  --tiers "4096:1:2,4096:8:2,16384:1:2,16384:8:2,4096:64:2,16384:64:1"

# 4) 汇总与对比
python3 scripts/summarize_full_bench.py \
  --run-dir outputs/fullbench_candidate_20261031_231152 --config candidate
python3 scripts/build_tuning_comparison.py --out reports/tuning_comparison.md

# 5) 提交包自检
bash tools/check_submission.sh
```
