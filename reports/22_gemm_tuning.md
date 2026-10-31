# 实验 E5：厂商 GEMM 算子调优（auto_select_tools）—— 已验证收益

日期：2026-10-30（远端时间）
状态：**已验证，建议采纳**；是否转正由用户决定。

> 前情：报告 11（E3，2026-10-27）判「本节点不具备执行条件」，依据是节点上没有厂商调优工具与 shape 数据。
> 本次用户提供了 `material/GEMM调优/auto_select_tools` 工具包，执行条件补齐，E3 重开并完成。

## 1. 实验设计

- **目的/假设**：在真实负载的 GEMM shape 集合上重选 rocBLAS kernel（写回 `GemmGenerality*_Eco` 配置），
  提升 decode 阶段吞吐；decode 形状（N=1~64）默认库 kernel 选择次优。
- **基线（对照组）**：graph 服务 = vLLM `0.15.1+das.opt1.alpha.dtk2604`，无 `--enforce-eager`，
  其余参数与首提配置一致（TP=1 / BF16 / mem_util 0.85 / max-model-len 32768 /
  max-num-batched-tokens 4096 / max-num-seqs 64 / no-prefix-caching / image≤8），
  使用 `/opt/dtk` 默认 `library_gpu4`。基线数据 = `tmp/graph_bench/baseline_oldlib/round2.jsonl`
  （graph_bench3 + 同配置 trace 轮，均为旧库）。
- **唯一主要变量**：`ROCBLAS_TENSILE_LIBPATH` 指向调优生成的新 `library_gpu4`
  （`tools/auto_select_tools/optimization_configs/new/config/library_gpu4`）。
  `/opt/dtk` 原库未做任何修改（`update_target_config=0`）。
- **负载**：官方压测客户端（SGLang bench_serving 适配脚本）image 数据集，每请求 1 张 1080p 图，
  输出 1024 token，`--seed 1`；档位 4096×c1/c8/c64 + 16384×c1，两轮取第二轮。

## 2. 关键事实（现场核查结论，2026-10-30）

| 核查项 | 结果 |
|---|---|
| 卡型归属 | K100AI = gfx928，rocBLAS 库目录 `library_gpu4`（94 个 gfx928 文件）；kernel 名含 `KME1`，运行时实际加载 `_Eco` 后缀配置（删除 `_Eco` 文件报 `Failed to open file: GemmGenerality_Eco.dat`）→ 工具参数 `hardware_type=ECO` |
| C++ 工具链 | `hipcc`（/opt/dtk/bin）、make、g++、jsoncpp 均在位；tune-all / tune-quick / bench-test 全部编译通过 |
| Python 依赖 | pandas、yaml 已满足 |
| layer3 trace | `ROCBLAS_LAYER=3` 有效（此前报告 11 用 `ROCBLAS_TRACE=3` 无效，是两个不同开关）；`ROCBLAS_DEBUG_GENERALITY=1` 可逐 GEMM  dump 实际 kernel 与来源 |
| 真实负载 shape | 带 trace 的 graph 服务跑 3 档 bench（4096 c1/c8、16384 c1），提取 **218 个唯一 bf16 size**（NN 215 / TN 3），覆盖 LLM 投影（4096/6144/12288/24576）、LM head（151936, TN）与视觉编码器 |

## 3. 执行

```bash
# 1) trace 采集（graph 服务 + ROCBLAS_LAYER=3，replay bench 流量后停服）
scripts/start_trace_service.sh
BENCH_TIERS='4096:1,4096:8,16384:1' bash tmp/bench_trace_run.sh
bash scripts/extract_fc_size.sh logs/serve_graph_trace_20261030-123545.log tmp/fc_size.log

# 2) 调优（单卡，耗时约 50 分钟）
cd tools/auto_select_tools   # configs/tuning_args.txt: mode=user source=layer3 hardware_type=ECO
                             # strategy=common gpu_ids=0 enable_bestkernel_csv=1 update_target_config=0
python3 -u Auto_tuning.py

# 3) AB 验证（同口径 4 档两轮）
scripts/start_tunedlib_service.sh   # 唯一区别: export ROCBLAS_TENSILE_LIBPATH=<新库>
BENCH_TIERS='4096:1,4096:8,4096:64,16384:1' bash tmp/bench_trace_run.sh
```

## 4. 结果

### 4.1 算子级（工具自带新旧库对比，bench-test 各 2 轮取最大 gflops）

107 个 size 完成对比：**104 提升 / 1 持平 / 2 微降（-0.1%~-0.2%，噪声内）**；
gflops_ratio mean +27.5%、median +19.1%、max +108%。

| size (M,N,K) | 旧 gflops | 新 gflops | 提升 | 说明 |
|---|---|---|---|---|
| 4096,64,4096 | 15383 | 32004 | +108% | decode 投影，c64 档 |
| 6144,64,4096 | 19582 | 37675 | +92% | decode QKV，c64 档 |
| 151936,1,4096 (TN) | 471 | 861 | **+83%** | LM head，c1 decode，每 token 一次 |
| 151936,8,4096 (TN) | 3880 | 5630 | +44% | LM head，c8 |
| 4096,1,4096 | 588 | 623 | +6% | decode 投影，c1 |

机制：默认库对小 N（decode 形状）选了 MMAC 通用 kernel，调优后换成 `GEMVMT256` 系列 GEMV 专用 kernel。

### 4.2 服务级 AB（端到端，同口径第二轮）

| 档位(in/conc) | 旧库 tok/s | 新库 tok/s | 提升 | TPOT 旧→新 ms |
|---|---|---|---|---|
| 4096/1 | 33.38 | 41.33 | **+23.8%** | 29.18 → 23.40 |
| 4096/8 | 151.96 | 182.15 | **+19.9%** | 46.88 → 38.38 |
| 4096/64 | 196.88 | 207.34 | **+5.3%** | 226.18 → 214.27 |
| 16384/1 | 25.87 | 26.18 | +1.2%（噪声内） | 35.04 → 34.66 |

注：16384/1 档旧库两次基线分别为 23.50（graph_bench3）与 25.87（trace 轮），该档仅 8 请求、波动大；+1.2% 不宣称收益。

### 4.3 正确性与稳定性

- 压测全程 **0 失败请求**（8+16+128+8 ×2 轮），无设备错误、无 OOM。
- 6 样例探针（768 token）：**6/6 可解析**；与 transformers 基线逐坐标比对，漂移 ≤3/1000，
  与 graph 基线自身的后端间漂移（1~30/1000）同量级；仅 affordance 样例从 (901,600) 变为 (904,600)。
- kernel 切换实证：`ROCBLAS_DEBUG_GENERALITY=1` 显示 (4096,1,4096) 由旧 kernel(index 288) 切换到
  调优 kernel `GEMVMT256`(index 712)。

## 5. 结论与回退

- **结论：采纳。** 官方口径核心档（4096/8）+19.9%、并发 1 档 +23.8%，远超同档波动（<2.3%）；
  高并发 c64 +5.3%；仅 16384/1 档在噪声内。无失败请求，输出质量变化在后端间既有漂移范围内。
- **回退**：不设置 `ROCBLAS_TENSILE_LIBPATH`（或用 `tools/start_bench_service.sh graph 30000` 启动）即回默认库，秒级回退；原库未被修改。
- **依赖**：新库目录 `tools/auto_select_tools/optimization_configs/new/config/library_gpu4`（约 780MB）
  必须随工作目录保留；启动脚本 `scripts/start_tunedlib_service.sh`。
- **遗留**：c64 档提升较小（调度/排队主导）；16384 档 prefill 主导的 size（如 m=4096 n≈820-864）有 2 个微降 0.2%（噪声内）。
  精度门槛（官方 ≥98% 一致率）仍依赖官方 GPU 参考输出，与本实验无新增阻塞。

## 6. 产物清单

| 产物 | 路径 |
|---|---|
| 调优工具（已编译） | `tools/auto_select_tools/` |
| 新 rocBLAS 配置库 | `tools/auto_select_tools/optimization_configs/new/config/library_gpu4/` |
| size trace | `tmp/fc_size.log`（218 唯一 size）、`logs/serve_graph_trace_20261030-123545.log` |
| 算子级对比 | `tools/auto_select_tools/optimization_artifacts/infer/compare/csv/bf16_{NN,TN}.csv` |
| 基线/AB 压测 | `tmp/graph_bench/baseline_oldlib/round{1,2}.jsonl`、`tmp/graph_bench/round{1,2}.jsonl`、`bench/summary.md` |
| 一致性探针 | `outputs/gemm_tune_ab/probe/probe_result.json`、`outputs/gemm_tune_ab/agreement.md` |
| 启动脚本 | `scripts/start_trace_service.sh`（trace）、`scripts/start_tunedlib_service.sh`（新库）、`scripts/extract_fc_size.sh` |
| 调优全量日志 | `tools/auto_select_tools/run.log` |
