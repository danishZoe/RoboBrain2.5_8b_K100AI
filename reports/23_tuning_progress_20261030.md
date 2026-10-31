# 调优进度总览（2026-10-30，graph 基线之后的第一轮调优）

> 用途：会话回溯锚点。记录当日全部实验、已采纳配置、产物哈希、回退命令与待办。
> 口径：单卡 K100AI（gfx928，63.98 GiB）/ vLLM `0.15.1+das.opt1.alpha.dtk2604` / BF16 / TP=1。
> 压测负载：官方客户端适配脚本（image 数据集，1×1080p 图/请求，输出 1024 token，seed 1），
> 档位耗时参考（旧库/新库）：4096/c8 ≈ 1'48"/1'30" < 4096/c1 ≈ 4'05"/3'18" < 16384/c1 ≈ 5'17"/5'13" < 4096/c64 ≈ 11'06"/10'32"。
> 快速验证协议（用户定）：先跑最快的 1–2 档（c8 → c1），持平或提升才补其余档；快速迭代阶段不跑全量。

## 1. 当日时间线

| 时段 | 事项 |
|---|---|
| 12:25 | 确认现场：graph 服务（无 `--enforce-eager`）运行中，作为新基线 |
| 12:28–12:35 | 上传 `auto_select_tools` 并编译（hipcc 一次通过）；实测确认卡走 `library_gpu4` + `_Eco` 配置 → `hardware_type=ECO`；`ROCBLAS_LAYER=3` trace 开关有效（此前报告 11 用的 `ROCBLAS_TRACE=3` 无效） |
| 12:35–13:05 | 带 trace 的 graph 服务回放 3 档 bench，提取 **218 个唯一 bf16 GEMM size**（`tmp/fc_size.log`） |
| 13:16–14:05 | GEMM 选优（user/ECO/layer3/common，gpu_ids=0）：custom 11+11 task + infer，全流程约 50 分钟 |
| 14:02–15:00 | E5 服务 AB（4 档 ×2 轮）：**采纳** |
| 15:19–16:00 | E6 mem_util 0.90：c8 门槛 -2.8%，**否决**，回退 0.85 |
| 16:00–16:30 | E7 `VLLM_ATTENTION_BACKEND=ROCM_AITER_FA`：16384/c1 +17.9%，其余持平，**采纳** |
| 16:31–16:40 | E8 `VLLM_ROCM_USE_AITER=1`：启动崩溃（ViT `mm_encoder_attention` flash op），**否决** |
| 16:41–17:10 | E9 `compile_mm_encoder=true`：两档均持平，**否决** |
| 17:10 | 停服务、释放 GPU，整理进度（本文档） |

## 2. 实验台账（单变量，相对上一已采纳配置）

### E5 GEMM 算子调优 —— 采纳（详见 `reports/22_gemm_tuning.md`）

- 唯一变量：`ROCBLAS_TENSILE_LIBPATH` → 调优生成的新 `library_gpu4`（stock 未动）。
- 算子级：107 size 对比，104 升 / 1 平 / 2 微降(≤0.2%)；中位 +19.1%；LM head(c1) +83%。
  机制：decode 形状(N=1~64)由通用 MMAC kernel 换成 GEMV 专用 kernel。
- 服务 AB（第二轮）：4096/c1 **+23.8%**（33.38→41.33）；4096/c8 **+19.9%**（151.96→182.15）；
  4096/c64 +5.3%；16384/c1 +1.2%（噪声内，不宣称）。0 失败请求。
- 质量：6/6 样例可解析，坐标漂移 ≤3/1000，与后端间既有漂移同量级。

### E6 `--gpu-memory-utilization` 0.85→0.90 —— 否决

- 假设：更多 KV cache 提升并发档。c8 门槛 176.96 vs 182.15（-2.8%）→ 按协议中止（c64 未跑完即停）。
- 附带事实（INFO 日志）：KV cache = 276,480 token（0.90 时）；cudagraph capture sizes 1..128 已密集覆盖；
  `enable_chunked_prefill=True`。回退完成。

### E7 `VLLM_ATTENTION_BACKEND=ROCM_AITER_FA` —— 采纳

- 16384/c1：26.18 → **30.87（+17.9%）**，TPOT 34.66→28.85 ms，TTFT 不变（3655ms）。
- 4096/c1：41.33→41.36 / 41.52（持平）；4096/c8：182.15→183.26（持平）。0 失败。
- 质量：6/6 可解析，输出与 E5 探针逐字一致。机制：长上下文 decode 的 aiter FA kernel 更快。

### E8 `VLLM_ROCM_USE_AITER=1`（aiter 总开关）—— 否决

- 启动期 profile_run 在 ViT `mm_encoder_attention._forward_fa`（`flash_attn_maxseqlen_wrapper`）崩溃，
  EngineCore 起不来。证据：`logs/serve_graph_aiterall_20261030-163121.log`。回退完成。
- 未验证假设：aiter LINEAR（绕过 rocBLAS 路径）的收益因此无法在本镜像评估，除非单独绕开 ViT 路径。

### E9 `compile_mm_encoder=true` —— 否决

- 4096/c8：180.90 vs 183.26（-1.3%，噪声）；4096/c1：41.39 vs 41.36（持平），TTFT 781 vs 804（-3%，噪声内）。
- 无可测收益且启动更慢，不采纳。

## 3. 当前已采纳配置（可复现）

```bash
# 服务启动（= scripts/start_aiterfa_service.sh）
export ROCBLAS_TENSILE_LIBPATH=/root/private_data/workspace/robo/tools/auto_select_tools/optimization_configs/new/config/library_gpu4
export VLLM_ATTENTION_BACKEND=ROCM_AITER_FA
python3 submission/scripts/serve_vllm_patched.py \
  --model $MODEL_PATH --served-model-name robobrain2.5-8b-nv \
  --host 127.0.0.1 --port 30000 \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --dtype bfloat16 --trust-remote-code \
  --gpu-memory-utilization 0.85 --max-model-len 32768 \
  --max-num-batched-tokens 4096 --max-num-seqs 64 \
  --no-enable-prefix-caching --limit-mm-per-prompt '{"image": 8}'
# 即: graph(非 eager) + GEMM 调优库 + aiter FA
```

当前性能（相对 graph+stock 库基线）：c1 +24%，c8 +20%，16384/c1 +19~32%（基线波动大），c64 +5%。

**回退**：依次可退——去掉 `VLLM_ATTENTION_BACKEND`（回 FLASH_ATTN）；去掉 `ROCBLAS_TENSILE_LIBPATH`（回 stock 库，`tools/start_bench_service.sh graph 30000`）。

## 4. 产物与哈希

| 产物 | 绝对路径 | SHA256（截断） |
|---|---|---|
| 候选 AI DAT（实际生效文件） | `/root/private_data/workspace/robo/tools/auto_select_tools/optimization_configs/new/config/library_gpu4/GemmGenerality_AI_Eco.dat` | `ad891f317f5c…` |
| 候选 Add DAT | 同目录 `GemmGeneralityAdd_Eco.dat` | `23e30c02a186…` |
| 候选 SolutionName | 同目录 `GemmGenerality_SolutionName_AI_Eco.txt` | `5509bf83b23f…` |
| stock AI DAT（回退对照） | `/opt/dtk/lib/rocblas/library_gpu4/GemmGenerality_AI_Eco.dat` | `be0f15afdd77…` |

注：候选 `GemmGenerality_Eco.dat` 与 stock 完全相同（`db7af090bc46…`），调优只改了 AI/Add 两个 DAT。
生效证据：`ROCBLAS_DEBUG_GENERALITY=1` 实测 (4096,1,4096) kernel 288→712(GEMVMT256)、
(4096,1,12288) 与 (4096,64,12288) 均切换到新 kernel；服务 AB 见上。

关键数据/日志：
- AB 原始数据：`tmp/graph_bench/baseline_oldlib/round{1,2}.jsonl`（旧库）、`tmp/graph_bench/tuned_085/round{1,2}.jsonl`（E5 新库）、`tmp/gate_aiterfa.jsonl`（E7）、`tmp/gate_vitcompile.jsonl`（E9）
- 算子对比：`tools/auto_select_tools/optimization_artifacts/infer/compare/csv/bf16_{NN,TN}.csv`
- 一致性探针：`outputs/gemm_tune_ab/`（E5）、`outputs/aiterfa_ab/`（E7）
- 调优全日志：`tools/auto_select_tools/run.log`

## 5. 与新规范（AGENTS.md 2026-10-30 版）的差距清单

| 要求 | 现状 |
|---|---|
| 新搜索用 `enable_bestkernel_csv=0` | 本轮用了 1（增量模式）。**下一轮搜索必须改 0** 或按规范重跑一轮干净搜索对照 |
| size 去重保留 dtype/transpose/ld/stride/alpha/beta/batch | fc_size.log 保留了完整 rocblas-bench 行（含 ld/alpha/beta），去重按整行；M/N/K 摘要另算。dtype 全 bf16、B=1，无遗漏维度 |
| `ROCBLAS_TENSILE_LIBPATH` 生效需 worker 环境+SHA+实际打开证据 | 有 kernel 切换实证与服务 AB；**未取** `/proc/<worker>/environ` 级证据，下轮补齐 |
| 热点定位需 HIP Trace（§3） | 本轮依据 layer3 调用频次（fc_size 计数）定位；**未采 HIP Trace/PMC**，下轮按 `hygon-dcu-operator-tuning` 技能补 |
| 分层回归 L1–L4 | 有算子微基准(L1)、6 样例端到端(L4)；**缺** 模块/序列层(L2) 与执行路径层(L3) 系统记录 |
| 并发摸高/长稳（E6 阶段） | 未做（用户暂缓）；c64 以上、长稳均 `NOT_RUN` |
| 技能入口 | `agent_skill/tuning_skills/`（gemm/operator/diagnosis）下轮先读 SKILL.md 再动手 |

## 6. 候选下一步（按新规范的热点驱动，未排优先级）

- HIP Trace 定位剩余热点（16384/c1 TTFT 3.6s、prefill 路径、ViT 开销）。
- GEMM 第二轮：`enable_bestkernel_csv=0` 干净搜索 + 补 c64/16384-c8 流量 shape；验证 DAT schema 与独立数值回归。
- ViT/TTFT 专项（aiter MHA for ViT 的可行替代路径；E8 的崩溃说明总开关不可行）。
- c64+ 并发摸高与长稳（用户暂缓中）。
- INT8（参考仓库路线）：改变输出数值，**需用户明确授权**后才可立项。
