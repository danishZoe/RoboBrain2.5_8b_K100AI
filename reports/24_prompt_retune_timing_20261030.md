# 新提示词后的精简调优流程与耗时（2026-10-30）

- 实验 ID：`prompt_retune_20261030_171709`
- 远端产物：`/root/private_data/workspace/robo/outputs/prompt_retune/prompt_retune_20261030_171709/`
- 目标：复用已有环境及候选优化，测量新提示词下走一遍必要调优流程的实际耗时，不做无依据的全量重跑。
- 现场：单卡 K100AI（运行时 `K500SM_AI` / gfx928 / 63.98 GiB），vLLM graph，BF16，TP=1。
- 正式计时起点：2026-10-30 17:17:09 +0800（epoch `1793351829`）；前期旧结果与环境核对约 4 分钟另计。

## 提示词版本核对

远端 `samples/tasks_official.json` SHA256 为 `1f33e1801009e8778b005ddb37856643d89c950f2248e7c31f2531c317773d45`，与本地旧版相同；`scripts/bench_serving_vllm_image.py` 仍为 2026-10-27 版本。当前未定位到实际提示词差异，因此本轮冻结的是远端当前文件，不宣称已验证一个可识别的新提示词版本。

## 精简协议与判定

1. 复用旧热点结论及候选：GEMM 调优库 + `VLLM_ATTENTION_BACKEND=ROCM_AITER_FA`。
2. stock graph 无 Profiler 基线：先跑 `4096/c8`，一轮预热、一轮记录；候选同口径 AB。
3. 候选吞吐提升超过 3%、零失败且输出未缩短，才补 `4096/c1`；为控制本轮时长，c64、16384 输入档和长稳不补跑。
4. 短 rocBLAS layer3 Trace 只用于核对新提示词是否引入未覆盖 GEMM shape；正式性能测量关闭 Trace。
5. 质量检查使用固定 6 样例，报告参考与候选差值。公开全量基准、长稳及赛事官方精度不在本轮快速范围，分别标为 `NOT_RUN` / `BLOCKED`，不以非空输出替代精度结论。

## 阶段耗时

| 阶段 | 实际耗时 | 状态/证据 |
| --- | --- | --- |
| 旧结果、环境及提示词核对 | 约 4 分钟（正式计时前） | 完成 |
| stock graph 启动 + 4096/c8 两轮 | 212 + 192 + 193 = **597 秒（9 分 57 秒）** | 完成；启动 212 秒，每轮含客户端预处理 |
| 短 Trace 与 shape 对比 | 237 + 198 = **435 秒（7 分 15 秒）** | 完成；102 个完整 GEMM 配置 |
| 候选启动 + 4096/c8 两轮 | 243 + 339 = **582 秒（9 分 42 秒）** | 完成 |
| 候选 4096/c1 两轮（复用服务） | **555 秒（9 分 15 秒）** | 完成 |
| 固定样例质量回归 | **16 秒** | 完成 |
| 候选 DAT/TXT 打开验证 | **258 秒（4 分 18 秒）** | 完成；仅启动监听，不跑压测 |
| 汇总与交付 | 约 3 分钟 | 完成；不占 GPU |
| **设备流程累计** | **2443 秒（40 分 43 秒）** | 含库打开验证，不含前期核对和汇总 |
| **正式流程端到端总耗时** | **3026 秒（50 分 26 秒）** | 17:17:09–18:07:35，含阶段间操作；加前期核对约 54 分 26 秒 |

## 性能结果

| 配置/档位 | 轮次 | 完成/计划 | 实际输出 token | 时长 s | 输出吞吐 tok/s | TTFT 均值 ms | TPOT 均值 ms | E2E 均值/P99 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stock graph，4096/c8 | 预热 | 16/16 | 16384 | 107.70 | 152.12 | 5941.56 | 46.80 | 53818.06 / 53938.99 |
| stock graph，4096/c8 | 记录 | 16/16 | 16384 | 108.96 | **150.36** | 5907.24 | 47.45 | 54451.06 / 54754.61 |
| 候选，4096/c8 | 预热 | 16/16 | 16384 | 89.63 | 182.80 | 5049.64 | 38.84 | 44782.39 / 45153.17 |
| 候选，4096/c8 | 记录 | 16/16 | 16384 | 90.62 | **180.81** | 5700.69 | 38.68 | 45270.45 / 45459.06 |
| 候选，4096/c1 | 预热 | 8/8 | 8192 | 198.81 | 41.21 | 796.62 | 23.51 | 24849.92 / 25169.79 |
| 候选，4096/c1 | 记录 | 8/8 | 8192 | 201.84 | **40.59** | 807.09 | 23.87 | 25228.65 / 25819.39 |

- 同轮 `4096/c8` 记录轮收益：`(180.8060795 / 150.3605123 - 1) = **+20.25%**`。stock 两轮波动约 1.16%，候选约 1.10%；收益超过波动。
- 所有压测请求均成功，输出预算均为完整 1024 token/请求，没有把缩短输出计作提速。
- 候选是旧流程已分别验证的“GEMM 库 + AITER FA”组合；本轮只验证组合在新冻结输入下仍有效，不重新归因两个变量各自的收益。
- 本轮没有新跑 stock `4096/c1`，因此 c1 只报告候选绝对值，不与旧提示词结果拼成同轮 AB。

## Trace、质量与决定

- 新 Trace 有 102 个完整 GEMM 配置，其中 12 个不在旧搜索输入中；全部是 `N=257/1771/1832` 的 NN 视觉/prefill 投影，decode shape 未变。证据：`trace/new_shapes_only.log`。
- 因 decode shape 未变且组合候选 AB 提升 20.25%，本轮跳过约 50 分钟的重复 GEMM 搜索；12 个新增 prefill shape 留作下一轮候选，不宣称已对其调优。
- 固定 6 样例均正常停止且可解析。与旧 Transformers 参考的坐标差为 0–30/1000，逐字符/逐坐标完全一致 1/6；该对比不是官方精度。由于本轮未预先冻结新的质量阈值，精度回归记为 `BLOCKED（阈值未定义）`，赛事官方精度仍为 `BLOCKED`。
- 决定：保留现有组合候选，不启动新搜索。c64、16384 输入、公开全量基准和长稳均为 `NOT_RUN`。

## 候选库生效与回退

- 候选 AI DAT：`tools/auto_select_tools/optimization_configs/new/config/library_gpu4/GemmGenerality_AI_Eco.dat`，SHA256 `ad891f317f5c097776e014d738f17e84d2e4af7dfc7ad352f01f0d60d28025f3`。
- 候选 Add DAT：同目录 `GemmGeneralityAdd_Eco.dat`，SHA256 `23e30c02a186d0c9a0cc851c2874752f1fc3f7fdd89ba77bf2cd5c5806324f58`。
- launcher 中 `ROCBLAS_TENSILE_LIBPATH` 和 `VLLM_ATTENTION_BACKEND=ROCM_AITER_FA` 正确。隔离启动的 inotify 记录到候选目录实际打开 `GemmGenerality_AI_Eco.dat`、`GemmGeneralityAdd_Eco.dat` 及对应 SolutionName TXT，`queue_overflow=false`。inotify 本身不归属 PID，证据组合为隔离服务启动、launcher 环境、文件哈希和目录打开事件；原始记录为 `library_open_verification/open_events.json`。
- 回退命令：`env -u ROCBLAS_TENSILE_LIBPATH -u VLLM_ATTENTION_BACKEND -u VLLM_ROCM_USE_AITER bash tools/start_bench_service.sh graph 30000`。
- 验证结束后已停止服务，`hy-smi` 显示 VRAM 0%、HCU 0%，无本任务残留推理进程。

## 实际执行命令与产物

以下命令均在 `cd /root/private_data/workspace/robo` 后执行。首轮 baseline 的本地等待在预热轮后到时，因此记录轮使用第二行断点续跑；其余阶段均完整成功。

```bash
RUN_ID=prompt_retune_20261030_171709 TIERS=4096:8 ROUNDS=2 bash scripts/run_prompt_retune.sh baseline
RUN_ID=prompt_retune_20261030_171709 REUSE_SERVICE=1 ROUND_START=2 TIERS=4096:8 ROUNDS=2 bash scripts/run_prompt_retune.sh baseline
RUN_ID=prompt_retune_20261030_171709 TIERS=4096:8 ROUNDS=1 bash scripts/run_prompt_retune.sh trace
RUN_ID=prompt_retune_20261030_171709 TIERS=4096:8 ROUNDS=2 bash scripts/run_prompt_retune.sh candidate
RUN_ID=prompt_retune_20261030_171709 REUSE_SERVICE=1 TIERS=4096:1 ROUNDS=2 bash scripts/run_prompt_retune.sh candidate_c1
RUN_ID=prompt_retune_20261030_171709 bash scripts/run_prompt_retune.sh quality
RUN_ID=prompt_retune_20261030_171709 bash scripts/verify_prompt_candidate.sh
```

- 本地脚本：`agent_work/scripts/run_prompt_retune.sh`、`agent_work/scripts/verify_prompt_candidate.sh`；远端同内容位于 `scripts/`。
- 原始输出：`/root/private_data/workspace/robo/outputs/prompt_retune/prompt_retune_20261030_171709/`。
- 分阶段计时：`timings.tsv`；输入哈希：`input_hashes.sha256`；各阶段原始 JSONL、服务日志和质量对比均保留。
