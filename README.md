# RoboBrain2.5-8B-NV × K100AI × vLLM —— 工作区索引

> 本文件在**远端**是 `/root/private_data/workspace/robo/README.md`，在**本地**是
> `agent_work/adaptation/README.md`（远端产物的离线镜像）；两处内容一致。

远端工作目录：`/root/private_data/workspace/robo`（SSH 别名 `dcu_1026`，持久盘）
本目录是**本地镜像**：远端产物与本文件保持双向同步，用于离线查阅与备份。

> 远端是真源：`ssh dcu_1026 "cd /root/private_data/workspace/robo && ls"`。
> 远端已挂载的自动流水线会在压测结束后自动刷新
> `submission/bench/`、`submission/README.md` 第 7 节、`env.log` / `smoke.log` /
> `verify_result.json`，并跑一次提交包自检（`tools/check_submission.sh`）。

---

## 1. 先看哪几份

| 顺序 | 文件 | 内容 |
|---|---|---|
| 1 | `submission/README.md` | **正式提交报告**（官方模板八章）：环境、适配方案、启动脚本、Smoke、精度、性能、注意事项 |
| 2 | `reports/05_final.md` | 总览与索引：是否跑通、任务范围、一条可复现命令、优化前后同口径指标、尚存限制 |
| 3 | `reports/10_phase5_bench.md` | 阶段 5 冒烟 + 官方六档压测（含两处口径错误的根因与修正） |
| 4 | `reports/11_tuning_experiments.md` | 服务级单变量调优实验 E1–E4（判定标准实验前登记） |
| 5 | `reports/08_gap_vs_official_baseline.md` | 与官方基线逐项差距 + 10 条口径风险 |
| 6 | `reports/09_vllm_adaptation.md` | vLLM 适配全过程（4 个阻塞 + grounding 根因闭环） |
| 7 | `reports/00_RUNBOOK.md` | 运行手册与 15 条实测故障排查 |

## 2. 报告全集

| 文件 | 主题 |
|---|---|
| `00_RUNBOOK.md` | 运行手册与故障恢复指南 |
| `01_environment.md` | 第一阶段：环境与模型检查 |
| `02_adaptation.md` | 第二阶段：最小推理与适配 |
| `03_benchmark.md` | 第三阶段：功能与性能基线 |
| `04_tuning.md` | 第四阶段前六项单变量实验（A–F） |
| `05_final.md` | **最终报告**（总览 + 索引） |
| `06_sglang_compatibility.md` | SGLang 不可用的根因定案 |
| `07_device_model_mapping.md` | K100AI ≡ K500SM_AI（gfx928）机型映射核实 |
| `08_gap_vs_official_baseline.md` | 与官方基线的差距评估 |
| `09_vllm_adaptation.md` | vLLM serving 适配 |
| `10_phase5_bench.md` | 阶段 5：冒烟与六档压测（含 §3.7 计量口径服务端核算、§3.8 随机 vs 自然提示词对照实验） |
| `10a_server_counter_crosscheck.md` | 服务端引擎计数器逐档交叉核对表（输出 token 计量口径的独立证据） |
| `11_tuning_experiments.md` | 服务级调优实验 E1–E4（判定标准实验前登记） |
| `bench_summary_custom.md` | 自有压测客户端结果 + 与官方客户端交叉验证 |

## 3. 目录

| 目录 | 内容 |
|---|---|
| `submission/` | **提交包**：`README.md`、`env.sh`、`env_check.sh`、`run.sh`、`smoke.sh`、`verify.py`、`bench.sh`、`requirements-extra.txt`、`patches/`、`envs/vllm_shim/`、`scripts/`、`samples/`、`baselines/`（空，放官方基线）、`bench/`、产物日志 |
| `reports/` | 报告全集 |
| `scripts/` | 环境检查、最小推理、探针、压测、图像规范化、输出格式桥、A/B 脚本、输出 token 计量核验 |
| `tools/` | 汇总、提交包自检、产物刷新、两段收尾流水线、内存安全网、报告注入器（README/报告 05/10/11）、服务端计数器核算 |
| `outputs/` | 原始测量数据（`phase3_baseline` / `phase4*` / `phase5_*` / `phase5_ab_*`） |
| `configs/`、`patches/`、`envs/`、`samples/`、`logs/`、`annotated/` | 配置、patch、垫片、样例、日志、目视校验图 |

## 5. 远端常用命令

```bash
ssh dcu_1026
cd /root/private_data/workspace/robo

# 提交包全流程
cd submission && bash run.sh && bash env_check.sh && bash smoke.sh && source ./env.sh && python verify.py
BENCH_MODE=image bash bench.sh            # 官方六档压测

# 阶段 5 一键
bash scripts/run_phase5.sh --serve-only --keep-server --mem-frac 0.85 --ctx 32768

# 自检
bash tools/check_submission.sh
bash tools/final_check.sh
```

## 5.1 远端正在自动运行的收尾流水线

`tools/pipeline_after_bench.sh` 已在远端以**独立会话**（`setsid`）运行，
**SSH 断连或本地会话结束都不会中断**。它依次执行：

| 步 | 动作 | 产物 |
|---|---|---|
| 1 | 等官方六档压测跑完（`bench/round2.jsonl` 出现 6 条） | — |
| 2 | 汇总 | `submission/bench/summary.md`、`summary_detailed.md` |
| 3 | 回填 README 第 7 节的官方 13 列表格 | `submission/README.md` |
| 4 | 刷新产物（对齐 `MAX_LEN=32768` 的最终配置） | `env.log`、`smoke.log`、`verify_result.json` |
| 5 | 提交包自检 | 输出到流水线日志 |
| 6 | **E1** 非 eager A/B（70 GiB 看门狗） | `outputs/phase5_ab_noneager/<tag>/` |
| 7 | **E2** `max-num-batched-tokens` A/B（4096/8192/16384） | `outputs/phase5_ab_bnb/<tag>/` |
| 8 | **E3** rocBLAS layer3 trace（诊断） | `outputs/phase5_gemm_trace/<tag>/` |
| 9 | 恢复最终基线服务（eager / 32768 / 30000） | `submission/server.log` |

日志：`logs/pipeline_after_bench.log`；内存安全网：`logs/guard_cgroup.log`（85 GiB 阈值）。

### 5.1.1 第二阶段流水线（`tools/pipeline_final.sh`，同样 `setsid` 独立会话）

排在第一阶段之后自动执行，日志 `logs/pipeline_final.log`：

| 步 | 动作 | 产物 |
|---|---|---|
| 1 | 等第一阶段流水线退出 | — |
| 2 | 等基线服务（30000）健康 | — |
| 3 | **E4** 计量口径对照实验（随机 token 提示词 vs 自然散文提示词） | `outputs/phase5_token_accounting/result.json` |
| 4 | §3.8 自动写回报告 | `reports/10_phase5_bench.md` |
| 5 | E1/E2/E3 结果自动写回报告 | `reports/11_tuning_experiments.md` |
| 6 | 重跑官方汇总并再次回填 README §7 / 报告 10 表格 | `submission/README.md`、`reports/10_phase5_bench.md` |
| 7 | 刷新产物 | `env.log`、`smoke.log`、`verify_result.json` |
| 8–9 | 提交包自检 + 最终检查（含回填 `reports/05_final.md` 的两张表） | `tools/final_check.sh` |

**取最终结果**：

```bash
# 看流水线进展
tail -80 /root/private_data/workspace/robo/logs/pipeline_after_bench.log
tail -80 /root/private_data/workspace/robo/logs/pipeline_final.log

# 一键拉回本地镜像（本目录）
powershell -File <local>\adaptation\tools\sync_from_remote.ps1
```

## 5.2 仍然只有用户能关闭的缺口

1. **精度项**：官方 GPU 基线文件、官方测试集、官方判分器未下发 → 节点无外网，
   只能由用户向平台索取。`verify.py` 已确保**没有证据绝不报 PASS**。
2. **`platform/` 截图**与平台区域 / 服务代理映射地址：需用户在控制台截图补入。
