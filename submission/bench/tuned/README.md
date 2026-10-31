# 最优版压测数据（2026-10-31）

**配置**：graph(非 eager) + GEMM 算子调优库 + `VLLM_ATTENTION_BACKEND=ROCM_AITER_FA`
（即 `env.sh` 里 `USE_TUNED=1` 的提交配置）。

**口径**：单卡 K100AI（K500SM_AI / gfx928 / 63.98 GiB）/ vLLM 0.15.1+das.opt1.alpha.dtk2604 /
BF16 / TP=1 / `--gpu-memory-utilization 0.85 --max-model-len 32768` /
`--max-num-batched-tokens 4096 --max-num-seqs 64 --no-enable-prefix-caching` /
`--limit-mm-per-prompt '{"image":8}'`；官方 `bench_serving` 客户端，image 数据集，
1×1080p 图/请求，输出 1024 token，seed 1，请求数 `max(8, 2×并发)`。

## ⚠️ 本目录是「会话日志转录版」，不是客户端原始 jsonl

原始文件在节点工作目录
`outputs/fullbench_candidate_20261031_231152/`（每档每轮一个 `round*.jsonl` +
同名 `.log`，另有 `run_config.txt` / `service_start.log` / `worker_env.txt` /
`summary.md` / `agreement.md`）。

2026-11-01 05:08 起该容器不可达（网关端口可连但收不到 SSH banner，同网关其他
容器正常），无法取回原始文件。本目录是把压测客户端与汇总脚本的**原样输出**
逐字转录恢复的实测值，字段名与官方口径一致；文件名前缀 `recovered_` 用于与
原始文件区分。

**未留存**：`total_throughput`、部分档位的 `p99_ttft_ms` / `p99_tpot_ms`。
这三项**没有做任何推算**，在报告表中以 `—` 标注。

来源与逐项说明见 `../../reports/evidence/bench_recovered_20261101/README.md`。

## 文件

| 文件 | 内容 |
|---|---|
| `recovered_round*_<输入>_1024_c<并发>.jsonl` | 每档每轮的实测值（字段与官方一致） |
| `summary.md` | 六档汇总（官方 13 列格式，取记录轮）与逐轮波动 |
| `../round1.jsonl`、`../round2.jsonl` | **首提基线（eager）的原始数据**，未做任何改动 |

## 节点恢复后的补全

```bash
cd /root/private_data/workspace/robo
bash scripts/collect_tuned_bench_into_submission.sh outputs/fullbench_candidate_20261031_231152
python3 scripts/fill_tuning_tables.py
bash tools/record_revision.sh
bash tools/check_submission.sh
```

`collect_tuned_bench_into_submission.sh` 会用**原始文件**覆盖本目录
（并写入 `run_config.txt`、`worker_env.txt`、`summary.md` 等），
`fill_tuning_tables.py` 重跑后报告表中的 `—` 会被真实数值填满。

> **跨节点的绝对值不可直接比较**：2026-10-31 节点在同配置下比 2026-10-30 节点
> 慢约 30%（已逐项核对服务参数、graph 捕获列表与 worker 环境），
> 收益结论一律用同节点同轮的 A/B，详见 `../../reports/25_tuning_delivery_20261031.md` §6。
