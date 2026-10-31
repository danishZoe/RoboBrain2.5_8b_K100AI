# 最优版压测数据（2026-10-31，会话日志转录版）

> ⚠️ **本目录内容不是压测客户端的原始 jsonl 文件。**
>
> 原始文件在节点工作目录
> `/root/private_data/workspace/robo/outputs/fullbench_candidate_20261031_231152/`
> （每档每轮一个 `round*.jsonl` + 同名 `.log`）。
> 2026-11-01 05:08 起该容器不可达（网关端口可连接但收不到 SSH banner，
> 同网关其他容器正常），无法取回原始文件。
>
> 本目录是把**压测客户端与汇总脚本的原样输出**逐字转录恢复的实测值，
> 字段名与官方口径一致。文件名前缀 `recovered_` 用于与原始文件区分。

## 数据来源

| 来源 | 内容 | 覆盖 |
|---|---|---|
| `summarize_full_bench.py` 汇总表 | 官方 13 列完整行 | 候选 7 轮（4096/c1×2、4096/c8×2、16384/c1×2、16384/c8 第 1 轮） |
| `bench_serving` 客户端日志块 | `Successful requests` / `Benchmark duration` / `Output token throughput` / `Mean E2E Latency` / `Mean TTFT` / `Mean TPOT` | 候选另 4 轮（16384/c8 第 2 轮、4096/c64×2、16384/c64） |
| 逐字段打印 | `total_output_tokens`、`p99_ttft_ms`（16384/c64） | 16384/c64 |

## 可精确复原的量

- `duration = completed × 1024 / output_throughput`
  —— 客户端口径（输出打满 1024 token）。已用 7 条完整行逐条核对，
  复原值与汇总表原文一致（如 16384/c64：131072 / 9.589760785790007 = 13667.911 s）。
- `request_throughput = completed / duration`

## 未留存的量（表中以 `—` 标注，不是 0）

`total_throughput`（需要逐请求输入 token 数）、部分档位的 `p99_ttft_ms` 与
`p99_tpot_ms`。这三项在转录来源里没有出现，**没有做任何推算**。

核心指标（`completed` / `failed` / `duration` / `output_throughput` /
`mean_ttft_ms` / `mean_tpot_ms` / `mean_e2e_latency_ms`）十一轮齐全。

## 复现与补全

节点恢复后，用下面的命令取回原始文件并覆盖本目录：

```bash
cd /root/private_data/workspace/robo
bash scripts/collect_tuned_bench_into_submission.sh outputs/fullbench_candidate_20261031_231152
bash tools/record_revision.sh
bash tools/check_submission.sh
```

并重跑 `scripts/fill_tuning_tables.py` 让表格与原始文件完全对齐
（届时 `—` 会被真实数值填满）。

## 对比口径

| 文件 | 内容 |
|---|---|
| `candidate_13col.tsv` | 最优版（graph + GEMM 调优库 + `ROCM_AITER_FA`）11 轮 |
| `graph_baseline_10-31_13col.tsv` | 同节点 graph 基线（stock 库、默认 attention）2 档 |
| `eager_repro_10-31_13col.tsv` | 同节点首提基线（eager）复现 5 轮 |

跨节点的绝对值不可直接比较（同配置在 10-30 节点快约 30%），
收益结论一律用同节点同轮 A/B，详见 `reports/25_tuning_delivery_20261031.md` §6。
