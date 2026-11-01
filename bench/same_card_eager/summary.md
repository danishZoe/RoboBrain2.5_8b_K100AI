# 完整压测汇总（eager）

- 运行目录：`/root/private_data/workspace/robo/outputs/fullbench_eager_20261101_171235`
- 结果条数：1

| random_input_len | random_output_len | max_concurrency | completed | failed | duration | request_throughput | output_throughput | total_throughput | mean_ttft_ms | p99_ttft_ms | mean_tpot_ms | p99_tpot_ms | mean_e2e_latency_ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16384 | 1024 | 64 | 128 | 0 | 14024.4 | 0.00912696 | 9.34601 | 185.407 | 4.21961e+06 | 6.26565e+06 | 1454.5 | 1820.71 | 5.70756e+06 |

## 轮次与波动

| 输入 | 并发 | 轮数 | 输出吞吐 tok/s | 轮间极差 |
| --- | --- | ---: | --- | --- |
| 16384 | 64 | 1 | 9.35 | 0.00% |

