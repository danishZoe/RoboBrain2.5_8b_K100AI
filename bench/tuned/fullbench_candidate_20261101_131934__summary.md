# 完整压测汇总（candidate）

- 运行目录：`/root/private_data/workspace/robo/outputs/fullbench_candidate_20261101_131934`
- 结果条数：1

| random_input_len | random_output_len | max_concurrency | completed | failed | duration | request_throughput | output_throughput | total_throughput | mean_ttft_ms | p99_ttft_ms | mean_tpot_ms | p99_tpot_ms | mean_e2e_latency_ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16384 | 1024 | 64 | 128 | 0 | 13480.8 | 0.00949502 | 9.7229 | 192.865 | 4.06223e+06 | 6.02678e+06 | 1395.95 | 1826.13 | 5.49028e+06 |

## 轮次与波动

| 输入 | 并发 | 轮数 | 输出吞吐 tok/s | 轮间极差 |
| --- | --- | ---: | --- | --- |
| 16384 | 64 | 1 | 9.72 | 0.00% |

