# GEMM 算子调优证据（2026-10-30）

厂商 `auto_select_tools` 在本机实测 shape 上做选优的**关键证据**。
工具工作目录本身有 8 GB+（含 Tensile 二进制与旧库副本），不入仓库；
这里固化的是可复核的中小文件：

| 文件 | 内容 |
|---|---|
| `tuning_run.log` | 选优全流程日志（custom + infer 两轮 tuner 的起止与耗时） |
| `shape/bf16_*.csv` | 采集到的真实业务 GEMM shape（含 transpose / ld / alpha / beta / batch） |
| `shape/fc_size.log` | layer3 trace 原始行（218 个唯一 bf16 GEMM size 的来源） |
| `compare/custom_*.csv`、`compare/infer_*.csv` | 算子级选优对比（old vs new 的 GFLOPs 与耗时） |
| `tuning_args.txt`、`multinode_args.txt` | 本次搜索使用的参数 |

**生效的候选库**（4 改写 + 2 新增，相对 stock 的全部差异）随仓库提交在
`submission/rocblas_tuned/library_gpu4/`，哈希见 `reports/25_tuning_delivery_20261031.md` §3.1。

> 节点上的完整工作目录：`/root/private_data/workspace/robo/tools/auto_select_tools/`
> （未入仓库，`.gitignore` 已排除）。
