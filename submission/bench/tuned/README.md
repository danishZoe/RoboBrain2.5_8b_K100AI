# 最优版压测原始数据（提交配置）

**配置**：graph(非 eager) + GEMM 算子调优库 + `VLLM_ATTENTION_BACKEND=ROCM_AITER_FA`
（即 `USE_TUNED=1` 的提交配置，见 `../../env.sh`）。

**会话**：当前节点（同一张卡）
card_id=host:crdnotebook-2114695384420274177-scnjz2s4uc-41571:boot:62ce6c81-9a7b-44b6-b994-b5054faa7f6d

**口径**：单卡 K100AI（K500SM_AI / gfx928 / 63.98 GiB）/ vLLM 0.15.1+das.opt1.alpha.dtk2604 /
BF16 / TP=1 / `--gpu-memory-utilization 0.85 --max-model-len 32768` /
`--max-num-batched-tokens 4096 --max-num-seqs 64 --no-enable-prefix-caching` /
`--limit-mm-per-prompt '{"image":8}'`；官方 bench_serving 客户端，image 数据集，
1×1080p 图/请求，输出 1024 token，seed 1，请求数 `max(8, 2×并发)`。

| 文件 | 内容 |
|---|---|
| `round*_<输入>_1024_c<并发>.jsonl` | 每档每轮的**客户端原始结果文件**（官方 13 列字段） |
| `round*_<输入>_1024_c<并发>.log` | 客户端完整日志（输入/输出 token 拆分、TTFT/TPOT/ITL 分布） |
| `recovered_round*.jsonl` | 节点不可达期间由**会话日志转录**恢复的结果（内容为压测客户端/汇总脚本的原样输出，**不是**客户端原始文件）；被新的客户端原始文件取代后保留作过程存档 |
| `run_config.txt` | 本次压测的配置指纹（卡指纹 `card_id=`、档位、模型、vllm 版本、stock/候选库哈希、git revision） |
| `service_start.log` | 服务启动输出（就绪耗时） |
| `worker_env.txt` | 推理 worker 进程的实际环境（调优开关是否真正进入进程） |
| `summary.md` | 汇总表（13 列 + 轮次波动） |
| `agreement.md` | 同口径探针与 transformers 基线的逐坐标比对 |

**首提基线（eager）的原始数据**：上一级目录 `bench/round1.jsonl` / `bench/round2.jsonl`
（10-27 节点的官方口径基准）；**同一张卡**上的 eager 与 graph 基线见
`bench/same_card_eager/` 与 `bench/same_card_baseline/`。

> `recovered_round*.jsonl` 是 2026-11-01 05:08 节点不可达期间，由压测客户端与汇总脚本的
> **原样输出转录**恢复的实测值（字段与官方口径一致，**不是**客户端原始文件），
> 文件名前缀用于与原始文件区分。被新的客户端原始文件取代后保留作过程存档。

> ⚠️ **跨节点的绝对值不可直接比较**。容器被平台回收重建后物理卡会更换，
> 收益结论一律用**同一张卡**上三套配置的 A/B（判定见 `../../scripts/bench_set.py`）。

> 同一配置若被拆到**多个服务实例**跑（例如 16384/并发64 这类单轮约 3.8 h 的重档
> 单独起一次服务），本目录会同时保留各次的 `round*.jsonl`；
> 同名但内容不同的文件以「运行目录名__原名」保存，**绝不静默覆盖**。
