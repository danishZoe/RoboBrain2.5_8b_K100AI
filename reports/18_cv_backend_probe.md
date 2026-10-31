# CV-Bench 后端对拍（小样本）

日期：2026-10-28

为释放显存，停止本任务 vLLM 服务后，以同一模型、同一张 K100AI、同一批 canonical 图像和同一 prompt 分别运行 Transformers 与 vLLM。

| sample | task | GT | vLLM | Transformers |
|---|---|---|---|---|
| CVBench/0 | Count | C | C | C |
| CVBench/342 | Relation | A | A | A |
| CVBench/1438 | Depth | A | A | A |
| CVBench/2038 | Distance | A | A | A |

4/4 样本答案一致，随后已用 `tools/start_bench_service.sh eager 30000` 恢复 vLLM，健康检查 200。

限制：该对拍是后端一致性小样本证据，不替代全量精度评测；四类各 1 题不足以估计准确率。
