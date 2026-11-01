# 跨后端一致性比对（transformers 基线 vs serving 后端）

- 基线：`/root/private_data/workspace/robo/outputs/phase3_baseline/baseline_raw.json`
- 对比对象：`/root/private_data/workspace/robo/outputs/fullbench_candidate_20261031_231152/probe/probe_result.json`
- 后端：vllm  接口：http://127.0.0.1:30000/v1/chat/completions

| 样例 | 任务 | 基线输出 | serving 输出 | 最大坐标偏差(/1000) | 完全一致 |
|---|---|---|---|---|---|
| official_general | general | `The image shows a white ceramic mug filled with a dark liqui` | `The image shows a white ceramic mug filled with a dark liqui` | - | ❌ |
| official_grounding | grounding | `(0, 309, 620, 989)` | `(0, 309, 620, 1000)` | 11 | ❌ |
| official_affordance | pointing | `[(901, 600)]` | `[(904, 600)]` | 3 | ❌ |
| official_pointing | pointing | `[(582, 725)]` | `[(582, 725)]` | 0 | ✅ |
| official_navigation | pointing | `[(400, 376)]` | `[(401, 375)]` | 1 | ❌ |
| official_trajectory | trajectory | `[(294, 525, 0.577), (405, 350, 0.607), (525, 300, 0.72)]` | `[(294, 525, 0.572), (400, 375, 0.605), (495, 325, 0.72)]` | 30 | ❌ |

**逐字符/逐坐标完全一致率：1/6 = 16.7%**

> 说明：这是「同一台卡、同一批样例、不同推理后端」之间的一致性，
> 不是官方要求的「国产卡 vs GPU 参考输出」一致率（后者需要官方 GPU 参考文件）。
> 但它能证明 serving 路径与已验证的 transformers 路径在数值上没有分歧。
