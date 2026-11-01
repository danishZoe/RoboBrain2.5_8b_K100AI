## 固定样例清单（来源 + 校验值）

上游仓库：`https://github.com/FlagOpen/RoboBrain2.5.git` @ `af98c932aac9`（2026-03-01T04:04:19+08:00）
提示词/正则一致性：`IDENTICAL_AFTER_STRIP_CR`（与 upstream HEAD 的 inference.py 逐字一致；唯一差异是本地快照用 CRLF 行尾、upstream 用 LF（diff --strip-trailing-cr 后无差异））

| 文件 | 字节 | sha256 | upstream 路径 | 本地校验 | 与 upstream HEAD 一致 |
|---|---|---|---|---|---|
| `samples/images/official_affordance.jpg` | 50047 | `53f3cd8e526ab685139db5389ffc9d1d78b5056800da230661900732cf8c8711` | `assets/demo/affordance.jpg` | 一致 | 是 |
| `samples/images/official_grounding.jpg` | 200997 | `c1c1613ba655fcc619b10aeedd213033ec4b7e536b7f567b09980acac80d7dfc` | `assets/demo/grounding.jpg` | 一致 | 是 |
| `samples/images/official_navigation.jpg` | 536198 | `c1f076cf528380d70ec7e1877088445de21fac484b2902103ba7899c287f8c42` | `assets/demo/navigation.jpg` | 一致 | 是 |
| `samples/images/official_pointing.jpg` | 72994 | `0758d1e8769f863431efb7a535315c2ad70226198f537608a7b39f2126de89fa` | `assets/demo/pointing.jpg` | 一致 | 是 |
| `samples/images/official_trajectory.jpg` | 56827 | `5d5c0e0bd124b0372fe1941d7385c2c2dff128fc4837b941d41682886005393e` | `assets/demo/trajectory.jpg` | 一致 | 是 |
| `samples/tasks_official.json` | 1410 | `1f33e1801009e8778b005ddb37856643d89c950f2248e7c31f2531c317773d45` | `-` | 本地固定样例 | - |
| `samples/tasks_multi_image.json` | 1095 | `e79b8800a924125385d1df45d665fb9707923d4b928c1c3826511d11951ee35a` | `-` | 本地固定样例 | - |

**校验结论**：5 个图像样例与 upstream 记录的 sha256 **逐位一致**，样例未被改动。

### 服务级输出材料（验收标准 §9「固定样例及输出」）

| 文件 | 内容 | 生成方式 |
|---|---|---|
| `samples/parsed_outputs.json` | 6 个官方样例的**模型原文**、**解析值**（框/点/轨迹）、结束原因、生成 token 数、耗时 | `scripts/make_sample_artifacts.py` 读服务级探针结果 `outputs/phase5_probe_canonical/probe_result.json` |
| `samples/annotated/*.png` | 6 张**坐标/框/轨迹叠加图**（红点=point、绿框=bbox、红线=trajectory、绿点=起点、蓝点=终点） | 复用 `scripts/robo_infer.py::draw_on_image()`，坐标系与上游 `draw_on_image()` 完全一致（0~1000 归一化 → 绝对像素） |

解析值一律由 `scripts/robo_infer.py::parse_answer()` 产出，其正则与官方 `inference.py:110-124`
**逐字一致**（见 `reports/03_benchmark.md` §5.1 的正则覆盖边界分析）；
叠加图与解析值共用同一套坐标换算，避免「图是另一套口径画的」。

> `general` 任务没有坐标类输出，因此 `parsed_outputs.json` 中该样例的 `parsed` 为空对象，
> 这是预期结果，不是解析失败。

### 其他配套产物（工作目录内，非提交包内）

- 任务提示词与解析正则的原始实现：`scripts/robo_infer.py`
- transformers 直跑（诊断口径）的逐题原文与解析值：`outputs/phase3_baseline/`
- 服务级 Smoke 的逐题输出：`outputs/phase5_smoke/`
- 服务侧同口径复跑（探针、A/B 实验）：`outputs/vllm_*`、`outputs/phase5_*`
