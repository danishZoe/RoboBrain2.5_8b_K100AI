# 15 · 公开基准评测协议冻结与差异表

> 适用范围：CV-Bench / RefSpatial-Bench / TraceSpatial-Bench 三个**公开上游评测集**。
> 本文是《RoboBrain2.5-8B-NV_精度测试验收要求.md》§4.1 / §4.2 / §4.3 要求的
> "协议冻结"与"协议差异表"，**在跑全量评测之前写定**。
>
> ⚠️ 本文的三个评测集**都不是赛事官方验收数据集**。赛事官方精度项仍为
> `BLOCKED`（缺 GPU 参考输出与官方判分器），本文只提供公开基准上的证据。

---

## 1. 坐标契约（最关键的一项）

### 1.1 结论

**RoboBrain2.5-8B-NV 的空间任务输出是 0~1000 相对坐标，换算基准是原图尺寸**
（不是缩放后尺寸）。深度 `d` 为米。

### 1.2 证据链（四条独立证据）

| # | 证据 | 出处 |
| --- | --- | --- |
| E1 | 上游 docstring 明确 "x,y are relative (0~1000)"，`rel_to_abs` 做 `x/1000*w`、`y/1000*h`，并 clamp 到图内 | `third_party/RoboBrain2.5/inference.py`（`draw_on_image`） |
| E2 | FlagEvalMM 官方判分对 RoboBrain 系提供 `_norm1k` 变体：`text2pts_norm1k` 做 `int((x/1000.0)*width)` | `FlagEvalMM/flagevalmm/evaluator/point_utils.py`；`tasks/robo_spatial_home/evaluate_norm1k.py`、`tasks/where2place/evaluate_norm1k.py` |
| E3 | 本机实测：affordance 图 640×426，模型输出 `[(901, 600)]` —— 像素坐标不可能超出 640×426 | `outputs/phase5_probe_canonical/probe_result.json` |
| E4 | 技术报告 §2.1.2 明确采用解耦的 `(u, v, d)` 表示，`d` 是绝对深度 | `material/robo_2601.14352v1.pdf` 第 4 页 |

### 1.3 图像预处理与换算基准

* 我们的服务侧预处理 `scripts/image_canonical.py` **逐像素复现**上游
  `qwen_vl_utils.process_vision_info`（实测 `official vs canonicalised` 的
  `max|diff| = 0.000000`；`official vs raw upload` 为 `0.470588`）。
* 因此"用**原图**尺寸把 0~1000 换算成像素"与上游 `rel_to_abs` 口径一致。
* ⚠️ `smart_resize` 会把长宽分别 round 到 28 的倍数，长宽比有 ≤1.5% 漂移
  （上游同样如此，不影响与上游可比）；但**绝不能**用 canonical 之后的尺寸换算。

---

## 2. 协议差异表

三个维度上存在**多套官方实现**。下表逐条列出差异与本次采用版本。

### 2.1 维度 A：RefSpatial 的坐标口径

| 实现 | 入口 | 坐标处理 | 越界点 | 与 0~1000 输出的关系 |
| --- | --- | --- | --- | --- |
| A1 | `RoboRefer/Evaluation/summarize_acc.py` 的 `"RoboBrain"` 分支 | `text2pts(..., is_absolute=True)`，**按像素**读 | 越界点计 0 分（计入均值分母） | ❌ 直接用会把 0~1000 当像素 → 全部越界 → 静默 0 分 |
| A2 | 同上，但 `is_absolute=False` | 只有**浮点**坐标才乘 W/H；整数原样保留 | 同上 | ⚠️ 整数 0~1000 仍被当像素 → 同样失效 |
| A3 | `FlagEvalMM/flagevalmm/evaluator/point_utils.py::text2pts_norm1k` + `calculate_mask_score` | `x/1000*W`、`y/1000*H` | **clamp 回图内**（与 A1 不同） | ✅ 原生吻合 |

**A1 是 RoboBrain 2.0 的坐标契约**（2.0 输出像素坐标，官方 prompt 也不给尺度
说明）。RoboBrain **2.5** 改用 0~1000 相对坐标，直接套用 A1 会静默判 0。

**本次采用**：

* **主口径 = A1 + 推理侧显式换算**：推理层把 0~1000 换算成**原图像素**（整数），
  再交给**未修改的**官方 `compute_accuracy`（`is_absolute=True` 语义）。
  这是"输出适配"，不是"改判分"。
* **交叉核对 = A3**：用官方 FlagEvalMM 的 `text2pts_norm1k` 独立复算一遍，
  验证换算没有引入偏差。
* **对照口径 = A1 逐字不换算**：量化"尺度不一致"造成的分差。

### 2.2 维度 B：RefSpatial / TraceSpatial 的提示词

| 实现 | 入口 | 提示词 |
| --- | --- | --- |
| B1 | `RoboRefer/Evaluation/test_benchmark.py::get_prompt` 的 `"RoboBrain"` 分支 | `f"{prompt} Please provide its 2D coordinates."` |
| B2 | `RoboBrain2.5/inference.py` 的 pointing 模板 | `{text}. Please provide its 2D coordinates. Your answer should be formatted as a tuple, i.e. [(x, y)], where the tuple contains the x and y coordinates of a point satisfying the conditions above.` |
| B3 | `RoboTracer/Evaluation/test_benchmark.py::get_prompt` 的 `"RoboTracer"` 分支 | `Point the {2D|3D} object-centric visual trace for the task "{prompt}". Your answer should be formatted as a list of tuples, i.e., ...` |
| B4 | `RoboBrain2.5/inference.py` 的 trajectory 模板 | `Please predict 3D end-effector-centric waypoints to complete the task successfully. The task is "{text}". Your answer should be formatted as a list of tuples, i.e., [(x1, y1, d1), ...]` |

B2 是 B1 + 一句格式约束；B3 与 B4 都是元组格式。

**本次采用**：RefSpatial 主口径 = **B2**（模型原生 SFT 格式），对照 = **B1**；
TraceSpatial 主口径 = **B3**（官方 TraceSpatial 评测入口，上游自带参考预测
`Model_Prediction_TraceSpatial_{2D,3D}.jsonl` 用的就是它），对照 = **B4**。

> B2 拼接时会把原始 prompt 的结尾句点变成 `..`；我们的规则是**只在自拼句点的
> 地方**去掉**一个**结尾句点（`common.strip_one_trailing_period`），官方逐字
> 提示词（B1）保持原文不动。

### 2.3 维度 C：TraceSpatial 的 model_id 与解析分支

官方 `summarize_acc.py` 的解析分派只认 `"RoboTracer"` / `"Qwen"` / `"Gemini"`，
其它名字直接 `raise ValueError("Unsupported model: ...")`。

我们的模型输出是元组格式，与 `"RoboTracer"` 分支一致，但模型名是
RoboBrain2.5 → **model_id 取 `RoboTracer_RoboBrain2.5-8B-NV`**，
让官方判分脚本走它为元组输出准备的
`text2pts_normalize(..., is_normalized_1000=True)` 分支 —— **判分代码零改动**。

### 2.4 维度 D：CV-Bench 的答案抽取

| 实现 | 出处 | 规则 |
| --- | --- | --- |
| D1 | `FlagEvalMM/model_zoo/vlm/robobrain/model_adapter.py::_process_single_result` | 输出含 `</think>` → 用官方 `parse_think_answer_string` 取 `<answer>...</answer>`；否则整段作为答案 |
| D2 | `FlagEvalMM/flagevalmm/evaluator/common_types.py::evaluate_multiple_choice` | `maybe_clean_answer` → `process_multiple_choice`（关键词切分 → 取最后一行 → `findall("[A-Z]")` → 取首字母） |

**已披露的补丁**：官方 `parse_think_answer_string` 在没有 `<answer>` 标签时返回
`None`，D2 对 `None` 取 `len()` 会抛 TypeError。本次在这种情况下**退回整段原始
输出**，其余情况与官方完全一致。

### 2.5 维度 E：CV-Bench 的**图像输入协议**

论文明确说明评测框架是 FlagEvalMM（p.13），因此 CV-Bench 的输入路径应与
`FlagEvalMM` 的 CV-Bench 链路一致。该链路的图像处理是：

```text
tasks/cv_bench/process.py :  data["image"].save(output_dir/data["filename"])
flagevalmm/prompt/prompt_tools.py::encode_image :
        可选按 max_long_side / min_short_side 缩放
        -> img.save(output, format="JPEG", quality=95)   # 有损
        -> base64 data URL
```

而本次接入走的是 **上游 RoboBrain2.5 `inference.py` 的 `qwen_vl_utils` 路径**
（28 倍数重采样 + 无损 PNG），这是为了与 RefSpatial / TraceSpatial 的
`RoboRefer` / `RoboTracer` 直接推理路径保持像素一致。两者并不相同：

| 维度 | 水平 | 说明 |
| --- | --- | --- |
| 缩放策略 | `canonical` / `none` / `long1500` / `maxpix501760` | `canonical`=上游 `qwen_vl_utils`；`long1500`=FlagEvalMM 仓库里 Qwen 系 `model_config` 的 `max_long_side`（RoboBrain 的配置未公开）；`maxpix501760`=本项目 transformers 基线的像素预算 |
| 编码格式 | `png` / `jpeg95` | `jpeg95`=官方 `encode_image` 的 JPEG q95（有损） |

**冻结选择**：主口径 = `canonical_png`（与另两个基准像素一致、可复现本项目
transformers 基线）；`raw_jpeg95` 是"最贴近官方 FlagEvalMM 链路"的对照。
两者的实测差异见 `reports/16_public_bench_results.md` §5.5 的因子实验
（Count 子任务 788 题，8 组）。

---

## 3. 冻结的生成配置（三个基准一致）

| 项 | 值 | 依据 |
| --- | --- | --- |
| 框架 | vLLM `0.15.1+das.opt1.alpha.dtk2604.torch290.2605261031.gdb9e4b` | 本机实测可用 |
| 精度 | BF16（权重与计算同为 bfloat16） | 模型声明与本机实测 |
| 解码 | `temperature=0.0, top_p=1.0, seed=0`（greedy） | 上游官方评测脚本同样用 `temperature=0.0` |
| `max_tokens` | 768 | 上游 `inference.py` 的 `max_new_tokens=768` |
| 思考开关 | 不开启（提示词里不注入 `<think>` 推理段；CV-Bench 按官方适配器注入 `<think></think><answer>`） | 官方适配器原文 |
| 图像数 | 1 | 三个基准均为单图任务 |
| 图像预处理 | `scripts/image_canonical.py`（复现上游 `qwen_vl_utils`） | 见 §1.3 |
| 深度 / 内参 | **不作为模型输入**；仅判分需要 | 官方 `RoboTracer` 分支的 `RoboTracer_Intrinsics_Depth` 才带内参+深度，我们用纯 RGB 分支 |
| 并发 | **16**（评测并发，非性能压测口径） | 固定并写入配置哈希；注意 vLLM 批内归约顺序可能带来极小的数值差异，重复性检查见 §6 L7 |

---

## 4. 逐基准的评分入口

| 基准 | 判分入口 | 本次调用方式 | 输出文件 |
| --- | --- | --- | --- |
| CV-Bench | `FlagEvalMM/tasks/cv_bench/evaluate.py::get_result` | **官方函数**，直接 import | `accuracy_2d_ade / accuracy_2d_coco / accuracy_3d_omni / accuracy_2d / accuracy_3d / accuracy` |
| RefSpatial-Bench | `RoboRefer/Evaluation/summarize_acc.py` | **官方 CLI 逐字运行**（`--model_name` / `--task_name`）+ 直接 import `compute_accuracy` | `Accuracy: mean, Evaluated, Total` |
| TraceSpatial-Bench | `RoboTracer/Evaluation/summarize_acc.py` | **官方 CLI 逐字运行**（`--model_name` / `--task_name 2D\|3D`） | 2D 路径指标 + Start-in-Mask / End-in-BBox；3D 路径指标 + Start/End 距离 + 成功率 + 碰撞率 |

### 4.1 RefSpatial 默认范围

《精度测试验收要求》§3.1：**Location 100 + Placement 100** 为默认验收范围。
`Unseen` 77 条是 Location/Placement 的**重叠子集**，**不得**与前两组相加当 277 道
独立题 → 本次对 Unseen **单列**，不计入默认总分。

### 4.2 分母（RefSpatial）

官方 `compute_accuracy` 在**解析抛异常**时 `continue` 跳过该题，均值分母 =
实际计分题数。本次同时给出：

* `total`：总题数
* `evaluated`：官方实际计分题数
* `parse_failed`：解析失败题数
* `audit_score_full_denominator`：全量分母、失败计零的审计分数

两种结果分别命名并报告差异。

### 4.3 TraceSpatial 的 3D 降级风险

官方 `compute_accuracy_3d` 的**碰撞检查**依赖 `open3d`（TSDF 体素化 + DBSCAN
去噪）。缺 `open3d` 时官方脚本会自己捕获成
`[Warn][3D] Collision check failed` → `collision_flag=False`，且目标点云构造
失败会让 **3D Start 直接算成 0**。因此 `open3d` 是 3D 判分的**硬依赖**，必须装上
（隔离前缀 `envs/pylibs`），并在结果里报告是否发生降级。

---

## 5. 官方对照分（论文数值，仅作对照信息）

出处：`material/robo_2601.14352v1.pdf` §6.1 Table 2 / §6.2 Table 3，
**RoboBrain-2.5 (8B) NV** 变体：

| 基准 | 论文 NV 分数 | 说明 |
| --- | --- | --- |
| CV-Bench | **94.58** | 官方聚合：`(Acc_ADE20K + Acc_COCO)/2` 与 `Acc_Omni3D` 再取平均 |
| RefSpatial | **60.50** | 论文未写明是 Location+Placement 三组的哪一套聚合 |
| TraceSpatial | **3D Start 83 / 3D End 63 / Success 44** | 与官方 `summarize_acc.py` 的 `Start Success Rate` / `End Success Rate` / `Overall Success Rate` 一一对应 |

⚠️ 论文未公开其评测子集切分、图像预处理与端点实现，**不能直接当作通过阈值**
（《精度测试验收要求》§7.1）。本文只把它作为"同量级对照"，并逐项标注差异。

---

## 6. 已知偏差与限制

| # | 事项 | 影响 | 处理 |
| --- | --- | --- | --- |
| L1 | CV-Bench 图像尺寸极不统一（256×256 ~ 2048×1536），其中超过 transformers 基线 `max_pixels=501760` 的图像，服务端**不额外降采样**（vLLM 侧未设该上限） | 与我们自己的 transformers 基线在部分大图上像素不同 | 在 `data_manifest.json` 记录超预算图像数量；论文侧协议未知，作为限制披露 |
| L2 | TraceSpatial 3D 的 `d` 单位按官方脚本假定为米 | 若模型输出单位不同，3D 距离指标整体偏移 | 官方脚本即如此，不改；与论文口径一致的可能性最高 |
| L3 | RefSpatial 的 `Unseen` 与 Location/Placement 重叠 | 加总会重复计题 | 单列，不计入默认总分 |
| L4 | 官方判分代码以 `main()` 的硬编码相对路径工作（`./RefSpatial-Bench`、`./TraceSpatial-Bench`、`./outputs`） | 必须在约定目录下运行 | 用符号链接在运行目录内复现官方目录约定，不改官方代码 |
| L5 | FlagEvalMM 的 `process.py` 需要联网 `load_dataset` | 本节点离线 | 直接从**本地 parquet** 构造 annotations，判分仍用官方 `get_result`；该改动是"输入来源适配"，已披露 |
| L6 | 赛事官方 GPU 参考输出未下发 | 赛事精度项 `BLOCKED` | 不自建官方评测集、不伪造基线 |

---

## 7. 产物与复现

```bash
cd /root/private_data/workspace/robo
# 1) 服务（首次提交基线配置：eager / 30000）
bash tools/start_bench_service.sh eager 30000

# 2) 全流程（manifest → 三个基准 → 离线重判分）
bash tools/run_public_bench.sh http://127.0.0.1:30000/v1/chat/completions baseline-eager all

# 3) 单独复跑某一阶段
bash tools/run_public_bench.sh http://127.0.0.1:30000/v1/chat/completions baseline-eager rescore
```

产物位置（远端工作目录内）：

```
outputs/bench_public/
  manifest/{data_manifest.json,data_manifest.md,cvbench_images/,cvbench_image_map.json}
  refspatial/{summary.json,upstream_scaled/,official_raw/,...}
  tracespatial/{summary.json,official/,upstream/}
  cvbench/{summary.json,annotations.json,predictions_official.json,per_sample.json}
  rescore_check.json
logs/public_bench_<run_id>/{manifest,refspatial,tracespatial,cvbench,rescore}.log
```
