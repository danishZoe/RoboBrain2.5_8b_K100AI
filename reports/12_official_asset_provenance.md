# 附：官方 upstream 资产与任务提示词溯源

> 目的：证明本任务服务侧的**输入侧语义**（图像资产 + 任务提示词 + 解析正则）与官方 upstream 逐字一致。
> 这不是官方精度证据——官方基线文件 / 评测子集 / 判分器仍未下发，精度项仍是 `BLOCKED`
> （见 `reports/08_gap_vs_official_baseline.md`）。
> 本报告回答一个具体问题：**官方评测集下发后，我们的输入侧能不能直接对拍官方判分？**

- 检查时间：2026-10-27 17:03 (+08:00)
- upstream 仓库：`https://github.com/FlagOpen/RoboBrain2.5.git`
- upstream HEAD commit：`af98c932aac9ff715d70da177088d7bb95573ff7`（2026-03-01T04:04:19+08:00，`Update RoboBrain2.5-4B`）
- 复现命令：`bash tools/fetch_upstream_head.sh`、`bash tools/diff_upstream.sh`
- 机器可读版本：`samples/upstream_provenance.json`

---

## 1. inference.py 与 upstream HEAD 逐字一致

工作目录内的 upstream 快照（`third_party/RoboBrain2.5/inference.py`）与刚从 GitHub 拉的 HEAD：

```
diff --strip-trailing-cr -q third_party/RoboBrain2.5/inference.py \
                      third_party/RoboBrain2.5-upstream-HEAD/inference.py
→ IDENTICAL
```

唯一差异是行尾符：本地快照 263 行 CRLF、upstream 0 行 CRLF（纯 LF）。去掉 CR 后**无任何内容差异**，
因此**任务提示词模板与解析正则天然就是官方原文**，不是我们改写的等价实现。

## 2. 5 张 demo 资产与 upstream HEAD 逐字节相同

| 本地文件 | upstream 路径 | SHA256 | 与 upstream HEAD 一致 |
|---|---|---|---|
| `samples/images/official_affordance.jpg` | `assets/demo/affordance.jpg` | `53f3cd8e…c8711` | ✅ |
| `samples/images/official_grounding.jpg` | `assets/demo/grounding.jpg` | `c1c1613b…d7dfc` | ✅ |
| `samples/images/official_navigation.jpg` | `assets/demo/navigation.jpg` | `c1f076cf…8c42` | ✅ |
| `samples/images/official_pointing.jpg` | `assets/demo/pointing.jpg` | `0758d1e8…de89fa` | ✅ |
| `samples/images/official_trajectory.jpg` | `assets/demo/trajectory.jpg` | `5d5c0e0b…05393e` | ✅ |

## 3. 任务提示词模板（upstream inference.py 第 50–58 行，逐字）

| task | 模板 |
|---|---|
| `general` | `{text}` |
| `pointing` | `{text}. Please provide its 2D coordinates. Your answer should be formatted as a tuple, i.e. [(x, y)], where the tuple contains the x and y coordinates of a point satisfying the conditions above.` |
| `trajectory` | `Please predict 3D end-effector-centric waypoints to complete the task successfully. The task is "{text}". Your answer should be formatted as a list of tuples, i.e., [(x1, y1, d1), (x2, y2, d2), ...], where each tuple contains the x and y coordinates and the depth of the point.` |
| `grounding` | `Please provide the bounding box coordinate of the region this sentence describes: {text}.` |

远端实际发送的提示词已在 `outputs/tasks_official/task_results.json` 的 `model_prompt` 字段留证，
逐字与上表一致。

## 4. 解析正则（upstream inference.py 第 110/116/122 行，逐字）

| task | upstream 正则 |
|---|---|
| `trajectory` | `(\d+),\s*(\d+),\s*([+-]?\d+\.\d+)` |
| `pointing` | `\(\s*(\d+)\s*,\s*(\d+)\s*\)` |
| `grounding` | `\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]` |

`scripts/answer_adapter.py` 第 49–51 行的 `RE_TRAJ` / `RE_POINT` / `RE_BOX` 就是这三条的逐字复制。

**本任务额外加的两条只是表面格式桥，不改任何数值**（`scripts/answer_adapter.py` 第 39–46 行）：

- `<bbox>x1 y1 x2 y2</bbox>` → `[x1, y1, x2, y2]`
- `<|box_start|>[x1, y1, x2, y2]<|box_end|>` → `[x1, y1, x2, y2]`

原因是同一后端在不同回答模式下表面形式不同，而 upstream 正则只认方括号形式。
原文始终与归一化文本同时保留，可审计。**不改模型、采样语义、权重。**

## 5. upstream demo 的 7 个 Case 与我们的覆盖情况

| Case | task | prompt | 资产 | 状态 |
|---|---|---|---|---|
| 1 | general | `What is shown in this image?` | upstream 用外网 COCO 图 | ⚠️ 改为本地 `official_affordance.jpg`（验收标准要求固定本地资源、不依赖临时外网图片） |
| 2 | grounding | `the person wearing a red hat` | `assets/demo/grounding.jpg` | ✅ 覆盖 |
| 3 | pointing | `the affordance area for holding the cup` | `assets/demo/affordance.jpg` | ✅ 覆盖 |
| 4 | trajectory | `reach for the banana on the plate` | `assets/demo/trajectory.jpg` | ✅ 覆盖 |
| 5 | pointing | `Identify spot within the vacant space that's between the two mugs` | `assets/demo/pointing.jpg` | ✅ 覆盖 |
| 6 | pointing | `Identify spot within toilet in the house` | 脚本写 `navigation1.jpg` | ✅ 用实际存在的 `navigation.jpg` |
| 7 | pointing | `Identify spot within sofa in the house` | 脚本写 `navigation2.jpg` | ❌ **NOT_REPRODUCIBLE** |

## 6. 已知缺口（如实记录，不掩盖）

1. **upstream 自身的 demo 脚本引用了两个仓库里不存在的文件**：
   `assets/demo/navigation1.jpg` 与 `assets/demo/navigation2.jpg` 在 upstream HEAD 中均不存在。
   - Case 6 用实际存在的 `navigation.jpg` 覆盖；
   - **Case 7 无法从官方资产复现**，标为 `NOT_REPRODUCIBLE`。
     不自造图片（自建评测集会被官方判分时剔除，且触及赛事红线）。
2. **官方解析正则的覆盖边界**：pointing / grounding 只认**整数**坐标，trajectory 的深度要求
   `带小数且可带符号`。科学计数法、负坐标、浮点坐标不被 upstream 正则覆盖。
   官方评测若使用更宽的容差或更全的解析，以官方判分器为准；我们保留原文 + 解析值 + 失败原因，
   解析失败不剔除出分母。

## 7. 对官方材料到位后的意义

官方 `baselines/*.json` + `data/*.jsonl` + `judge.py` 一旦下发：

- 输入侧（图像资产、提示词、解析口径）已与官方 upstream 对齐，**无需再改任何适配代码**；
- `verify.py` 会自动走「按冻结配置真实调用服务 → 逐样本评测 → 按官方口径算一致率 → 逐项 PASS/FAIL」，
  逐样本证据写 `verify_evidence.json`；
- 唯一仍需确认的是**官方子集的图像资产与官方判分容差**是否与 upstream 一致——
  这一项在材料到位前无法核实，本报告不代为假设。

## 8. 容器联网（本次新增，2026-10-27）

远端此前判定"无外网"是**路径问题**，不是平台限制：

- 联网指南写的 `~/.ai_user_info/ai_proxy` 在本容器**不存在**（root 的 `$HOME` 是 `/root`，
  而家目录挂载点是 `/public/home/scnjz2s4uc`）；
- 实际代理文件在 `/root/private_data/.ai_user_info/ai_proxy`
  （= `/public/home/scnjz2s4uc/.ai_user_info/ai_proxy`）；
- 已固化 `tools/env_net.sh`，一次 source 即同时得到 DCU 运行时与平台出口代理：

```bash
source /root/private_data/workspace/robo/tools/env_net.sh
```

验证结果（source 后）：`www.scnet.cn` 301、`gitlab.scnet.cn:9002` 302、
`developer.sourcefind.cn` 200、`huggingface.co` 200、`pypi.tuna.tsinghua.edu.cn` 200；
`torch 2.9.0 | hip 6.3.26113 | cuda_avail True`。

代理文件含明文口令，**只 source 不打印**，不写入任何脚本、日志、报告或提交物。
