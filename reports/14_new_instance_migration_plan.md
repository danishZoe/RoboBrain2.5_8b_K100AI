# 附 14：换镜像迁移方案（保留 vLLM 适配成果，不手工重配 SGLang）

> 决策时间：2026-10-27 18:40
> 决策人：任务负责人（用户）——"之后重启的时候直接换镜像，但得保留之前 vllm 适配的结果，而不是自己重新配 sglang 环境"
> 事实依据：`reports/13_scnet_image_registry_survey.md`
> 一键重放脚本：`tools/bootstrap_vllm_adaptation.sh`

---

## 0. 决策

| 项 | 决定 |
| --- | --- |
| 镜像 | 新实例改用 **官方 `jsyadmin/sglang-0.5.12-ubuntu22.04-dtk26.04-py3.10:v1.0`**（torch 2.10.0 / vllm 0.18.1 / sglang 0.5.12 全栈），不再手工拼 SGLang 环境 |
| vLLM 适配成果 | **整体保留**（补丁 / 垫片 / 服务脚本 / 启动参数 / 探针 / 压测口径），新实例里**一条命令重放**，不手工重做 |
| SGLang | 换成镜像后**先跑通再谈计件**；跑不通就如实记 `NOT_ACHIEVABLE`，不伪造 |
| 精度 | 仍 `BLOCKED`（官方材料未下发），与镜像无关，换镜像不会改变这一点 |

---

## 1. 为什么换镜像是对的（而不是继续手工修 SGLang）

报告 06 的结论是：本节点 SGLang 的 8 组后端组合全灭，根因在 **aiter wheel 与 SGLang 期望接口不匹配**，而**厂商自己的三个 SGLang 构建共用同一套导入块** → 属于厂商内部口径问题，本机装任何版本都救不了。

而 registry 实证（报告 13 §3）显示：官方镜像里 SGLang 是**按完整依赖闭包**装的（`sglang==0.5.12 sglang-router sglang[diffusion]` + `aiter` + `deep_ep` + `tilelang` + `flash_mla` + `sgl-deep-gemm` + `causal_conv1d` + `deepgemm` + `lightop` + `lmslim` + `mooncake_transfer_engine`），
torch 也是 **2.10.0**（= das185/torch2100 栈，报告 06 §6.5 判定的"唯一可能路径"）。

**即：官方镜像装的就是我们手工拼不出来的那一套。** 手工修 = 把厂商的镜像构建脚本再跑一遍，既慢又不可复现；换镜像 = 直接用厂商成品。

---

## 2. 换实例会保留什么、丢什么

### 2.1 保留（持久化挂载 `/root/private_data`）

`/root/private_data/workspace/robo` 是**持久卷**，容器换掉也在：

```
scripts/ configs/ samples/ logs/ reports/ outputs/ envs/ cache/ tmp/
third_party/ patches/ submission/ tools/
```

其中必须完整带走的"适配成果"：

| 成果 | 位置 | 说明 |
| --- | --- | --- |
| tie_word_embeddings 垫片 | `envs/vllm_shim/sitecustomize.py` | 运行时注入，**不改 site-packages**；版本自适应 |
| 上游等价 patch | `patches/0001-vllm-qwen3-tie-word-embeddings.patch` | 供提交上游仓库，不落盘应用 |
| 服务入口 | `submission/scripts/serve_vllm_patched.py` | 垫片 + 官方同参启动 |
| 启动编排 | `submission/run.sh`、`submission/env.sh` | 含**必须**的 `--enforce-eager` 等参数 |
| 隔离依赖 | `envs/pylibs/`（`qwen_vl_utils`） | `--no-deps --target`，不污染公共环境 |
| 任务脚本 | `submission/scripts/{robo_infer,run_tasks,answer_adapter,image_canonical,...}.py` | 客户端图像规范化 + 输出格式桥（只改格式不动数值） |
| 探针/压测 | `submission/scripts/{probe_http_request,bench_service,benchmark}.py`、`submission/bench_serving_vllm_image.py` | 口径固定 |
| 已取基线 | `submission/bench/{round1,round2}.jsonl`、`bench/summary.md` | **12 轮官方格式压测数据，不可重取** |
| 精度自测 | `submission/verify_result.json` | 状态 `BLOCKED`（等官方材料） |
| 报告 | `reports/`（00–14） | |

### 2.2 丢（重启后需要重建，都是自动化的）

| 丢的 | 重建方式 |
| --- | --- |
| 运行时环境（镜像自带 site-packages） | 换镜像即换新环境 → 这正是本次决策的目的 |
| 已装到 `/usr/local/lib/python3.10/dist-packages` 的东西 | 本来就没装（红线：公共环境只读） |
| 正在跑的服务 / 压测进程 | 换实例后由 `bootstrap_vllm_adaptation.sh --install --start` 重建 |
| `env.log`（版本指纹） | `submission/env_check.sh` 重跑 |

---

## 3. vLLM 适配成果如何在新实例重放（一条命令）

```bash
# 体检 + 补齐隔离依赖 + 拉起服务 + 跑 6 样例探针
bash /root/private_data/workspace/robo/tools/bootstrap_vllm_adaptation.sh --install --probe
```

脚本做四件事，每件都是**已在本节点实测过**的动作，不引入新花样：

1. **体检**：工作目录完整性、python/torch/transformers/vllm/sglang/aiter 版本指纹、DTK 版本、cgroup（确认是否仍在容器内）。
2. **模型只读校验**：权重分片数、`config.json`、`index` 里是否真有 `lm_head` 张量（= 垫片取值的依据）。
3. **垫片判定**：用**全新解释器**探测 `Qwen3VLTextConfig.tie_word_embeddings` 是否缺失
   - 缺失 → 必须启用垫片；
   - 不缺失（新 transformers 已修）→ 垫片自动变空操作，**保持启用即可**，无副作用。
4. **拉服务**：`cd submission && bash ./run.sh`（内含 `--enforce-eager` 等已验证参数）+ 轮询 `/health`；失败自动打印 `server.log` 尾部。

### 3.1 新栈（vllm 0.18.1 + torch 2.10）会带来的差异 —— 三点预判与处置

| 预判 | 依据 | 处置 |
| --- | --- | --- |
| `tie_word_embeddings` 错位**可能已不存在** | vLLM 0.18.1 / transformers 版本都变了 | 脚本自动判定；两种情况都能跑 |
| vLLM 0.18.1 的 CLI 参数**可能改名** | 大版本跨越 | 启动失败时看 `server.log`；**只改参数、不改口径** |
| 新的 `serve_vllm_patched.py` 入口可能需要调整 | 它是按 0.15.1 的 `vllm serve` 入口写的 | 保留脚本 + 兜底：若入口失效，退回 `vllm serve` + `PYTHONPATH` 注入垫片（等价） |

> 兜底命令（不依赖我们自己的入口）：
> ```bash
> PYTHONPATH="$WORK_DIR/envs/vllm_shim:${PYTHONPATH:-}" \
> ROBO_VLLM_TIE_SHIM=1 ROBO_MODEL_PATH="$MODEL_PATH" \
> vllm serve "$MODEL_PATH" --served-model-name robobrain2.5-8b-nv \
>   --host 127.0.0.1 --port 30000 --tensor-parallel-size 1 --pipeline-parallel-size 1 \
>   --max-model-len 32768 --gpu-memory-utilization 0.85 --dtype bfloat16 --trust-remote-code \
>   --enforce-eager --no-enable-prefix-caching --max-num-batched-tokens 4096 \
>   --max-num-seqs 64 --limit-mm-per-prompt={"image":8}
> ```

---

## 4. 性能基线政策（重要，别踩红线）

赛事口径：**首次性能基准只能取一次**。

- 已在**本实例（zhr5535 镜像 + vllm 0.15.1 + torch 2.9.0）**取到的 12 轮官方格式压测数据（`submission/bench/round{1,2}.jsonl`、`summary.md`）是**本件 vLLM 计件的性能基准，保留不变**。
- 新实例（torch 2.10 / vllm 0.18.1）跑出来的数字属于**另一个配置**：
  - 只能作为**补充对照**写进报告（"换栈后的表现"），**不得**替换上面的基准；
  - 不得回头重跑本实例的六档压测去"换一个更好看的数"；
  - 若新实例的数字更好，结论只能是"新栈更有潜力"，不能改写已报出的基线。
- 同理，SGLang 在新实例上跑出的性能/精度属于**框架 × 卡型 × 镜像**的新计件，与 vLLM 件分开报。

---

## 5. 新实例验证顺序（每步都有"通过/不通过"判据）

| 步 | 动作 | 通过判据 | 不通过怎么办 |
| --- | --- | --- | --- |
| 1 | `bootstrap_vllm_adaptation.sh`（体检） | 环境指纹齐全、模型只读校验通过、垫片判定明确 | 缺件就从本地 `agent_work/adaptation/` 同步；模型不可读则换节点/卡 |
| 2 | `--install` | `envs/pylibs` 里出现 `qwen_vl_utils` | 离线则看镜像是否自带 |
| 3 | `--start` | `/health` 返回 200 | 看 `server.log`：OOM → 确认 `--enforce-eager` 在；参数报错 → 只改参数 |
| 4 | 最小推理（文本 + **图像**） | 有与任务相关的有效输出 | 调 `robo_infer.py` 的输入构造 |
| 5 | 6 样例探针 | 与已保存的 transformers 直跑基线**逐坐标**比对，差异可解释 | 差异超阈值先查客户端图像规范化是否生效 |
| 6 | SGLang 起服务 | `sglang.launch_server` 不再命中 `paged_attention_ragged` / aiter import ValueError | 换 `--attention-backend`（triton/flashinfer/wave 依次）；仍不行 → 记 `NOT_ACHIEVABLE` |
| 7 | SGLang 最小推理 + 图像 | 有有效输出 | 同上 |
| 8 | 官方材料到位后 | `python submission/verify.py` → `status=passed` | 材料未到位则维持 `BLOCKED`（不伪造） |

---

## 6. 风险与回退

| 风险 | 概率 | 影响 | 回退 |
| --- | --- | --- | --- |
| 新镜像里 SGLang 仍起不来 | 中 | SGLang 计件落空 | 本件按 vLLM 单件提交（计件粒度 = 模型 × 卡型 × 框架，SGLang 不计件不扣分） |
| vllm 0.18.1 参数/接口变动 | 中 | vLLM 件需返工 | 用 §3.1 的兜底命令；必要时退回 `vllm serve` + 垫片 |
| 新实例卡时不足 | 中 | 验收做不完 | 优先做精度项与 SGLang 冒烟，性能压测可延后（官方口径允许补充说明，但不得无数据） |
| 换实例丢产物 | 低 | 返工 | 产物都在 `/root/private_data/workspace/robo`；换实例前先 `ls` 校验 |
| 官方精度材料一直不下发 | 高 | **整件卡在 Gate** | 维持 `BLOCKED` 并在报告写明阻塞原因与已完成的机制演练（`tools/verify_path_dryrun.sh`） |

**回退路径**：若新镜像整体不可用，**本实例的镜像仍可重建**（镜像名 `jupyter-sglang5.12-ubuntu22.04-dtk26.04-py3.10-qwen3.6:v1.0` 已知可用，vLLM 全套已跑通），按 `reports/00_RUNBOOK.md` §2 的恢复流程重建即可 —— **但这会丢掉新镜像带来的 SGLang 机会**。

---

## 7. 卡时与工作量估算（给决策参考）

| 工作项 | 预估 |
| --- | --- |
| 新实例开起来 + 体检 + 依赖 | 0.5 h（不含镜像创建/调度等待） |
| vLLM 在新栈上跑通（启动 + 最小推理 + 探针） | 0.5–1 h（若参数要改则更多） |
| SGLang 冒烟（起服务 + 最小推理 + 1 次图像） | 0.5–2 h（**不确定性最高**） |
| 六档 × 2 轮压测（若要补测新栈） | ≈ 4 h（**建议不补，或作为对照只跑 2 档**） |
| 精度项（材料到位后） | ≈ 0.5 h |
| 报告回填 | 0.5 h |

**建议优先级**：精度项 > SGLang 冒烟 > vLLM 新栈复验 > 新栈压测对照。

---

## 8. 换实例前的检查清单

- [ ] 本地 `agent_work/adaptation/` 已与远端 `reports/ samples/ tools/ envs/ patches/` 同步（`tools/sync_from_remote.ps1`）
- [ ] 远端 `/root/private_data/workspace/robo` 四个关键产物存在：`submission/bench/round1.jsonl`、`round2.jsonl`、`submission/verify_result.json`、`envs/vllm_shim/sitecustomize.py`
- [ ] 平台侧：用**新镜像**创建实例，确认分配到卡、显存可见（`hy-smi`）
- [ ] 新实例内第一条命令：`bash /root/private_data/workspace/robo/tools/bootstrap_vllm_adaptation.sh`
- [ ] 新实例内第二条：`bash .../bootstrap_vllm_adaptation.sh --install --probe`
- [ ] 更新 `submission/env.sh` 的 `IMAGE` 字段为新镜像名（保证报告版本号与 `env.log` 一致）
- [ ] 新栈的 `env.log` 重跑并归档（`submission/env_check.sh`）

---

## 9. 关联文档

| 文档 | 内容 |
| --- | --- |
| `reports/13_scnet_image_registry_survey.md` | 镜像库 22 条清单、Harbor 读镜像方法、各镜像构建栈 |
| `reports/09_vllm_adaptation.md` | vLLM 四个阻塞与解决（要保留的适配成果本体） |
| `reports/06_sglang_compatibility.md` | 本节点 SGLang 不可用的根因（为什么换镜像） |
| `reports/00_RUNBOOK.md` §2 | Pod 重启/换实例后的恢复流程 |
| `reports/08_gap_vs_official_baseline.md` | 精度缺口与官方材料清单 |
| `samples/scnet_image_inventory.json` | 机器可读镜像清单 |
| `tools/bootstrap_vllm_adaptation.sh` | 一键重放脚本 |
