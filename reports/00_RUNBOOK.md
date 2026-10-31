# 运行手册与故障恢复指南（RoboBrain2.5-8B-NV / K100AI）

> 工作目录（远端，持久盘）：`/root/private_data/workspace/robo`
> 模型目录（远端，只读）：`/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV`
> 节点：单卡 K100AI（`K500SM_AI` / `gfx928` / 63.98 GiB），DTK 26.04，torch 2.9.0，transformers 5.3.0
> 连接：`ssh dcu_1026`（别名：HostName ksai.scnet.cn / Port 10312 / User root）

---

## 0. 一分钟版

```bash
ssh dcu_1026
cd /root/private_data/workspace/robo
source ./env.sh                       # 内部已 source /opt/dtk/env.sh 与 /opt/hyhal/env.sh
python3 scripts/minimal_inference.py \
  --image samples/images/official_affordance.jpg \
  --prompt "What is shown in this image?" \
  --max-new-tokens 64 --repeat 2 --out-dir outputs/phase2
```

预期：`VERDICT PASS`，两次输出逐字符相同，峰值显存 ≈ 16.53 GiB。

---

## 1. 环境自检（每次新会话先跑这 5 条）

```bash
cd /root/private_data/workspace/robo
source ./env.sh
hostname                                    # 期望 crdnotebook-...-scnjz2s4uc-68757
/opt/hyhal/bin/hy-smi                      # 期望 HCU[0] 1 张，VRAM 0%，Normal
python3 -c "import torch; p=torch.cuda.get_device_properties(0); \
print(p.name, p.gcnArchName, p.major, p.minor, round(p.total_memory/1024**3,2))"
# 期望: K500SM_AI gfx928:sramecc+:xnack- 9 2 63.98
python3 -c "import qwen_vl_utils, transformers, torch; \
print('qwen_vl_utils', qwen_vl_utils.__version__, '| transformers', transformers.__version__, '| torch', torch.__version__)"
```

> **坑**：`torch` 直接 import 会报 `ImportError: libgalaxyhip.so.5` —— 必须先 `source ./env.sh`
> （它内部 `source /opt/dtk/env.sh`）。`hy-smi` 不在默认 PATH，要用 `/opt/hyhal/bin/hy-smi`。

### 1.1 容器联网（2026-10-27 更正：此前"节点无外网"的结论是错的）

平台在**家目录**自动生成出口代理配置。联网指南写的 `~/.ai_user_info/ai_proxy` 在本容器**不存在**
（root 的 `$HOME` 是 `/root`，而家目录挂载点是 `/public/home/scnjz2s4uc`）。实际路径：

```
/root/private_data/.ai_user_info/ai_proxy        # = /public/home/scnjz2s4uc/.ai_user_info/ai_proxy
```

已固化为 `tools/env_net.sh`（一次 source 同时得到 DCU 运行时与平台代理）：

```bash
source /root/private_data/workspace/robo/tools/env_net.sh
# 验证
env | grep -iE '^(http|https|ftp|no)_proxy' | cut -d= -f1     # 应有 4 个键名
curl -sS -m 20 -o /dev/null -w '%{http_code}\n' https://www.scnet.cn/
python3 -c "import torch;print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
```

实测（2026-10-27 source 后）：`www.scnet.cn` 301、`gitlab.scnet.cn:9002` 302、
`developer.sourcefind.cn` 200、`huggingface.co` 200、`pypi.tuna.tsinghua.edu.cn` 200；
`torch 2.9.0 | hip 6.3.26113 | cuda_avail True`。

> **凭据纪律**：代理文件含明文口令，**只 source 不打印**，不写进任何脚本、日志、报告或提交物。
>
> **本次变更的影响**：`bench.sh` 里"节点无外网"导致的那条 workaround **仍然需要**——
> 官方压测客户端用 `--served-model-name`（服务名，不是路径）去 `AutoProcessor.from_pretrained()`，
> 而服务名不是 HF 上的真实 repo，联网也解析不出来，仍需在 `$HF_HOME/hub/` 建只读软链接指向本地权重。

---

## 2. Pod 重启 / SSH 断连后的恢复流程

超算互联网的 notebook 实例重启时，**`/root/private_data/` 是持久盘，工作目录与 `envs/` 里的离线依赖都会保留**；
`/public` 模型目录只读共享也会保留。丢失的只有运行中的进程和 `cache/`、`tmp/` 里的临时物。

按顺序执行，每步都有判定：

```bash
# 步骤 1：确认工作目录还在
cd /root/private_data/workspace/robo && ls -1
# 判定：应能看到 scripts/ configs/ samples/ logs/ reports/ outputs/ envs/ third_party/ patches/

# 步骤 2：确认产物没丢（重点）
ls -1 outputs/
# 判定：应能看到 phase2/ phase3_baseline/ phase4/ phase4_batch/ tasks_official/ batch_bs4/
#        若 outputs/ 为空 → 本地 agent_work/adaptation/ 里的报告仍可用，但原始 JSON 已丢失，
#        需重跑第 3 节基线命令重新生成（数据可复现，因为样例与提示词都是固定的）。

# 步骤 3：重载运行时
source ./env.sh

# 步骤 4：GPU 与依赖自检
/opt/hyhal/bin/hy-smi
python3 -c "import qwen_vl_utils; print('deps ok')"

# 步骤 5：清理可能残留的推理进程（只清本任务的）
ps -eo pid,etime,cmd | grep -E "serve_vllm|serve_sglang|scripts/" | grep -v grep
# 若有残留：kill -TERM <pid>，确认 hy-smi 显示 VRAM 0% 后再继续
```

---

## 3. 完整复跑顺序（从零到全部产物）

```bash
cd /root/private_data/workspace/robo
source ./env.sh

# ① 第二阶段验收：最小推理跑通 + 两次一致
python3 scripts/minimal_inference.py \
  --image samples/images/official_affordance.jpg \
  --prompt "What is shown in this image?" \
  --max-new-tokens 64 --repeat 2 --out-dir outputs/phase2

# ② 功能验证：四类任务 + 标注图（人工目视校验用）
python3 scripts/run_tasks.py --tasks samples/tasks_official.json \
  --image-dir samples/images --out-dir outputs/tasks_official

# ③ 第三阶段基线：6 样例 × 3 次，稳态 n=12
python3 scripts/benchmark.py --tasks samples/tasks_official.json \
  --image-dir samples/images --repeat 3 --out-dir outputs/phase3_baseline

# ④ 第四阶段单变量实验（每次只改一个变量）
python3 scripts/tune.py --tag attn_fa2 --attn-impl flash_attention_2 --repeat 3 --out-dir outputs/phase4
python3 scripts/tune.py --tag pix_501760 --max-pixels 501760 --repeat 3 --out-dir outputs/phase4
python3 scripts/tune.py --tag pix_200704 --max-pixels 200704 --repeat 3 --out-dir outputs/phase4

# ⑤ 批处理扫描（吞吐）
python3 scripts/benchmark_batch.py --batch-sizes 1 2 4 8 --repeat 3 --out-dir outputs/phase4_batch

# ⑥ 对比总表
python3 scripts/compare_experiments.py --baseline outputs/phase3_baseline/baseline_raw.json \
  --dirs outputs/phase4 --out outputs/phase4/comparison.md

# ⑦ vLLM 服务化（官方认可的指定框架之一）
#    参数与最终提交配置逐字一致（见 submission/bench/run_config.txt）
python3 scripts/serve_vllm_patched.py \
  --model "$MODEL_PATH" --served-model-name robobrain2.5-8b-nv \
  --host 127.0.0.1 --port 30000 \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 --dtype bfloat16 \
  --trust-remote-code --gpu-memory-utilization 0.85 --max-model-len 32768 \
  --enforce-eager --no-enable-prefix-caching \
  --max-num-batched-tokens 4096 --max-num-seqs 64 \
  --limit-mm-per-prompt '{"image": 8}'          # 需另开终端
# 另开终端：探针 + 与基线逐坐标比对
python3 scripts/probe_http_request.py --backend vllm --port 30000 \
  --tasks samples/tasks_official.json --image-dir samples/images \
  --repeat 2 --out-dir outputs/probe_vllm
python3 scripts/compare_backends.py \
  --baseline outputs/phase3_baseline/baseline_raw.json \
  --candidate outputs/probe_vllm/probe_result.json \
  --out outputs/probe_vllm/agreement.md

# ⑧ 阶段 5：一键完成「起服务 → 规范化 A/B 探针 → 服务冒烟 → 六档压测」
#    注意：压测会占很久，建议按档拆分执行（见 §3.1）
bash scripts/run_phase5.sh --serve-only --keep-server --mem-frac 0.85 --ctx 32768
python3 scripts/smoke_service.py --url http://127.0.0.1:8000 \
  --model robobrain2.5-8b-nv --out-dir outputs/phase5_smoke
python3 scripts/bench_service.py --url http://127.0.0.1:8000/v1/chat/completions \
  --model robobrain2.5-8b-nv --rounds 2 --out-dir outputs/phase5_bench
```

### 3.1 阶段 5 压测按档拆分（单次任务 1 小时上限）

六档全跑需要 2.5～4 小时，超过单次后台任务的时限，按下面拆开执行，
每档单独 `--out-dir`，最后合并：

```bash
python3 scripts/bench_service.py ... --rounds 2 \
  --tiers "4096:1024:1:8,4096:1024:8:16" --out-dir outputs/phase5_bench/tier4096_a
python3 scripts/bench_service.py ... --rounds 2 \
  --tiers "4096:1024:64:128"          --out-dir outputs/phase5_bench/tier4096_b
python3 scripts/bench_service.py ... --rounds 2 \
  --tiers "16384:1024:1:8"            --out-dir outputs/phase5_bench/tier16384_a
python3 scripts/bench_service.py ... --rounds 2 \
  --tiers "16384:1024:8:16"           --out-dir outputs/phase5_bench/tier16384_b
python3 scripts/bench_service.py ... --rounds 1 \
  --tiers "16384:1024:64:128"         --out-dir outputs/phase5_bench/tier16384_c_r1
python3 scripts/bench_service.py ... --rounds 1 \
  --tiers "16384:1024:64:128"         --out-dir outputs/phase5_bench/tier16384_c_r2
```

> `bench_service.py` 每跑完一轮就把 `bench_results.json` / `bench_records.json` 落盘，
> 即使任务超时，已完成的档位/轮次也不会丢。

### 3.2 提交包路线（当前推荐，与 §3 的历史序列等价但更短）

```bash
cd /root/private_data/workspace/robo/submission

# 一键：run.sh → env_check.sh → smoke.sh → verify.py → 官方六档压测
bash reproduce.sh
#   verify.py 退出码 2 = 精度项 BLOCKED（官方基线未下发，预期结果）

# 单独跑官方格式压测（6 档 × 2 轮，单卡约 4 小时）
BENCH_MODE=image bash bench.sh

# 提交包自检（对照官方模板逐项校验）
bash ../tools/check_submission.sh

# 干净环境复现（全新起服务 → 环境采集 → Smoke → 精度 → 性能复跑）
bash ../tools/clean_repro.sh
```

### 3.3 收尾流水线（长任务自动衔接，无需盯屏）

```bash
# 第一阶段：等六档压测结束 → 汇总 → 回填 → 刷新产物 → 自检 → E1/E2/E3 → 恢复基线服务
setsid nohup bash tools/pipeline_after_bench.sh > logs/pipeline_after_bench.log 2>&1 &
# 第二阶段：计量口径对照实验 → 回填报告 10/11/05 → 刷新 → 最终检查
setsid nohup bash tools/pipeline_final.sh > logs/pipeline_final.log 2>&1 &
# 干净复现
setsid nohup bash tools/clean_repro.sh > logs/clean_repro_stdout.log 2>&1 &
# 内存安全网（85 GiB 阈值，先于 kernel OOM killer 动手）
setsid nohup bash tools/guard_cgroup.sh 85 > /dev/null 2>&1 &
```

### 3.4 公开基准评测（精度优先，2026-10-27 新增）

三个**公开上游评测集**（CV-Bench / RefSpatial-Bench / TraceSpatial-Bench）的
协议冻结见 `reports/15_public_bench_protocol.md`，结果见
`reports/16_public_bench_results.md`。

```bash
cd /root/private_data/workspace/robo

# ① 依赖：open3d（TraceSpatial 3D 判分硬依赖）+ numpy 隔离前缀
bash tools/install_official_deps.sh
bash tools/prepare_core_prefix.sh     # 通用依赖 -> envs/pylibs_core（无 numpy2）
bash tools/place_open3d.sh             # open3d + numpy2 -> envs/pylibs_o3d
# ⚠️ open3d 0.19/0.20 的 ScalableTSDFVolume 在官方脚本那组参数下抽不出点，
#    会让官方 3D 碰撞检查恒为空转（No-Collision 恒 1.0）。必须用 0.18.0：
bash tools/install_o3d_018b.sh         # 解包 open3d 0.18.0 + numpy 1.26.4 -> envs/pylibs_o3d_v2
bash tools/install_o3d_018c.sh         # 补 sklearn 桩件（open3d.ml 的 import 期依赖）
export ROBO_O3D_PREFIX=$PWD/envs/pylibs_o3d_v2

# ② 起服务（eager = 首次提交基线配置；graph = E1 采纳后的配置）
bash tools/start_bench_service.sh eager 30000

# ③ 数据完整性 manifest（SHA256 / 条目核对 / 全量解码检查）
python3 submission/bench_public/manifest.py --out-dir outputs/bench_public/manifest

# ④ 三个基准 + 稳定性检查 + 离线重判分
bash tools/run_accuracy_first.sh        # RefSpatial(4变体×3split) + CV-Bench(2638)
bash tools/finish_accuracy.sh           # 重判RefSpatial + TraceSpatial + 稳定性 + 重判分

# ⑤ 候选配置复测（E1 非 eager）：唯一变量 = 去掉 --enforce-eager
#    脚本内部会停掉 eager 服务 -> 起 graph 服务 -> 全量重跑三基准 -> 候选侧离线重判分 -> 对比
bash tools/run_accuracy_candidate.sh
#    串行衔接（基线跑完后自动接候选，避免两条链抢 CPU / 抢端口）：
#    setsid nohup bash tools/chain_candidate.sh <finish_accuracy_pid> > logs/chain_candidate.log 2>&1 &

# ⑥ TraceSpatial 官方判分（2D + 3D，4 个组合共用一个线程池；官方代码零改动）
bash tools/run_ts_score_2d.sh           # 2D（不依赖 open3d）
bash tools/run_ts_score_3d.sh 16 45     # 3D（并发 16，内存闸门 45GiB）
python3 submission/bench_public/merge_ts_scores.py --bench-root outputs/bench_public
python3 submission/bench_public/merge_ts_scores.py --bench-root outputs/bench_public_noneager

# ⑦ 分片合并精确性验证（整集单进程官方跑 vs 分片合并，逐指标对拍）
bash tools/verify_3d_merge.sh

# ⑧ CV-Bench 图像输入协议因子实验（8 组，只跑 Count 子任务）
bash tools/exp_cvbench_protocol_remaining.sh

# ⑨ RefSpatial 噪声底（全量同配置重跑一次）
bash tools/measure_refspatial_noise.sh

# ⑩ 汇总表（验收要求 §8 的表格格式）
python3 submission/bench_public/build_summary_table.py \
  --bench-root outputs/bench_public --out outputs/bench_public/summary_table.md

# ⑪ 精度回归对比表（参考=eager基线 / 候选=非eager）
python3 submission/bench_public/compare_accuracy_runs.py \
  --reference outputs/bench_public --candidate outputs/bench_public_noneager \
  --out outputs/bench_public/accuracy_regression.md

# ⑫ CV-Bench 上采样协议实验（up2/up4 × png/jpeg95 + 同会话 canonical 对照）
bash tools/exp_cvbench_upscale.sh

# ⑬ CV-Bench 提示词口径实验（数据集自带 prompt 字段 vs FlagEvalMM build_prompt）
bash tools/exp_cvbench_prompt_style.sh

# ⑭ Count 缺口归因（误差分布 + 失分样例导出，供人工目视核对）
python3 tools/diag_cvbench_count.py
python3 tools/diag_cvbench_count2.py
python3 tools/diag_cvbench_protocol_detail.py
python3 tools/diag_cvbench_imgsize.py
python3 tools/export_count_samples.py

# ⑮ 分片合并口径复核（改了 merge 逻辑后必须跑；差异不允许静默通过）
python3 tools/remerge_ts_3d.py --bench-root outputs/bench_public
```

> ⚠️ **耗时提示（2026-10-28 修正）**：TraceSpatial 的 3D 官方几何判分是**单核**的。
> 修复 open3d 之后实测**单样本 ~7~30 s**（16 路并发时），整集 100 条单进程 ≈ 12 分钟；
> 用 `tools/run_ts_score_3d.sh`（16 并发 + cgroup 内存闸门）跑完
> **4 个组合 × 100 题 = 400 个分片约 30 分钟**。
> 早先"80 条合计 9 秒、剩 2 块跑几十分钟、单进程全量 80 分钟"的观测，
> 是在 **open3d 失效**（TSDF 直接抛异常）的情况下测的，**已作废**。
>
> ⚠️ **官方判分代码零改动**：所有 TraceSpatial 数字都来自**未经修改**的
> `third_party/upstream_eval/RoboTracer/Evaluation/summarize_acc.py`，
> 我们只改变「喂给它的样本分组」与「并发调度」；
> 分片合并是否等价于整集单进程，由 `tools/verify_3d_merge.sh` 逐指标对拍证明。

**踩过的坑（都已修，见 §4）**：

* `envs/pylibs` 里装 open3d 会连带装 **numpy 2.x**，遮蔽系统 numpy 1.25 →
  torch 导入失败 → 必须拆前缀（`pylibs_core` 通用 / `pylibs_o3d` open3d）。
* **open3d 0.19/0.20 的 `ScalableTSDFVolume` 改了体素语义**（新增
  `volume_unit_resolution`/`depth_sampling_stride`），官方脚本给的
  `voxel_length=0.02, sdf_trunc=0.08` 在新语义下 SDF 带比体素还窄 →
  **TSDF 恒抽不出点** → 官方 3D 碰撞检查整体空转。必须用 **0.18.0**（见 §5.7）。
* FlagEvalMM 的 `flagevalmm/evaluator/__init__.py` 会拖进 `accelerate`，而本机
  `accelerate` 与 numpy 1.25 不兼容 → 官方 `tasks/cv_bench/evaluate.py` 直接
  import 就炸。解法见 `submission/bench_public/official_flagevalmm.py`
  （轻量 stub 包跳过框架重 `__init__`，**判分代码原文照用**）。
* `nohup` 起服务挡不住**进程组 SIGTERM**，会话结束会把服务带走 → 必须用
  `setsid`（`tools/start_bench_service.sh` 已改）。
* 官方 stdout 的**行首冒号 / 阈值括注 / 单位后缀**会让朴素的字符串截断解析静默失败
  （指标变成 `None` 再合并成全 0）→ 见 §5.8。

---

## 4. 故障排查表（均为本节点实测）

| 症状 | 根因 | 处置 |
|---|---|---|
| `ImportError: libgalaxyhip.so.5` | 未加载 DTK 运行时 | `source ./env.sh`（内部已 source `/opt/dtk/env.sh`） |
| `hy-smi: command not found` | 不在默认 PATH | 用 `/opt/hyhal/bin/hy-smi` |
| SSH `banner exchange` 超时 / `Connection closed by 223.113.240.64` | 平台端口转发未就绪 / 实例重启 | 退避重试；查控制台实例状态；**端口可能已变**，改本机 `~/.ssh/config` 的 `Port` |
| sglang `NameError: name 'paged_attention_ragged' is not defined` | 镜像内 `aiter` 残缺构建；`aiter_backend.py:36-50` 的 `except ImportError` 只 print 不 re-raise，13 个符号全未定义 | 镜像内**无解**（env/flag 改不动）；需离线引入完整匹配的 `aiter` wheel |
| sglang `--attention-backend flashinfer` | `flashinfer_python >= 0.6.11.post1` 未安装 | 需离线引入 |
| sglang `--attention-backend wave` | `wave_lang` 未安装 | 需离线引入 |
| sglang `--attention-backend triton` | `Invalid address access ... Error code: 3` → `Rank 0 scheduler died during initialization (exit code: -6)` | HIP 非法访存发生在 scheduler 初始化阶段；换后端 |
| sglang `--attention-backend torch_native`（±sdpa / ±`--model-impl transformers`） | 服务能起（Uvicorn running），请求时 `HSA_STATUS_ERROR_EXCEPTION 0x1016` / `KERNEL VMFault`，崩在 `transformers/integrations/sdpa_attention.py:92` → `/pytorch/aten/src/ATen/native/transformers/hip/cutlassfa_adapter.h:148` | torch 侧 HIP SDPA 内核问题，本任务范围外；换后端 |
| vLLM `AttributeError: 'Qwen3VLTextConfig' object has no attribute 'tie_word_embeddings'` | vLLM 0.15.1 从**子 config** 取该字段，transformers 5.3.0 把它挪到了**顶层复合 config** | 用 `scripts/serve_vllm_patched.py`（sitecustomize 垫片，**不改 site-packages**）；等价 patch 见 `patches/0001-vllm-qwen3-tie-word-embeddings.patch` |
| 垫片在父进程生效但子进程仍报同一错 | vLLM 用 `VLLM_WORKER_MULTIPROC_METHOD=spawn` 起 EngineCore，monkey-patch 不被继承 | 垫片必须放 `sitecustomize.py`（每个解释器启动时执行） |
| 批处理（bs>1）输出被污染（出现 `Human:`、长串 `0`） | decoder-only 批处理用 right padding | `processor.tokenizer.padding_side = "left"`（`scripts/batch_infer.py` 已处理） |
| grounding bbox 明显变形 | `max_pixels` 降到 200704 使视觉 token 减少 42.8%，坐标改变约 48% | 回到 `501760`（已采纳配置） |
| `sglang.launch_server` 作为可执行文件 → `No such file or directory` | 本版本入口是 `sglang serve` 或 `python3 -m sglang.launch_server` | 二选一 |
| `--limit-mm-per-prompt`（sglang）不存在 | 本版本参数名为 `--limit-mm-data-per-request` | 改名 |
| `--enable-prefix-caching`（sglang）不存在 | radix cache 默认开，关掉用 `--disable-radix-cache` | 改参数 |
| **serving 的 grounding 输出不可解析（`<bbox>x y x y</bbox>`）** | 官方正则只认 `[x1, y1, x2, y2]`；vLLM 在"分析模式"下输出尖括号形式 | 用 `scripts/answer_adapter.py` 做纯格式桥（整数原样搬运，原始文本单独保留） |
| **serving 的 grounding 输出与直跑完全不同（整个回答风格都变了）** | 官方 upstream 用 `qwen_vl_utils` 取图 → **二次重采样**（先按 28 倍数 PIL 缩放，processor 再按 32 倍数缩放）；直接上传原图只经历一次缩放，像素最大差 **0.47**，翻转了首个生成 token（`The`/`To` 在 bf16 下 logit 相等 24.5） | 客户端用 `scripts/image_canonical.py` 规范化后再以 PNG 上传（`probe_http_request.py` 默认已启用，`--no-canonical` 可关闭做 A/B） |
| **压测数字偏乐观（TTFT 异常低）** | vLLM 默认开启 prefix caching，相同 prompt 的 prefill 直接命中缓存（本轮实测命中率 **64%**） | 加 `--no-enable-prefix-caching`，并让每个请求使用不同 filler 文本（token 数相同） |
| `vllm serve` 启动时被 OOM 杀掉、Pod 重建 | 非 eager 模式下 inductor 编译期内存峰值触碰 115 GiB cgroup 上限 | **首跑一律 `--enforce-eager`**；要试非 eager 必须带内存看门狗，见 `scripts/run_non_eager_ab.sh` |
| **16384 档请求被服务拒收（400）** | `--max-model-len 20480` 装不下「随机文本解码后重编码 ~17.3K + 视觉 2,042 + 输出 1,024 ≈ 20.3K」 | `MAX_LEN` 改回官方 `env.sh` 的 **32768**；模型 `max_position_embeddings=262144` 未改。实测该改动**不改变 KV cache 容量**（均 253,312 token） |
| **官方压测吞吐虚高约 4×，但 `total_output_tokens` 显示 1024×N** | 客户端 `output_lens.append(output_len)` 统计的是**计划长度**；且 `--disable-ignore-eos` 实为**允许提前 EOS**（`payload["ignore_eos"] = not args.disable_ignore_eos`），官方脚本不传该开关即 `ignore_eos=True` | `bench.sh` **删掉**该开关，回到官方默认；核对 `total_output_tokens_retokenized`（真实输出长度）而非只看 `total_output_tokens` |
| 官方客户端报 `Image dataset only supports backends: [...]` | `sglang/benchmark/datasets/image.py` 对后端做硬白名单，`--backend vllm` 直接抛错，一个请求都发不出 | 用 `scripts/bench_serving_vllm_image.py` 在**进程内**把数据集构造的后端映射为 `sglang-oai-chat`（服务端套 chat template，语义一致），不改公共 site-packages |
| 官方客户端 image 数据集加载 processor 失败（节点无外网） | `AutoProcessor.from_pretrained(model_id)`，而 `model_id` 取 `--served-model-name`（服务名非路径） | 在**工作目录内** HF 缓存建只读软链接 `$HF_HOME/hub/models--<服务名>/snapshots/main -> $MODEL_PATH` |

---

## 5.5 vLLM 启动期内存风险（2026-10-27 事件记录与规避）

**事件**：2026-10-27 01:02:34 启动 vLLM（try3，第一个越过 config 修复、进入
`torch.compile` / CUDA graph capture 阶段的版本），约 01:07 SSH 通道即告中断
（TCP 可达但网关在 SSH 握手前关闭），持续 49 分钟未恢复。
该阶段是 vLLM 启动期内存占用最高的环节（inductor 编译 + 权重加载 + graph 捕获），
而 notebook 配额约为 **6 核 / 115 GB**，**容器被 OOM killer 杀掉是合理解释**。

**恢复后必须先做的两件事**：

```bash
# ① 确认是否 OOM
dmesg -T 2>/dev/null | grep -iE "oom|killed process" | tail -20
cat /sys/fs/cgroup/memory.max 2>/dev/null; cat /sys/fs/cgroup/memory.peak 2>/dev/null
# ② 复现时降低启动期内存峰值（一次只加一个）
```

**降内存启动参数（按风险从低到高）**：

| 措施 | 命令/环境变量 | 作用 | 代价 |
|---|---|---|---|
| 跳过编译与 graph 捕获 | `--enforce-eager` | 免掉 inductor 编译，峰值大幅下降 | 吞吐下降（需实测） |
| 降低 KV 池占比 | `--gpu-memory-utilization 0.70` | 给编译期留余量 | KV 容量变小 |
| 限制编译并发 | `VLLM_COMPILE_WORKER_COUNT=1` | 减少编译子进程内存 | 编译变慢 |
| 减小 batched tokens | `--max-num-batched-tokens 4096` | 降低 profiling 峰值 | 吞吐下降 |
| 关前缀缓存 | `--no-enable-prefix-caching` | 省一点 KV | 重复前缀变慢 |
| **加内存看门狗（强烈建议）** | `bash scripts/run_non_eager_ab.sh --mem-limit-gib 70` | 每 3 s 读 cgroup 用量，超阈值先杀进程，**避免 kernel OOM killer 杀掉容器导致 Pod 重建** | 非 eager 起不来时只能记为"本容器装不下" |

> **看门狗的必要性**：历史上正是非 eager 启动把容器内存顶到 115 GiB 上限，
> OOM killer 杀掉容器 → Pod 重建 → SSH 侧车失效 → 5 小时不可达。
> 在同一台机器上复现这个实验时，**必须**先用看门狗版本，不要裸跑。
> 被看门狗中止的实验是**有效结论**（"非 eager 在 115 GiB 配额下装不下"），不是脚本失败。

> 纪律不变：**一次只改一个变量**，改完用第 3 节 ③ 的基线命令复测，
> 用 `outputs/phase3_baseline/baseline_raw.json` 的 TTFT / 吞吐做同口径对比。
> 若 `--enforce-eager` 让首提性能基线明显变差，则该参数**不能**进首提配置——
> 应改为「先单独跑一次 eager 确认能起，再用非 eager 跑首提基线」。

---

## 5.6 换实例 / 换镜像的一键重放（2026-10-27 新增）

适用场景：Pod 被重建、或按决定**换用官方 sglang 镜像**（torch 2.10 / vllm 0.18.1 栈）的新实例。

```bash
# ① 体检（工作目录完整性 / 环境指纹 / 模型只读校验 / 垫片是否需要）
bash /root/private_data/workspace/robo/tools/bootstrap_vllm_adaptation.sh

# ② 补齐隔离依赖 + 拉起服务 + 跑 6 样例探针
bash /root/private_data/workspace/robo/tools/bootstrap_vllm_adaptation.sh --install --probe
```

脚本原则与红线一致：不写公共 site-packages（垫片走 `sitecustomize` + `PYTHONPATH` 注入）、
新增依赖只装进 `$WORK_DIR/envs/pylibs`（`--no-deps --target`）、模型目录只读、
`--enforce-eager` 等已验证启动参数一个不改。

**换镜像前务必确认**（产物都在持久卷 `/root/private_data/workspace/robo`，但仍要查一眼）：

```bash
cd /root/private_data/workspace/robo
ls -l submission/bench/round1.jsonl submission/bench/round2.jsonl \
      submission/verify_result.json envs/vllm_shim/sitecustomize.py
```

完整方案与镜像栈证据见 `reports/14_new_instance_migration_plan.md`、
`reports/13_scnet_image_registry_survey.md`。

---

## 5.7 官方 3D 判分的碰撞检查空转（2026-10-28 定位并修复）

**症状**：官方 `summarize_acc.py` 对每条样本都打 `[Warn][3D] Collision check failed`，
`ValueError: TSDF extracted no points`；结果是 `No-Collision Rate` **恒为 1.0**、
`Overall Success` 偏乐观。

**根因**：`open3d` **0.19 起改写了 `ScalableTSDFVolume` 的体素语义**，
新增 `volume_unit_resolution`（默认 16）与 `depth_sampling_stride`（默认 4）。
官方脚本给的 `voxel_length=0.02, sdf_trunc=0.08` 在新语义下
**SDF 带（±0.08 m）比体素还窄**，没有任何体素能落进 SDF 带 →
`extract_point_cloud()` 恒返回空集。

**证据链**（全部实测，见 `reports/16_public_bench_results.md` §5.6）：

| 实验 | 结果 |
|---|---|
| 合成平面（正前方 1 m）过 TSDF | 0 点（与数据无关） |
| `volume_unit_resolution=64` 或 `sdf_trunc=1.28` | **2700 点** |
| 读 0.20 的构造签名 | `(voxel_length, sdf_trunc, color_type, volume_unit_resolution=16, depth_sampling_stride=4)` |
| 换 open3d 0.18.0（隔离前缀） | 合成平面 2700 点；真实样本 21036 点 / 13717 体素 |

**修复**：

```bash
bash tools/install_o3d_018b.sh   # 解包 open3d 0.18.0 + numpy 1.26.4 -> envs/pylibs_o3d_v2
bash tools/install_o3d_018c.sh   # 补 sklearn 桩件（open3d.ml 的 import 期依赖）
export ROBO_O3D_PREFIX=$PWD/envs/pylibs_o3d_v2
bash tools/verify_o3d018_tsdf.sh # 复验：官方那组参数能抽出点
```

> ⚠️ **官方 `summarize_acc.py` 一字未改**，只换环境（隔离前缀）+ 用
> `ROBO_O3D_PREFIX` 切换判分子进程的 `PYTHONPATH`。
> `envs/pylibs_o3d`（0.20）**保留作对照**，但不要用于 3D 判分。

---

## 5.8 官方 stdout 的指标抓取陷阱（2026-10-28 自查发现并修复）

**症状**：判分跑完了、日志里也有指标行，但我们自己汇总出来的数值**全是 0**
（`metrics` 全 0、三个 Success Rate 全 0）。如果没发现，会产出"看起来正常
但全是 0"的错误报告。

**根因**：官方打印有三种修饰，朴素的 `line[len(key):]` 截断 + `float()` 会静默失败：

| 官方输出 | 朴素截断后 | 结果 |
|---|---|---|
| `2D Path Metrics (Normalized [0,1]): {...}` | `": {...}"` | `literal_eval` 失败 → `None` |
| `Start Success Rate (<0.20m): 1.0000` | `"<0.20m): 1.0000"` | `float()` 失败 → `None` |
| `Start Point Distance (mean): 0.0748 m` | `": 0.0748 m"` | `float()` 失败 → `None` |

**修复**（`submission/bench_public/score_3d_chunked.py`）：

* `_grab`：截断后 `lstrip(":")`；
* `_grab_float`：取**最后一个冒号之后**的**首个数字**（同时覆盖阈值括注与单位后缀）。

**回归防线**：`tools/verify_3d_merge.sh` —— 把「整集单进程官方跑」与
「逐样本分片合并」的结果**逐指标对拍**。合并口径一旦再坏，对拍立刻报不一致。

---

## 5.9 分片合并口径与深景病态样本（2026-10-28 新增）

### 5.9.1 官方对「无效样本」是**静默丢弃**的 → 合并分母必须按空 dict 规则还原

官方 `calculate_metrics` 与 `compute_accuracy_2d/3d` 对以下样本**不打印任何标记**
就丢掉：

* 归一化后含 NaN、或图像尺寸无效的样本（2D 任务：`valid_pred_norm` 过滤）；
* `dfd > 100` 的样本（3D 任务，单位是米）。

唯一的可观测信号是：**有效样本数为 0 时官方打印空 dict `{}`**。
逐样本分片时一个分片恰好一个样本，于是口径可以被**精确还原**：

| 分片打印 | 官方含义 | 合并时的处理 |
|---|---|---|
| 非空 dict | 该样本进入指标计算且未被丢弃 | 计入分母，值即该样本值 |
| **空 dict `{}`** | 该样本被官方丢弃 | **该分片对指标零贡献，分母不加** |
| 全 0 的 dict | 有效样本被 `dfd>100` 全部丢弃 | 记为 `ambiguous`，不计入分母（审计计数） |

⚠️ 若把「空 dict」分片也计入分母，2D 路径指标会偏 **3.0%**
（实测 `dfd` 0.2055 → 0.1993，分母 97 被错记成 100）。
`tools/remerge_ts_3d.py` 可用已落盘的 `per_chunk` 重算并逐指标对比，
改动合并逻辑后**必须**跑一次，差异不允许静默通过。

⚠️ **`--chunk-size > 1` 时路径指标不再精确**：多样本分片下"只丢了一部分样本"
与"正常分片"在 stdout 上无法区分。因此**路径指标的权威口径始终是 `--chunk-size 1`**，
多样本分片只能用于速率类指标的交叉验证。

### 5.9.2 深景病态样本：单条判分 >45 分钟

open3d 修复后 TSDF **真正生效**，代价是深景样本要在大点云上做 DBSCAN：

* 100 条里 **4 条**（索引 56 / 78 / 82 / 98）单条判分 **>45 min**（16 并发下仍是），
  其余 96 条 15~30 s；
* 这 4 条在「整集单进程」下会被**串行化** → 墙钟不可接受；
* 因此 `tools/verify_3d_merge.sh` 的对拍取**前 N 条干净样本**（默认 N=50），
  阶段 A / 阶段 B 取**同一区间**逐指标对拍（合并代码对所有样本同路径）。

### 5.9.3 `score_3d_chunked.py` 的 work-dir tag 冲突

分片目录 tag 只取 `out_dir` 的 basename：

```python
tag = "%s_%s_%s" % (os.path.basename(os.path.normpath(j["out_dir"])), variant, task)
```

于是 `outputs/bench_public/tracespatial` 与 `outputs/bench_public_noneager/tracespatial`
会落到**同一个 tag 目录**，而 `official` 变体的 `model_id` 也相同
→ 同一个 `outputs/<model_id>_3D.jsonl` 被两次写入。

* 队列是按 job 顺序推进的，**同一 chunk 索引不会并发写**，所以已完成的分片不会被污染；
* 但**排查时必须知道**：`tmp/ts3d/tracespatial_official_3D/c0056/` 里的 jsonl
  可能是**候选配置**的预测，不是基线的。
* 新脚本一律**按 bench 根目录拆 work-dir**（`tmp/ts3d_verify_full` / `..._ne`）。

---

## 5.10 评测耗时分级：**先快后慢，慢的设上限**（2026-10-28 用户要求后固化）

**规则**：评测一律按「预估墙钟」分级执行，**先跑快的拿到结论**，
慢的评测必须**先设上限**（超时即停并记录为「已排除/未完成」），不允许无限期挂着。

| 级别 | 预估墙钟 | 本任务实例 | 处置 |
|---|---|---|---|
| L1 快 | < 5 min | 指标抓取自检、协议解析核对、误差分布统计、合并口径重算 | 随时跑 |
| L2 中 | 5~20 min | CV-Bench `Count` 子任务 788 题、TraceSpatial 3D 判分 96 条（16 并发，实测 941 s） | 主力档 |
| L3 慢 | 20~60 min | CV-Bench 全量 2638 题、4× 上采样协议 | **设上限**，超时即停 |
| L4 极慢 | > 60 min | 含深景大点云样本的官方 3D 判分（整集）、4× 上采样 | **不跑**或**排除病态样本后跑** |

**本轮的实际教训（2026-10-28）**：

* TraceSpatial 3D 生产判分在 **4 个深景病态样本**上卡了 **1.5 h**（已停止）；
  改用「排除这 4 条 + 96 条兜底口径」后 **941 s 跑完 4 个组合** ✅；
* `up4`（4× 上采样）单组 **>23 min** 未出结果（已停止），
  `up2` 已经给出 +2.9~3.3 pt 的结论，**边际收益不值得再等**；
* 判分器本身有单样本 O(病态) 的尾部，必须靠**排除**而不是**等待**；
* 提并发**不等于**更快：`up2` 协议下并发 16 → 48，吞吐从 1.12 req/s 降到 0.9 req/s
  （视觉负载变重后 prefill 侧已饱和）。**评测并发仍固定 16**。

**耗时台账（自动生成，不手工记账）**：

```bash
python3 tools/build_eval_timing.py     # -> outputs/bench_public/eval_timing.{md,json}
```

| 产物 | 内容 |
|---|---|
| `outputs/bench_public/eval_timing.md` | 已完成评测的墙钟表 + 主动中止的沉没成本 + 分级规则 |
| `outputs/bench_public/eval_timing.json` | 同上，机器可读 |
| `tools/build_eval_timing.py` | 台账生成脚本（从日志与产物自动汇总） |
| `tools/watch_up2_full.sh` | 慢评测看门狗：到硬上限自动止损并标记「未完成」 |
| `tools/chain_cvbench_fast.sh` | 串行链：等前一个实验结束后自动接下一个 |

> ⚠️ **统计陷阱（务必记住）**：`scores_3dchunk_*.json` 里有两个墙钟字段，
> **`wall_s_total` 是各分片墙钟之和（并发累加，不是墙上时钟）**，
> 并发 16 时约为真实墙钟的 4 倍；**`wall_s_all_jobs` 才是进程级真实墙钟**。
> 台账一律用后者。

**落地做法**：

```bash
# 慢评测一律先做「小样本可行性探针」，再决定是否放全量
bash tools/probe_3d_fixed.sh                 # 3D 判分成本探针（先量单样本耗时）
bash tools/run_ts_score_3d_96.sh            # 3D 判分（排除病态样本，941 s）
# 超时即停的标准做法：按样本索引排除，并把索引写进产物字段
python3 submission/bench_public/score_3d_chunked.py ... --exclude 56,78,82,98

# 慢评测带硬上限（到点自动停 + 标记未完成）
bash tools/launch_watch_up2.sh               # 上限 17:16:00
```

---

## 6. 产物清单

| 路径 | 内容 |
|---|---|
| `outputs/phase2/phase2_result.json` | 第二阶段验收结论（`passed`、`failures`） |
| `outputs/phase3_baseline/baseline_raw.json` | 第三阶段基线原始记录（18 条：6 样例 × 3 次） |
| `outputs/phase4/<tag>.json` | 第四阶段单变量实验原始数据 |
| `outputs/phase4/comparison.md` | 实验对比总表 |
| `outputs/phase4_batch/batch_sweep.json` | 批处理扫描数据 + 正确性逐条比对 |
| `outputs/tasks_official/` | 四类任务输出 + 标注图 |
| `outputs/batch_bs4/` | bs=4 标注图 |
| `outputs/probe_vllm/` | vLLM 服务化探针与跨后端一致性比对 |
| `outputs/phase5_probe_canonical/` | **规范化后**的 serving 探针（`probe_result.json` + `agreement.md`） |
| `outputs/phase5_probe_nocanonical/` | **未规范化**（修复前行为）的对照探针 |
| `outputs/phase5_smoke/smoke_results.json` | 服务级冒烟：四类任务 / 多图 / 流式 / 多轮 / 异常输入 |
| `outputs/phase5_bench/<chunk>/bench_results.json` | 六档压测汇总（每轮落盘） |
| `outputs/phase5_bench/<chunk>/bench_records.json` | 六档压测逐请求原始记录 |
| `logs/probe_tie.json` | grounding 首 token 平局的取证（两条取图路径的 top-10 概率与 logit） |
| `logs/determinism_probe.jsonl` | 3 进程 × 4 次的确定性记录（12/12 相同） |
| `logs/` | 每次运行的完整日志（含 8 份 sglang 后端探针） |
| `submission/bench/` | **官方格式压测**：两轮原始 jsonl、12 份档位日志、`summary.md`（官方 13 列）、`summary_detailed.md` |
| `submission/bench_pitfall_ignore_eos/` | `--disable-ignore-eos` 语义反转导致吞吐虚高 4× 的**错误口径原始数据**（留证） |
| `submission/bench_verify/` | 「每请求是否真生成满 1024 token」的独立小样本核对 |
| `submission/samples/MANIFEST.md` | 固定样例的 sha256、来源（upstream commit）与校验结论 |
| `outputs/phase4_parse_coverage/` | 官方解析正则的**覆盖边界**检查（验收标准 §4.3）：20 条用例，逐条给原文/解析值/失败原因 |
| `outputs/phase5_ab_noneager/<tag>/` | **E1** 非 eager A/B 产物（jsonl + 内存峰值 + 一致性比对） |
| `outputs/phase5_ab_bnb/<tag>/` | **E2** `--max-num-batched-tokens` A/B 产物（4096/8192/16384） |
| `outputs/phase5_gemm_trace/<tag>/` | **E3** rocBLAS layer3 trace（诊断用，非性能数据） |
| `outputs/phase5_token_accounting/` | **E4** 输出 token 计量口径对照实验（随机 vs 自然提示词） |
| `outputs/phase5_token_accounting/server_counter_segments.json` | 服务端引擎计数器逐窗口核算结果 |
| `logs/clean_repro/` | **干净环境复现**日志（env / smoke / verify / 性能复跑） |
| `reports/10a_server_counter_crosscheck.md` | 服务端计数器 vs 官方客户端逐档交叉核对表 |
| `reports/00_RUNBOOK.md` | 本文件 |
| `reports/01..10_*.md` | 环境 / 适配 / 基线 / 调优 / 最终 / SGLang 兼容 / 设备映射 / 基线差距评估 / vLLM 适配 / 阶段 5 压测 |
| `reports/11_tuning_experiments.md` | 服务级单变量调优实验 E1–E4（判定标准实验前登记） |
| `reports/12_official_asset_provenance.md` | upstream 资产/提示词溯源（case7 标 `NOT_REPRODUCIBLE`） |
| `reports/13_scnet_image_registry_survey.md` | **镜像库 22 条 SGLang 镜像清单 + Harbor 读镜像方法 + 各镜像构建栈实证** |
| `reports/14_new_instance_migration_plan.md` | **换镜像迁移方案：保留 vLLM 适配成果、一键重放、基线政策、风险与卡时估算** |
| `samples/scnet_image_inventory.json` | 机器可读镜像清单（22 条 + sf_b075 关联项） |
| `tools/bootstrap_vllm_adaptation.sh` | **新实例一键重放 vLLM 适配**（体检 / 装依赖 / 起服务 / 探针） |
| `outputs/bench_public/manifest/` | **公开基准数据完整性 manifest**（逐文件 SHA256 / 条目核对 / 全量解码检查 / CV-Bench 图像落盘映射） |
| `outputs/bench_public/refspatial/` | RefSpatial 逐题预测 + 官方 CLI 判分结果（4 变体 × 3 split） |
| `outputs/bench_public/tracespatial/` | TraceSpatial 逐题预测 + 官方 2D/3D 几何指标 |
| `outputs/bench_public/cvbench/` | CV-Bench 逐题预测 + 官方 FlagEvalMM 分组加权结果 |
| `outputs/bench_public/stability/stability.json` | 每基准 12 样例 × 2 次重复推理的一致性检查 |
| `outputs/bench_public/rescore_check.json` | **离线重新判分**的逐指标复现比对 |
| `submission/bench_public/` | 公开基准评测代码（协议层 / 引擎 / 三个 runner / manifest / rescore / stability） |
| `reports/15_public_bench_protocol.md` | 公开基准**协议冻结与差异表**（坐标口径两套官方实现的分歧与采用理由） |
| `reports/16_public_bench_results.md` | 公开基准评测**结果**（明确标注非官方赛事评测集；含 CV-Bench 差距分解与图像协议因子实验） |
| `reports/17_accuracy_regression.md` | **精度回归报告**（参考=eager 基线 / 候选=E1 非 eager，唯一变量=服务是否开图模式） |
| `outputs/bench_public/cvbench_protocol/` | **CV-Bench 图像输入协议因子实验**（缩放策略 × 编码格式，8 组，只跑 Count 788 题） |
| `outputs/bench_public/cvbench_upscale/` | **CV-Bench 上采样协议实验**（up2/up4 × png/jpeg95 + 同会话 canonical 对照，只跑 Count 788 题） |
| `outputs/bench_public/cvbench_count_samples/` | **Count 失分样例的图像导出**（15 张 + manifest.json，供人工目视核对标注口径） |
| `outputs/bench_public/verify_3d_merge.json` | **分片合并精确性对拍**结果（整集单进程 vs 逐样本分片，逐指标） |
| `outputs/bench_public/summary_table.md` | 三个公开基准的**汇总表**（验收要求 §8 的表格格式） |
| `outputs/bench_public/accuracy_regression.md` / `.json` | **精度回归对比**（参考=eager 基线 / 候选=E1 非 eager，含噪声底） |
| `outputs/bench_public_noneager/` | 候选配置（非 eager）的同口径复测产物 |
| `outputs/bench_public_noisecheck/` | **RefSpatial 噪声底**实测（全量同配置重跑一次） |
| `submission/bench_public/score_3d_chunked.py` | TraceSpatial 官方判分的**分片并行调度器**（官方 `summarize_acc.py` 零改动） |
| `submission/bench_public/merge_ts_scores.py` / `build_summary_table.py` / `compare_accuracy_runs.py` | 分片结果回填 / 汇总表 / 精度回归对比 |
| `outputs/bench_public_3d96/` | **TraceSpatial 3D 兜底判分**（排除 4 个深景病态样本，96/100，4 组合，941 s） |
| `outputs/bench_public_verify/` | 分片合并对拍的**阶段 A**（整集单进程）产物 |
| `outputs/bench_public/verify_3d_merge.json` | **对拍结果**：48 项指标全部一致 ✅ |
| `tools/run_ts_score_3d_96.sh` / `tools/verify_3d_merge.sh` / `tools/remerge_ts_3d.py` | 3D 兜底判分 / 合并精确性对拍 / 合并口径重算 |
| `outputs/bench_public/cvbench/task_breakdown.json` | CV-Bench 按 task / 选项数 / source 的失分层级分解（`analyze_cvbench.py` 产出） |
| `outputs/bench_public/accuracy_regression.{md,json}` | 基线 vs 候选的逐指标差值表 + 逐题输出变化率 |
| `outputs/bench_public_noneager/` | **E1 候选（非 eager）**的全量预测与判分（与基线同协议同样本） |
| `outputs/bench_public_noisecheck/` | **同配置重跑**的噪声底测量（判定 1 pt 级差值能否归因于配置） |
| `tools/run_accuracy_pipeline.sh` | 精度流水线（基线收尾 + 候选复测 + 汇总，判分任务多路并发） |
| `tools/exp_cvbench_image_protocol.sh` | CV-Bench 图像输入协议因子实验（8 组） |
| `tools/exp_cvbench_protocol_remaining.sh` | 同上，但**跳过已完成**的协议组（可续跑） |
| `tools/measure_refspatial_noise.sh` | RefSpatial 同配置重跑的噪声底测量（全量，三方对比） |
| `tools/status_acc.sh` | 精度流水线状态速查（各阶段/各协议进度） |
| `tools/fix_prefix_metadata.sh` | 隔离前缀里孤立 numpy 元数据的修复（accelerate/numpy 冲突） |
| `tools/install_o3d_018b.sh` / `tools/install_o3d_018c.sh` | 装 **open3d 0.18.0 + numpy 1.26.4** 到 `envs/pylibs_o3d_v2` 并补 sklearn 桩件（修复 3D 碰撞检查空转） |
| `tools/verify_o3d018_tsdf.sh` | 验证 0.18.0 下官方那组参数能抽出 TSDF 点（修复前为 0） |
| `tools/diag_tsdf_param_scan.sh` | open3d 0.20 的 `ScalableTSDFVolume` 参数空间扫描（定位体素语义变化） |
| `tools/diag_open3d_3d.sh` / `tools/diag_tsdf_synth.sh` / `tools/diag_rgbdtype.sh` | 3D 判分缺陷的逐步定位（深度图布局 / 合成平面对照 / RGBDImage 构造矩阵） |
| `tools/probe_3d_fixed.sh` | 3D 判分定点探测（指定样本区间的耗时 + 峰值 RSS） |
| `tools/fix_ts_terminal_states.sh` | 修正 TraceSpatial 预测记录的 `terminal_state` 记账（2D 2 元组被误判） |
| `submission/bench_public/score_3d_chunked.py` | **官方判分的逐样本分片并行调度器**（官方代码零改动；含 cgroup 内存闸门与精确合并口径） |
| `submission/bench_public/merge_ts_scores.py` | 把分片判分结果回填进 `tracespatial/summary.json` |
| `tools/run_ts_score_2d.sh` / `tools/run_ts_score_3d.sh` | TraceSpatial 2D / 3D 官方判分（4 个组合共用一个线程池） |
| `tools/verify_3d_merge.sh` | **分片合并精确性验证**（整集单进程官方跑 vs 分片合并，逐指标对拍） |
| `outputs/bench_public/verify_3d_merge.json` | 上述对拍结果 |
| `outputs/bench_public/cvbench_protocol/protocol_compare.json` | 8 种图像协议的 Count 准确率对比 |
| `outputs/bench_public_noisecheck/noise_floor.json` | RefSpatial **全量**同配置重跑的噪声底（逐 split + 默认范围均值） |
| `outputs/bench_public/eval_timing.md` / `.json` | **评测耗时台账**（自动生成：已完成墙钟 + 中止沉没成本 + L1~L4 分级） |
| `outputs/bench_public/cvbench_paired_tests.json` | **CV-Bench Count 配对符号检验**（噪声基线 / up2 / 提示词口径三组对拍的逐题翻转与 p 值） |
| `outputs/bench_public/cvbench_answer_format_audit.json` | **答案格式审计**（`(C) 1` vs 纯字母；官方判分器归一化口径 vs 严格字母口径） |
| `outputs/bench_public/cvbench_up2_full/` | **全量 CV-Bench（up2 第二口径）** 的逐题预测与官方判分结果 |
| `outputs/bench_public/cvbench_prompt/` | **提示词口径实验**（dataset 自带 prompt vs FlagEvalMM build_prompt，Count 788 题） |
| `tools/build_eval_timing.py` | **耗时台账生成脚本**（从日志/产物自动汇总） |
| `tools/paired_sign_test.py` / `tools/audit_prompt_format.py` | 配对符号检验 / 答案格式审计（可复算） |
| `tools/watch_up2_full.sh` / `tools/launch_watch_up2.sh` | 慢评测**看门狗**（到硬上限自动止损并标记「未完成」） |
| `tools/chain_cvbench_fast.sh` | 实验链（等前一个结束 → 自动接下一个，慢档自动换高并发） |
| `envs/pylibs_core/` | 通用隔离依赖前缀（**不含 numpy**，保持系统 numpy 1.25 + torch 可用） |
| `envs/pylibs_o3d/` | open3d 0.20.0 + numpy 2.2.6（**仅保留作对照**；它的 TSDF 失效，见 §5.7） |
| `envs/pylibs_o3d_v2/` | **open3d 0.18.0 + numpy 1.26.4 + sklearn 桩件**（3D 判分实际使用的前缀） |

---

## 7. 红线（本次任务全程遵守）

- ❌ 不改 `/opt/dtk`、`/opt/hyhal`、site-packages、公共 Python 环境（需要时在工作目录内做副本 + patch）
- ❌ 不改模型目录（只读）
- ❌ 不终止其他任务的进程、不重置设备、不改调度资源
- ❌ 不做会改变输出质量的改动（量化 / 换精度 / 改采样语义）除非明确授权
- ❌ 每次实验只改一个主要变量；变化未超过测量波动不宣称收益
- ✅ 所有缓存 / 临时文件 / 新增依赖都在工作目录内（`envs/`、`cache/`、`tmp/`）
- ✅ 不把私钥、令牌写入脚本或报告
