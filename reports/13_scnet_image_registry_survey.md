# 附 13：SCNet 镜像库与镜像仓库勘查（换镜像决策的事实基础）

> 采集时间：2026-10-27 17:40 – 18:35（浏览器枚举 + Harbor Registry API 核验）
> 适用对象：RoboBrain2.5-8B-NV / K100AI 适配任务的**换镜像决策**与后续复用
> 机器可读清单：`samples/scnet_image_inventory.json`
> 复现脚本：`tools/probe_scnet_images.sh`、`tools/probe_img_cfg.sh`、`tools/probe_jsyadmin_img.sh`、`tools/probe_sf_b075.sh`、`tools/probe_nightly_idx.sh`

---

## 0. 一句话结论

**本节点起不了 docker，但完全可以在不启动镜像的前提下把镜像的构建栈读出来**；平台镜像库里与 SGLang 相关的镜像共 **22 条**，其中
`aihub/dcu/jsyadmin/sglang-0.5.12-ubuntu22.04-dtk26.04-py3.10:v1.0`
与本节点 **DTK 同代（26.04-rc4）**，但 torch/vllm/sglang 全部换栈到 **2.10.0 / 0.18.1 / 0.5.12(das185)** —— 这正是报告 06 §6.5 判定的"SGLang 唯一可能路径"。

---

## 1. 「能不能直接起 docker 跑这个镜像」：不能

实测（本节点）：

```bash
for c in docker podman nerdctl ctr crictl skopeo crane; do command -v "$c" || echo "missing $c"; done
# 全部 missing
ls -la /var/run/docker.sock        # no such file
ls -la /run/containerd/containerd.sock   # no such file
head -3 /proc/1/cgroup
# 11:memory:/kubepods.slice/kubepods-pod747040e7_.../cri-containerd-7f4bcb47...scope
ls /dev | grep -iE 'kfd|dri|hsa'
# dri  kfd  mkfd
```

结论与理由（逐条已验证，不是推断）：

| 事实 | 依据 |
| --- | --- |
| 我们自己就是 k8s 容器里的进程 | `/proc/1/cgroup` = `kubepods.slice/.../cri-containerd-*.scope` |
| 没有容器运行时客户端，也没有 daemon socket | 7 个 CLI 全部 missing；`docker.sock` / `containerd.sock` 均不存在 |
| 嵌套容器拿不到 DCU 设备 | DCU 由宿主通过 `/dev/kfd` 透传（当前可见），嵌套层无法再透传，且平台没有暴露 `--privileged` / `--device` 入口 |
| 平台只提供"用镜像开实例"的入口 | 镜像详情页只给 `docker pull` 字符串与「快速开发 / 克隆」按钮，没有"在节点上跑容器"的入口 |

所以：**运行镜像 = 在平台新开一个实例（消耗新卡时）**，不是在本 Pod 里 `docker pull && docker run`。

### 1.1 但镜像"内容"可以在不运行的情况下读出来

Harbor（`appstore.scnet.cn:5000`）**匿名可取 token**，因此可以像读 OCI 制品一样读镜像元数据：

```bash
H=appstore.scnet.cn:5000
REPO=aihub/dcu/jsyadmin/sglang-0.5.12-ubuntu22.04-dtk26.04-py3.10

# ① 匿名 token（无需登录态）
curl -sS "https://$H/service/token?service=harbor-registry&scope=repository:$REPO:pull"
# -> {"token":"eyJhbGciOiJSUzI1NiIs..."}

# ② manifest
curl -sS -H "Authorization: Bearer $TOK" \
  -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
  "https://$H/v2/$REPO/manifests/v1.0"

# ③ config blob（Env / Labels / history.created_by = 完整构建命令）
CFG=$(python3 -c 'import json;print(json.load(open("m.json"))["config"]["digest"])')
curl -sS -H "Authorization: Bearer $TOK" "https://$H/v2/$REPO/blobs/$CFG"

# ④ layer 清单（size + digest），可流式 tar -tz 扫描指定文件
curl -sS -H "Authorization: Bearer $TOK" "https://$H/v2/$REPO/blobs/<layer_digest>" | tar -tz | grep -E 'aiter|sglang|torch'
```

> 凭据纪律：本仓库/报告**不写入任何 token、口令、代理地址**。镜像 `Env` 里内嵌的代理凭据已在落盘前用
> `sed -E 's#(https?://)[^:/@]+:[^@/]+@#\1[REDACTED]@#g'` 脱敏（见 `tools/probe_jsyadmin_img.sh` 第 1 步）。

---

## 2. 平台镜像库：SGLang 相关镜像全量清单（22 条）

入口：`https://www.scnet.cn/ui/aihub/image`，搜索框输入 `sglang` → 共 22 条，2 页（每页 12 条）。

> 平台「框架类型」筛选里**没有 SGLang 这个选项**（只有 PyTorch / vLLM / SGInfer / TGI 等），
> 所以 SGLang 镜像只能靠**名称与描述**命中，不能靠框架标签枚举。这是枚举口径的已知局限。

按上传者归并（下载量为 2026-10-27 快照）：

| # | 镜像名 | 上传者 | 下载 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | `sglang_qwen3.8_27b_runner` | string199 | 73 | 描述明写 "DTK26.04+sglang 0.5.12"，单卡免费跑 2 小时 |
| 2 | `jupyter-sglang-qwen3.8` | jsyadmin | 56 | 光合社区发版 |
| 3 | `jupyter-sglang-deepseek-v4-dev-20260623` | jsyadmin | 44 | |
| 4 | `jupyter-sglang0.5.12-ubuntu22.04-dtk26.04-py3.10-20260620` | wxh01 | 20 | |
| 5 | `jupyter-sglang5.12-ubuntu22.04-dtk26.04-py3.10-qwen3.6` | **zhr5535** | 28 | **★ 本实例当前镜像** |
| 6 | `sglang-0.5.12-ubuntu22.04-dtk26.04-py3.10` | **jsyadmin** | 22 | **★★ 迁移首选**（见 §3） |
| 7 | `jupyterlab-sglang` | wanglufei1984 | 6 | 基于官方 base `harbor.sourcefind.cn:5443/hcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10` |
| 8 | `jupyter-sglang-qwen3.8` | ac5vlfhchc | 4 | |
| 9 | `sglang0.5.12_zhuzhou` | jsyadmin | 3 | |
| 10 | `sglang-0.5.10` | aisearch_sudy | 3 | |
| 11 | `sglang-v2-jupyter` | hww18680316531 | 2 | DTK **26.04.2** 变体，底层 `pypi.sourcefind.cn/jenkins/model_test_env/sglang:0.5.12-ubuntu22.04-dtk26.04.2-YYBU-py3.10-20260731` |
| 12 | `sglang0.5.12-ubuntu22.04-dtk26.04-py3.10-20260620` | clock88731 | 2 | clone 自 harbor `dcu/admin/base/custom:...` |
| 13 | `jupyter-sglang5.12-ubuntu22.04-dtk26.04-py3.10-qwen3.6` | jsyadmin | 16 | 与 #5 同名的另一个上传者 |
| 14 | `qwen3.5_sglang_fix20260818` | jsyadmin | 6 | |
| 15 | `sglang0.5.18-ubuntu22.04-dtk2604-py3.10-sf_b075-minimax-h3` | jsyadmin | 5 | torch **2.11.0** + sglang **0.5.18**，新 AICC（见 §4） |
| 16 | `qwen3.8_sglang_hjq` | jsyadmin | 4 | |
| 17 | `qwen3.8-sglang0.5.12-ubuntu22.04-dtk2604-py3.10` | acb9yhyenr | 4 | |
| 18 | `flagos-sglang` | guancl | 1 | |
| 19 | `qwen3.5_sglang_fix20260818` | Qin1024 | 0 | |
| 20 | `jupyter_qwen-3.8-flash` | eq1qe | 27 | sglang0.5.12 |
| 21 | `k100ai_qwen3.8_0828` | stable86 | 6 | 少见的显式 **K100AI** 命名 |
| 22 | `sgalng-v0512` | jsyadmin | 2 | `shca17` 分支 |

关键词 `b075` 额外命中的同批次镜像（非 sglang 关键词内）：

- `vllm_0.21_sf_b075_d2` —— 与 #15 同批 `sf_b075` 栈；`vllm 0.21` 对应厂商源 `vllm-0.21.0+das.dtk2604.torch2100`（torch 2.10 系）
- `tmk_0.21.0-ubuntu22.04-dtk2604-py3.10-sf_b075` —— **registry 已核验**：created 2026-10-22，DTK-26.04-rc4，torch 2.10.0

镜像详情页 URL 形态（可复用于直接定位）：
`https://www.scnet.cn/ui/aihub/image/<org>/<imageId>`，版本页 `<imageId>/versions`。
例：`https://www.scnet.cn/ui/aihub/image/jsyadmin/2096964376247836674`

---

## 3. 迁移首选镜像的完整构建栈（registry 实证）

`aihub/dcu/jsyadmin/sglang-0.5.12-ubuntu22.04-dtk26.04-py3.10:v1.0`

| 项 | 值 |
| --- | --- |
| config `created` | `2026-07-09T01:11:41Z` |
| layer 数 / 压缩体积 | 38 层 / 10,649,318,176 B（9.92 GiB；平台显示 37.51 GB 为解包后） |
| DTK | `DTK-26.04-rc4-ubuntu20.04-x86_64.tar.gz`（`DTK_VERSION=2604`） |
| 构建期 pip 索引 | `http://10.16.1.201:9929/nightly/dtk2604/+simple/`（内网 nightly） |
| 平台显示 | 大小 37.51 GB / Python py3.10 / DTK dtk26.04 / Ubuntu22.04 / Jupyter + SSH / 使用区域 华东一区[昆山] |

构建命令（config `history` 原文，节选去重，**未改写数值**）：

```bash
torchversion=2.10.0
pip install ray[data,train,tune,serve] -i https://mirrors.aliyun.com/pypi/simple/
pip install amdsmi
pip install cupy
pip install vllm==0.18.1
pip install vllm-hcu==0.18.1
pip install apache-tvm-ffi==0.1.9 -i https://mirrors.aliyun.com/pypi/simple/
pip install torch==${torchversion} torchvision
pip install torch==${torchversion} flash-attn
pip install torch==${torchversion} lightop
pip install torch==${torchversion} lmslim
pip install torch==${torchversion} deepgemm
pip install torch==${torchversion} aiter
pip install torch==${torchversion} mooncake_transfer_engine
pip install torch==${torchversion} deep_ep
pip install torch==${torchversion} tilelang
pip install torch==${torchversion} vllm-hcu
pip install torch==${torchversion} causal_conv1d
pip install torch==${torchversion} flash_mla
pip install torch==${torchversion} fastsafetensors
pip install numpy==1.25.0
git clone https://gh-proxy.com/https://github.com/openucx/ucx.git && ./configure --prefix=/opt/ucx --with-rocm=/opt/dtk --without-cuda && make && make install
pip install nvidia-cutlass-dsl==4.4.2 -i https://mirrors.aliyun.com/pypi/simple/
pip install sgl-deep-gemm==0.1.0 -i https://mirrors.aliyun.com/pypi/simple/
pip install sglang==0.5.12 sglang-router sglang[diffusion]
pip install numpy==1.25.0
pip install kernels==0.14 -i https://mirrors.aliyun.com/pypi/simple/
```

**这个栈对本任务的意义（三点，均为事实推论）**：

1. `torch 2.10.0` + `aiter`（来自 nightly dtk2604 索引）≈ 厂商源的 `aiter-0.1.5+das185.dtk2604.torch2100.2608131714`，
   即报告 06 §6.5 判定的 **das185 / torch2100 栈** —— 本节点缺的就是这一层。
2. `vllm==0.18.1` + `vllm-hcu==0.18.1` = 厂商源 `vllm-0.18.1+das.dtk2604.torch2100.2606041655.ge5a62f`，
   与我们当前用的 `0.15.1+das.opt1.alpha...torch290` **不同大版本、不同 torch 大版本** → 现有 vLLM 适配成果需要在新栈上**重放并重新验证**（见报告 14）。
3. `sglang==0.5.12` + `sglang-router` + `sglang[diffusion]` + `sgl-deep-gemm` + `deep_ep` + `tilelang` + `flash_mla`
   —— 说明厂商在这张镜像里是**按 SGLang 的完整依赖闭包装的**，而不是我们当前镜像那种"只装 sglang 本体、aiter 残缺"的状态。

---

## 4. 其他候选镜像的栈（registry 实证）

| 镜像 | config created | DTK | torch | vllm | sglang | 与本节点 DTK 同代？ |
| --- | --- | --- | --- | --- | --- | --- |
| `jsyadmin/sglang-0.5.12-ubuntu22.04-dtk26.04-py3.10` | 2026-07-09 | 26.04-rc4 | 2.10.0 | 0.18.1 + hcu | 0.5.12 | **是** |
| `jsyadmin/sglang0.5.18-...-sf_b075-minimax-h3` | 2026-09-07 | 2604，但 AICC 为 `nightly dtk_llvm_286f35_202608311711`（**更新 AICC**） | 2.11.0 | — | 0.5.18 | 否（跨代） |
| `jsyadmin/tmk_0.21.0-...-sf_b075` | 2026-10-22 | 26.04-rc4 | 2.10.0 | vllm-router 0.1.15+dtk2604.torch2100.2609021808.g1d10e7 | — | **是** |
| `zhr5535/jupyter-sglang5.12-...-qwen3.6`（当前） | — | 26.04 | 2.9.0 | 0.15.1 | 0.5.12（残缺） | 是（就是本节点） |

> `sf_b075` = 同一批新栈标记：同批还有 `vllm_0.21_sf_b075_d2`（对应厂商源 `vllm-0.21.0+das.dtk2604.torch2100`）。
> 但 `sglang0.5.18-...-sf_b075` 换成了 **torch 2.11.0 + 新 AICC**，落地不确定性高于 §3 那张。

---

## 5. 厂商 PyPI：两个公共索引（本次新发现第二个）

### 5.1 `https://pypi.sourcefind.cn/release/dtk2604/`（devpi，原先用这个）

torch2100 系（= torch 2.10.0）：

```
torch-2.10.0+das.opt1.dtk2604.2606161421.g91a580-cp310-cp310-manylinux_2_28_x86_64.whl
torch-2.10.0+das.opt1.dtk2604.2608131652.g28498c-cp310-cp310-manylinux_2_28_x86_64.whl
vllm-0.18.1+das.dtk2604.torch2100.2606041655.ge5a62f-cp310-cp310-...whl
vllm-0.21.0+das.dtk2604.torch2100.2606111143.g8c979d-cp310-cp310-linux_x86_64.whl
sglang-0.5.12+das185.dtk2604.torch2100.2608121934.gcbf025-cp310-cp310-linux_x86_64.whl
sglang-kernel-0.4.2.post2+das185.dtk2604.torch2100.2608121934.gcbf025
sglang-router-0.3.2+das.dtk2604.torch2100.2608030956.g3eb90b
aiter-0.1.5+das185.dtk2604.torch2100.2608131714.gb0979d
```

同索引另有 torch 2.11.0 / 2.12.0 / 2.9.0 / 2.8.0 / 2.7.1 / 2.5.1。
**不含** `wave-lang`；`flashinfer-python` 来自 `open/pypi` 上游 CUDA 轮子（DCU 不可用，见报告 06 §6.4）。

### 5.2 `https://pypi.sourcefind.cn/nightly/dtk/`（**本次新发现**）

```
aiter-0.1.7+dtk2610.torch2110.2610271247.g759dc1-cp312-...whl
sglang_kernel-0.4.6.post1+dtk2610.torch2110.2610271454.g9f60f8-cp39-abi3-...whl
sglang_router-0.3.2+dtk2610.torch2110.2610121921.gc9a84f-cp38-abi3-...whl
vllm_hcu-0.28.1rc1.dev491+dtk2610.torch2110.2610271809.g5615af-cp312-...whl
torch-2.9.0+das.opt1.dtk2604.2608201057.g463d51-cp310-...whl
```

注意：这一代是 **DTK 26.10 + torch 2.11.0 + Python 3.12（cp312）**，与本节点（DTK 26.04 / py3.10）**不同代**，
只有它对应的镜像（如 `sf_b075` 那批）才可能跑。索引本身**不能**在本节点直接用。

---

## 6. 本节点现状 vs 迁移目标对照

| 维度 | 本实例（zhr5535/jupyter-sglang5.12-...-qwen3.6） | 迁移目标（jsyadmin/sglang-0.5.12-...dtk26.04） |
| --- | --- | --- |
| DTK | 26.04（`/opt/dtk/dcc version 25.10.0-0`，clang 17.0.0） | 26.04-rc4（**同代**） |
| torch | 2.9.0（hip 6.3.26113，`cuda_avail=True`） | **2.10.0** |
| vllm | 0.15.1 | **0.18.1 + vllm-hcu 0.18.1** |
| sglang | 0.5.12（残缺：aiter import 期 ValueError 等，见报告 06） | 0.5.12 + router + diffusion + deep_ep + tilelang + flash_mla |
| python 布局 | `/opt/conda` + `/opt/sglang-kernel` 两个 venv | 系统 python3.10（`/usr/local/lib/python3.10/dist-packages`） |
| pip 默认源 | `https://pypi.tuna.tsinghua.edu.cn/simple` | 构建期内网 nightly `10.16.1.201:9929/nightly/dtk2604` |

---

## 7. 已验证事实 / 待验证假设 / 已知阻塞

**已验证事实**
- 本节点无任何容器运行时 CLI 与 daemon socket，嵌套容器无 DCU 透传 → 不能本地跑镜像。
- Harbor 匿名可取 token，可读 manifest / config / layers（已多次实测）。
- 平台镜像库 SGLang 关键词命中 22 条，已全量枚举（含两页）。
- §3 镜像的完整构建栈、layer 体积、config created 时间均为 registry 实测。
- 厂商源两个公共索引的包清单已实测（release/dtk2604 与 nightly/dtk）。

**待验证假设（只能在新实例里证）**
- H1：torch 2.10.0 + `aiter 0.1.5+das185` 能让 SGLang 0.5.12 的 aiter 后端**不再**触发报告 06 §3.1 的 import 期 `ValueError`。
- H2：`sglang[diffusion]` / `deep_ep` / `tilelang` 齐备后，SGLang 能起服务（不再命中 `NameError: paged_attention_ragged`）。
- H3：vLLM 0.18.1 在 torch 2.10 栈上可跑通 RoboBrain2.5-8B-NV；`Qwen3VLTextConfig.tie_word_embeddings` 错位**是否仍存在**取决于 transformers 版本，需现场探测。

**已知阻塞**
- B1：官方精度材料（基线 JSON / 评测子集 / 判分器）仍未下发 → 精度项 `BLOCKED`，且精度是 Gate 级门槛（赛事 PDF 第 5 页）。
- B2：换实例 = 新卡时；官方"首次性能基准只能取一次"，换栈后 vLLM 数字属**新配置**，不能替换已取基线（处置见报告 14 §4）。
- B3：新实例里 SGLang 是否真能跑通未知（依赖 H1/H2），必须留足排障时间。

---

## 8. 证据索引

| 内容 | 文件 |
| --- | --- |
| 22 条镜像清单（机器可读） | `samples/scnet_image_inventory.json` |
| 镜像库枚举 | `tools/probe_scnet_images.sh` |
| Harbor token + manifest + config + layers | `tools/probe_img_cfg.sh`、`tools/probe_jsyadmin_img.sh` |
| sf_b075 / tmk 栈核验 | `tools/probe_sf_b075.sh` |
| nightly 索引清单 | `tools/probe_nightly_idx.sh` |
| 本机 DTK/torch 现状 | `tools/probe_torch210.sh` |
| 迁移方案 | `reports/14_new_instance_migration_plan.md` |
| SGLang 不可用根因（前置） | `reports/06_sglang_compatibility.md` |
