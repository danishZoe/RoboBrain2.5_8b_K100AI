# 设备型号映射说明：平台卡型「K100AI / 异构加速卡 AI-64GB」↔ 运行时 `K500SM_AI`

> 采集时间：2026-10-27　节点：`crdnotebook-2114695384420274177-scnjz2s4uc-68757`（SCNet K8s Pod）
> 目的：满足《RoboBrain2.5-8B-NV 验收标准》§2「平台设备名称与实际运行时名称的对应证据；型号不一致不能自行视为同卡型」
> 与 §11「设备名称若实际属于同一平台卡型，需要提供可核验的映射说明」。

---

## 1. 结论

本任务实际占用的加速卡**就是平台卡型 K100AI**（SCNet 控制台标签「DCU 异构加速卡 AI 64GB ×1」）。
其运行时 Card Series 字符串为 **`K500SM_AI`**，与平台采购口径的名字不同，**但指同一机型**——依据是三条彼此独立、
互不依赖的证据轴同时吻合：

| 证据轴 | K100AI（平台口径 / proj_v1 已调优节点） | 本节点实测 | 吻合 |
|---|---|---|---|
| 架构 codename | `gfx928` | `gfx928:sramecc+:xnack-` | ✅ |
| KFD 设备节点名 | `kongminge` | `kongminge` | ✅ |
| 单卡显存 | 64.0 GB | 63.98 GiB | ✅ |

因此无需再假设「手头是一张与 K100AI 不同型号的卡」：**是同一张卡，只是驱动层字符串换了叫法**。
这一结论同时解除了 `01_environment.md` / `05_final.md` 中「型号与任务描述不符」的表述。

---

## 2. 本节点实测证据

全部命令在容器内直接执行，输出为原样摘录。

| # | 命令 | 关键输出 |
|---|---|---|
| 1 | `/opt/hyhal/bin/hy-smi --showproductname` | `Card Series: K500SM_AI` / `Card Vendor: Chengdu C-3000 IC Design Co., Ltd.` |
| 2 | `/opt/hyhal/bin/hy-smi -a` | `Device ID: 0x6210`、`VBIOS version: 5.715.001000p.618009`、300 W 上限、PCIe x16 8.0GT/s |
| 3 | `/opt/hyhal/bin/hy-smi --showuniqueid` / `--showserial` | `714e6a2dfd1240a1` / `TRYQ170009080501`（本卡唯一标识，用于与平台工单核对） |
| 4 | `python3 -c "import torch; p=torch.cuda.get_device_properties(0); ..."` | `name=K500SM_AI`、`gcnArchName=gfx928:sramecc+:xnack-`、`cc=(9,2)`、`mem=63.98 GiB`、`SMs=120`、`L2=8 MiB` |
| 5 | `cat /sys/class/kfd/kfd/topology/nodes/5/name` | `kongminge`（本 Pod 绑定的 HCU 节点） |
| 6 | `grep gfx_target_version /sys/class/kfd/kfd/topology/nodes/5/properties` | `gfx_target_version 90208`（= gfx 9.2.8 = gfx928）；同文件 `simd_count 480`、`lds_size_in_kb 64`、`max_engine_clk_fcompute 1400` |
| 7 | `/sys/bus/pci/devices/*/class` 扫描 | 宿主共 4 张 HCU：`0000:04:00.0 / 27:00.0 / 43:00.0 / 63:00.0`，全部 `vendor=0x1d94 device=0x6210`、`driver=hycu`；本 Pod 只分到 1 张（`/dev/dri/card2` → `0000:27:00.0`） |
| 8 | `cat /sys/module/hycu/version` | `6.3.16-V1.1.0c`（内核驱动） |
| 9 | `/opt/dtk/bin/hipconfig --version` | `6.3.26113`（DTK 运行时）；`HIP_RUNTIME: rocclr` |
| 10 | `torch.cuda.device_count()` | `1`（容器内可见卡数） |

**卡数说明**：宿主机物理 4 张卡，本任务 Pod 按资源配额分到 1 张（64 GB / 1 卡），与控制台「DCU 异构加速卡AI 64GB, 1」一致。

---

## 3. 跨节点对照：与本工作区 proj_v1 已调优的 K100AI 节点比对

proj_v1 的 `qwen38-bf16-K100AI-tuning` 任务在**另一台** K100_AI 节点上完成调优，其自述机型标识与本节点逐项吻合：

| proj_v1 记录位置 | proj_v1 记录内容 | 本节点实测 | 吻合 |
|---|---|---|---|
| `agent_space/GEMM调优纠错清单.md:23` | 「机型映射：K100_AI = `KME` → `library_gpu4`（gfx928）」 | gfx928 | ✅ |
| `agent_space/GEMM调优纠错清单.md:107` | 「`library_gpu4` = gfx928 = ✅ K100_AI (KME)」 | gfx928 | ✅ |
| `agent_space/prepare_kme_tuner.sh:133` | 「GPU target: gfx928 (K100_AI, node name kongminge)」 | gfx928 + `kongminge` | ✅ |
| `agent_space/note.md:485` | 「我们（K100_AI 项目）= K100_AI / gfx928（KME）」 | gfx928 | ✅ |
| `_cleanup_backups/.../summary.md:1868` | 「K100_AI 每卡 total 64.0 GB」 | 63.98 GiB | ✅ |
| `note.md:767` | 「芯片 / 架构 = K100AI / gfx928」 | gfx928 | ✅ |

**架构 codename + KFD 节点名 + 单卡容量三项同时一致**，不可能是另一代硅：proj_v1 已用
「K100_AI ≡ gfx928 ≡ KME ≡ library_gpu4」这一映射做过机型纠错（曾因误用 BW/gfx936 的 `library_gpu5` 导致 1.11x 收益错位），
该纠错结论对本节点同样成立。

---

## 4. 名字为什么不同

1. **两套命名体系**：平台/赛事按采购型号命名卡型（K100-AI）；驱动层 HAL 另有一套 SKU 字符串表。
2. **HAL 的 SKU 表同时含多代命名**：DTK 26.04 的 `libhydmi.so.1` 中同时存在
   `K100_AI` / `K100_AI_i` / `K100_LC` / `K100_LC_AI` / `K100_LC_E_AI` / `K500SM_AI` / `K500SM_B` / `K550SM_AI`
   （`strings /opt/hyhal/lib/libhydmi.so.1` 摘录）。
3. **决定性判据**：若 `K500SM_AI` 是比 K100AI 更新一代的硅，其 gfx target 应为 gfx93x/gfx95x；
   实测仍是 **gfx928**，与 proj_v1 记录的「K100_AI = gfx928」完全一致。
   → 名字差异是同一代硅在不同命名体系/不同 DTK 版本下的字符串差异，**不是换卡**。
4. **时间线佐证**：proj_v1 在 DTK 25.04 宿主 + DTK 26.04 容器上把该机型记为「K100_AI / gfx928 / kongminge」；
   本节点是 DTK 26.04 镜像，报「K500SM_AI」，但架构与节点名未变。

---

## 5. 对已有适配结论的影响

- 本次全部 API / 运行时结论都基于**实测运行时**（DTK 6.3.26113、hip 6.3、torch 2.9.0、transformers 5.3.0），
  与卡型如何命名无关，**无需重做、结论全部有效**。
- proj_v1 的机型相关结论可迁移到本节点：gfx928 → `KME` → `library_gpu4`（**不是** BW 的 `library_gpu5`），`cu_count` 仅 0。
- proj_v1 用过的 `SGLANG_KV_LAYOUT_HCU_FA`（旧 DTK 的 K100 专属 KV layout）在**本镜像不存在**，
  照搬该环境变量会得到「未知变量」而非加速，属于无效迁移。
- 报告统一写法：**「K100AI（运行时 Card Series: `K500SM_AI`，gfx928，64 GB）」**。

---

## 6. 赛事口径：报告里的"卡型"到底该填什么

回到赛事原文（`material/超算互联网 × 光合开发者社区_国芯国模_黑客松 赛事详情.pdf` + `material/国芯国模黑客松-技术报告模板.md`）：

| 依据 | 原文 | 含义 |
|---|---|---|
| PDF 第 1 页「硬件固定」 | 「本次硬件为 **异构加速卡BW** 和 **异构加速卡AI-64GB**，其它硬件暂考虑不考虑」 | 比赛用卡**限定为两个档位**，不是任意国产卡 |
| PDF 第 3 页「模型清单」 | 计件粒度 = 模型 × 卡型 × 框架；「卡型不同按不同件数计（异构加速卡BW、异构加速卡AI-64GB 各算一件）」 | 两个卡型是**两条平行赛道** |
| PDF 第 1 页 | 「**不排名、不跨题比**：每个模型只对照自己的官方 GPU 基线，达标即过」 | 不做跨卡型 PK |
| PDF 第 5 页「性能」 | 「首次提交须附性能实测…结果作为该件的性能基准；之后在**同卡型、同框架、同负载**下性能提升 10% 及以上的优化另单独计费」 | 优化收益是**同卡型自比** |
| 报告模板 第 2 节 | 卡型占位符 `【BW1000 / K100-AI】`、`Device: 【K100_AI】 x 【N】`、`加速卡与显存 【K100_AI / BW1000】×【N】；【64GB / 其他】` | **"K100-AI" 就是官方对「异构加速卡AI-64GB」这一档的写法** |

结论：**"异构加速卡"是官方对这类国产 DCU 的统称，"K100AI"是官方报告模板给出的该卡型档位名，
两者指同一件事，不是"比赛要求异构卡 vs 指定 K100AI"的二选一。**

**官方已直接给出对应关系（决定性）**——赛事随题下发的 `env.sh` 脚手架模板原文：

```bash
export CARD="K100-AI"  # 卡型：BW1000 | K100-AI（K500-SM 按 K100-AI 计）
```

即官方明文规定：**`K500-SM` 按 `K100-AI` 计**。本节点运行时 Card Series 正是 `K500SM_AI`，
故 `env.sh` 中 `CARD="K100-AI"` 为官方口径下的正确取值，无需任何自行解释或映射论证。

同一官方脚手架的 `env_check.sh` 会把卡型写进 `env.log`：

```bash
hy-smi   # 国产异构加速器卡型、显存与占用
python -c "... print('Device:', torch.cuda.get_device_name(0), 'x', torch.cuda.get_device_count())"
```

→ 本节点 `env.log` 实际会呈现 `Card Series: K500SM_AI` 与 `Device: K500SM_AI x 1`，
故报告第 2 节 `Device:` 字段应写 `K500SM_AI (K100-AI) x 1`，
既符合官方占位符 `【K100_AI】 x 【N】`，又与 `env.log` 原文一致（模板第 7 页要求两者必须一致）。

本次任务落在哪一档、该怎么填：

| 模板字段 | 填法 |
|---|---|
| 卡型 | `K100-AI`（平台资源档：异构加速卡 AI-64GB） |
| 卡数 | `1`（单机 TP=1） |
| `Device:` | `K500SM_AI (K100-AI) x 1` |
| 加速卡与显存 | `K100-AI × 1；64GB` |

三条实际影响：

1. **硬件合规无风险**：本 Pod 资源即控制台所示「DCU 异构加速卡 AI 64GB ×1」，正落在允许档位内；
   PDF 第 6 页的反作弊条款针对改权重/硬编码答案/伪造日志，与卡型命名无关。
2. **计件与基线不跨卡型**：精度只需对照"官方 GPU 结果"（劣化 ≤2%），性能优化在同卡型同框架同负载下自比；
   运行时字符串叫 `K500SM_AI` 还是 `K100_AI` **对计件零影响**。
3. **真正的验收缺口仍是框架**：PDF 第 2 页 T02 明确是「SGLang / vLLM 适配」，验收要求
   OCRBench + MMMU 较官方 GPU 基线劣化 ≤2%（OCRBench 千分制 ≤20 分 / MMMU ≤2 pt）、
   VLA/具身模型动作与坐标一致率 ≥98%、单图/多图均正常。
   本次目前只有 transformers 直跑基线，SGLang 在本节点 5 个后端全崩 —— **这是离"正式验收通过"最远的一项，与卡型无关**。

---

## 7. 仍需外部确认（容器内无法取得，不影响上述结论）

| 项 | 需要的材料 | 获取方 |
|---|---|---|
| 平台卡型 SKU 书面凭证 | SCNet 资源详情页/工单中该实例的卡型字段（现有文本记录：「DCU : 异构加速卡AI 64GB, 1」） | 用户 / 平台控制台 |
| 官方命名演进说明 | DTK/HAL release notes 中 gfx928 各 SKU 别名的对应关系（本节点无外网，镜像内无 release notes） | DTK 发行说明 |

---

## 8. 算子库与 GEMM 调优工具的机型映射（本节点一手核实）

> 这一节是本机**直接读到官方工具源码** + **code object 文件名**得出的，不引用任何转述。

### 8.1 官方 `constants.py` 原文（本节点 `/opt/dtk/lib/rocblas/auto_select_test/auto_select_tools/src/settings/constants.py`）

```python
HARDWARE_LIBRARY_DIR: Dict[str, str] = {
    "ZF": "library",  "ZF2": "library_gpu2", "KM": "library_gpu3",
    "KME": "library_gpu4", "ECO": "library_gpu4",
    "BW": "library_gpu5", "NMZ": "library_gpu6",
}
HARDWARE_YAML_DIR: Dict[str, str] = {
    "ZF": "hpl_lite",  "ZF2": "hpl_lite_gpu2", "KM": "hpl_lite_gpu3",
    "KME": "hpl_lite_gpu4", "ECO": "hpl_lite_gpu4",
    "BW": "hpl_lite_gpu5", "NMZ": "hpl_lite_gpu6",
}
```

### 8.2 code object 文件名直接印证架构

| rocBLAS 库目录 | `.co` 文件名里的架构 | 对应卡型 |
|---|---|---|
| `library_gpu2` | `..._gfx906.co` | ZF2 |
| `library_gpu3` | `..._gfx926.co` | KM（K100_LC） |
| **`library_gpu4`** | **`..._gfx928.co`** | **KME = K100-AI / K500-SM ← 本节点** |
| `library_gpu5` | `..._gfx936.co` | BW = BW1000 |
| `library_gpu6` | （`..._72cu` 系列） | NMZ |

本节点 `torch` 报告 `gcnArchName = gfx928:sramecc+:xnack-` → **必须用 `library_gpu4`**。

### 8.3 `cu_count` 机制（由 `.dat` 文件名直接证实）

| 库目录 | 带 cu 后缀的配置文件 | 结论 |
|---|---|---|
| `library_gpu4` | **无任何 `_NNcu` 文件** | KME 只能 `cu_count=0` |
| `library_gpu5` | `GemmGeneralityAdd_48cu.dat`、`_64cu.dat` | BW 允许 `[48, 64]` |
| `library_gpu6` | `GemmGeneralityAdd_72cu.dat` | NMZ 允许 `[72]` |

### 8.4 ⚠️ 随包默认配置在 K100AI 上是错的（proj_v1 已踩过）

本节点 `/opt/dtk/lib/rocblas/auto_select_test/auto_select_tools/configs/tuning_args.txt` 随包默认值：

```ini
mode=user
source=csv
hardware_type=BW          # ← 在 K100AI 上照抄即错：指向 library_gpu5/gfx936
strategy=common
gpu_ids=0 1 2 3 4 5 6 7  # ← 本容器只有 1 张卡（容器内 index 0）
should_clean=1            # ← 会删除 optimization_configs/ 与 optimization_artifacts/
cu_count=0                # ← 与 KME 兼容（BW 场景下才需要 48/64）
update_target_config=0    # ← 保留：不覆盖目标包，可秒级回滚
```

proj_v1 的教训：该工具链会**静默**按 `hardware_type` 重定向到另一代架构的库，
在 K100_AI 上误用 `BW` 导致 2.5 小时 / 4647 个候选全部作废、产出 2.1 GB gfx936 无效产物。
**本卡正确取值：`hardware_type=KME`、`cu_count=0`。**

### 8.5 可用性与约束

- ✅ 工具**就在本容器内**：`/opt/dtk/lib/rocblas/auto_select_test/auto_select_tools/`（含 `Auto_tuning.py`、
  `source-all` / `source-quick` / `source-test`、`csv_size/csv` 现成数据）。
- ⚠️ `rocblas_layer3_size/` 与 `rocblas_layer4_size/` 是**空目录** → `source=layer3/layer4` 需要自采 trace，
  本次应走随包的 `source=csv`。
- ✅ `librocblas.so` 支持 **`ROCBLAS_TENSILE_LIBPATH`** 环境变量 → 调优产物可放在工作目录内加载，
  **无需修改 `/opt/dtk`**（符合"工作目录外环境只读"的边界）。
- ⚠️ `HARDWARE_YAML_DIR` 指向的 `hpl_lite_gpu*` 目录**在本镜像 DTK 中不存在** → hipBLASLt/HPL 那条 YAML 路线不可用，
  只有 rocBLAS/Tensile 一条路可走。

---


## 9. 复现命令

```bash
# 在 K100AI 节点容器内（工作目录 /root/private_data/workspace/robo）
source /opt/dtk/env.sh

/opt/hyhal/bin/hy-smi --showproductname
/opt/hyhal/bin/hy-smi --showuniqueid
/opt/hyhal/bin/hy-smi --showserial

python3 -c "import torch; p=torch.cuda.get_device_properties(0); \
print(p.name, p.gcnArchName, (p.major,p.minor), round(p.total_memory/1024**3,2), p.multi_processor_count)"

cat /sys/class/kfd/kfd/topology/nodes/5/name
grep -E "gfx_target_version|simd_count|lds_size" /sys/class/kfd/kfd/topology/nodes/5/properties
for d in /sys/bus/pci/devices/*/; do \
  [ "$(cat $d/class 2>/dev/null)" = "0x0b4000" ] && echo "$(basename $d) $(cat $d/vendor):$(cat $d/device)"; done
```

> 注：`/sys/class/kfd/kfd/topology/nodes/5` 中的编号 5 是本 Pod 的 HCU 节点号；
> 换 Pod 后需先按 `simd_count > 0` 定位真正的 HCU 节点（CPU 节点 `simd_count=0`）。
