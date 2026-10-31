# 本地磁盘与运行时容量实测报告

- 环境：`crdnotebook-2114695384420274177-scnjz2s4uc-68757`（SCNet Notebook 容器，ksai.scnet.cn:10312 / ssh 别名 `dcu_1026`）
- 采集时间：2026-10-26 22:10 - 22:35（容器内实测）
- 采集脚本：`/root/private_data/workspace/robo/scripts/probe3.sh`、`probe4.sh`

## 结论摘要

**没有 50G 的磁盘配额**。容器系统盘与个人数据盘都不存在 50G 上限（均已用写满探测实证）。
真正的硬约束是 **CPU 6 核 / 内存 115GiB（cgroup 强制）** 和 **无外网**。

## 1. 各类存储实测

| 挂载点 | 类型 | 容量口径 | 当前用量 | 写入权限 | 说明 |
|---|---|---|---|---|---|
| `/` | overlay (NVMe ext4) | 节点级 1.8T，剩余 449G | 容器自身 31G | rw | 与**同节点其他容器共享**，节点盘已用 73% |
| `/root/private_data` | parastor | 平台共享池 61P，剩余 16P | 用户 17G | rw | 等同 `/public/home/scnjz2s4uc`（同一挂载） |
| `/root/group_data` `/root/public_data` | parastor | 平台共享池 | — | **ro** | 只读 |
| `/dev/shm` | tmpfs | 115G | 0 | rw | 共享内存，重启丢失 |
| `/opt/hyhal` | ext4 (/dev/nvme0n1p3) | 节点级 | — | ro | DTK/Hyhal 运行时，只读 |

## 2. 配额验证方法与证据

### 2.1 系统盘 `/`：无 50G 上限
```
fallocate -l 60G /tmp/_probe60   -> 成功（exit=0），随后 truncate -s 0 清空并 unlink
```
容器根已用 31G 的情况下仍能一次性预留 60G → 文件系统层无 50G 配额。

### 2.2 个人盘 `/root/private_data`：无 50G 上限
parastor 不支持 `fallocate`（`Operation not supported`），改用真实写入探测：
```
dd if=/dev/zero of=tmp/_diskprobe bs=1M count=5120 oflag=append conv=notrunc   # 每块 5G
```
| 块 | 结果 | 累计文件大小 | 用户目录总量 |
|---|---|---|---|
| 1-8 | 全部 OK | 40G | 峰值 57G |

写入 40G 后仍未触顶，用户目录峰值 57G，探测文件已删除，目录回到 17G。
→ 个人盘同样是共享池，无用户级 50G 配额。

### 2.3 cgroup 实际限制（这才是真实配额）
```
cpu.cfs_quota_us / cpu.cfs_period_us = 600000 / 100000  -> 6 核
memory.limit_in_bytes                = 123480309760     -> 115 GiB
nproc = 32 / Mem 总量 503G           -> 宿主机视角，非本容器配额
```
与控制台「CPU 6核 / 内存 115GB / DCU 异构加速卡 64GB×1」一致。
`nproc`、`free -g` 显示的是宿主机 32 核/503G，**不能**当作本容器可用资源。

## 3. 其他实测结论

- **无外网**：`pypi.tuna.tsinghua.edu.cn`、`pypi.org`、`hf-mirror.com`、`github.com`、`huggingface.co` 全部不可达
  （`pip download tqdm` → `Errno 101 Network is unreachable`）。→ **不能 pip 安装新依赖**，只能用镜像内已有包。
- **必须加载 DTK 环境**：不 `source /opt/dtk/env.sh && source /opt/hyhal/env.sh` 时 `import torch` 报
  `ImportError: libgalaxyhip.so.5: cannot open shared object file`。
- 镜像内已有可用栈：torch `2.9.0+das.opt1.dtk2604`、sglang `0.5.12+das.opt1.dtk2604`、
  vllm `0.15.1+das.opt1.alpha.dtk2604`、transformers `5.3.0`、flash-attn `2.8.3+das`、
  triton `3.5.1+das`、DTK `6.3.26113`。
- 设备：1×HCU `K500SM_AI`，64GB 显存，capability (9,2)，bf16 可用。
- 模型权重：`RoboBrain2.5-8B-NV` 16.33GB（2 个分片齐全），`Qwen3VLForConditionalGeneration` / bf16 / max_pos 262144。
- 平台「保存开发环境」上限 15G（控制台提示），与运行时磁盘配额无关。

## 4. 落地约束（对 RoboBrain2.5-8B-NV 适配）

1. **所有写入走 `/root/private_data/workspace/robo`**：系统盘是节点共享盘（已用 73%），大文件（模型副本、
   venv、编译产物）一律落到个人盘；`env.sh` 已把 HF_HOME / TORCH_HOME / TRITON_CACHE_DIR / PIP_CACHE_DIR /
   TMPDIR 全部重定向到工作目录。
2. **不要指望装新包**：外网不通，只能使用镜像内已有的 sglang / vllm / transformers 组合。
3. **CPU 6 核是真实上限**：多进程数据预处理、并行 TP 都要按 6 核规划，别按 32 核估。
4. 模型权重已在个人盘内（`/public/home/scnjz2s4uc/...`），**无需复制**，避免额外 16G 占用。
