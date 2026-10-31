# platform/ —— 需要用户从超算互联网控制台补齐的证据

本目录的证据**节点访问不到**（Pod 内无法打开平台控制台），只能由用户截图后放入本目录。
以下 4 类截图是官方技术报告模板第 2 节「验证环境」要求的内容，README 第 2 节已留好占位。

| 文件名（建议） | 需要截的内容 | 对应报告位置 |
|---|---|---|
| `01_notebook_create.png` | Notebook 创建页：实例规格（CPU / 内存 / 系统盘 / 文件存储）、加速卡型号与数量、镜像选择 | `submission/README.md` §2「超算互联网平台」表 |
| `02_env_check.png` | 环境自检页面（卡型、驱动、运行时版本） | 同上 |
| `03_model_detail.png` | 模型管理页：模型名称、路径、精度、授权状态 | 同上 |
| `04_service_mapping.png` | 服务/代理映射页：对外访问地址与端口映射 | 同上 |

## 已补齐：代码托管链接

| 项 | 值 |
|---|---|
| 代码仓库（SSH） | `ssh://git@developer.sourcefind.cn:2222/g_exhyd6h6/robobrain2.5_8b_k100ai.git` |
| 代码仓库（网页） | <https://developer.sourcefind.cn/codes/g_exhyd6h6/robobrain2.5_8b_k100ai> |
| 分支 / 标签 | `main` = 最优版；`baseline-submission` = 首提基线（eager） |
| 提交记录 | 见仓库 `git log`，与 `reports/25_tuning_delivery_20261031.md` §7 的 revision 一致 |

> **镜像链接仍待补**：赛事要求代码与**镜像**分别提交到指定托管平台。
> 镜像名 `jupyter-sglang5.12-ubuntu22.04-dtk26.04-py3.10-qwen3.6:v1.0` 需要补全
> 完整地址与 digest；该动作需要平台账号，节点无外网，无法代替上传。

## 还需要文字补齐的两项

1. **区域**（如 `ksai.scnet.cn` 对应的区域名/区域 ID）
2. **服务代理映射地址**（对外可访问的 URL 或 IP:Port）

这两项直接写在 `submission/README.md` §2 的表格里，截图无法替代。

## 补齐后的自检

```bash
cd /root/private_data/workspace/robo
ls -l platform/            # 确认 4 张截图都在
```

提交包自检脚本 `tools/check_submission.sh` 不会因为本目录为空而失败 ——
这些材料属于**用户侧待补项**，在报告里已如实标注为「待补」，不用伪造截图占位。
