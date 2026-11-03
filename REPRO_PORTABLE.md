# K100AI 复现入口（vLLM）

代码从 GitLab 获取；基础镜像需包含 DTK、PyTorch、vLLM、SGLang 和 `/opt/robo-image-v1/pylibs-clean/qwen_vl_utils`。模型权重单独挂载，`MODEL_PATH` 可覆盖。不得把历史调优收益或输出缩短作为本次实测收益。

在可写工作目录中准备环境：

```bash
export WORK_DIR="$PWD/.runtime"
export MODEL_PATH=/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV
export VLLM_NO_USAGE_STATS=1
bash /opt/robo-image-v1/prepare.sh "$PWD" "$WORK_DIR"
bash run.sh
bash smoke.sh
BENCH_MODE=image bash bench.sh
```

服务监听 127.0.0.1:30000；先检查设备占用，避免干扰其他任务。默认压测为 4096/16384 × C1/C8/C64，输出预算 1024，每档两轮，请求数为 max(8, 2×并发)。数据默认写入工作目录下的新 `outputs/bench_*`；可通过 `BENCH_OUT_DIR` 指定全新目录。客户端采用仓库的 `scripts/bench_serving_vllm_image.py`：对镜像中的 SGLang 客户端进行进程内后端兼容适配，不代表赛事方提供的评测脚本。

只检查运行链路（不代表正式吞吐或真实 C8 并发）：

```bash
BENCH_MODE=image BENCH_TIERS=4096:1 BENCH_NUM_PROMPTS=1 BENCH_OUTPUT_LEN=32 bash bench.sh
```

`MODEL_PATH`、工作目录、系统盘内容以及评测方挂载条件必须与其环境匹配。准备脚本位于镜像系统盘，不在 GitLab 中；缓存、样例和调优库在工作目录重建。尚未在保存后的新实例验证。`smoke.sh` 含异常输入测试，服务日志可能出现预期的 400 错误。报告的公开精度指标应按各评测集自身口径解释。
