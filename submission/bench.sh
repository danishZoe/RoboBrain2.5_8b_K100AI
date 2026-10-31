#!/usr/bin/env bash
# 首次提交必跑：结果即该「模型 × 卡型 × 框架 × 精度」的性能基准；
# 后续优化按同一负载复测对比。
#
# 负载固定：输入 4K / 16K × 输出 1K × 并发 1 / 8 / 64，共 6 档；
# 每档连测两次，第一次预热，第二次计入成绩。
set -uo pipefail   # 不加 -e：某一档失败时记录下来，继续跑其余档
source ./env.sh
cd "$REPO_DIR"

BENCH_MODE=${BENCH_MODE:-text}   # text：纯文本；image：多模态理解，每个请求带 1 张图
IMAGE_RES=${IMAGE_RES:-1080p}    # BENCH_MODE=image 时的图片分辨率，以分题材料为准
mkdir -p bench
# 结果文件是追加写入；只在跑全部 6 档时清空旧结果，
# 按档拆分执行（BENCH_TIERS）时保留前一次的结果以便最终汇总。
if [ -z "${BENCH_TIERS:-}" ]; then
  rm -f bench/round*.jsonl
fi

BENCH=sglang.benchmark.serving                 # SGLang 压测客户端，SGLang / vLLM 服务通用
python -c "import $BENCH" 2>/dev/null || BENCH=sglang.bench_serving   # 旧版 sglang 的模块名

# 官方客户端的 image 数据集需要用 AutoProcessor.from_pretrained(model_id) 加载 processor，
# 而 model_id 取自 --served-model-name（服务名，不是路径），节点无外网会失败。
# 这里在**工作目录内的 HF 缓存**里为服务名建一个指向本地权重的快照（只读软链接），
# 使 AutoProcessor 能离线解析；不触碰模型目录本身。
HF_SNAP="$HF_HOME/hub/models--${SERVED_NAME}/snapshots/main"
mkdir -p "$(dirname "$HF_SNAP")"
if [ ! -e "$HF_SNAP" ]; then
  ln -sfn "$MODEL_PATH" "$HF_SNAP"
  echo "[bench] linked $HF_SNAP -> $MODEL_PATH (offline processor lookup)"
fi

if [ "$BENCH_MODE" = image ]; then
  DATA=(--dataset-name image --image-count 1 --image-resolution "$IMAGE_RES")
else
  DATA=(--dataset-name random-ids)            # 随机 token，无需下载数据集
fi

for ILEN in 4096 16384; do
  for C in 1 8 64; do
    # BENCH_TIERS 可按 "in:conc" 过滤档位，便于把 6 档拆成多次执行；
    # 不设置时行为与官方脚本完全一致（6 档全跑）。
    if [ -n "${BENCH_TIERS:-}" ]; then
      case ",$BENCH_TIERS," in
        *",$ILEN:$C,"*) ;;
        *) continue ;;
      esac
    fi
    N=$(( C * 2 > 8 ? C * 2 : 8 ))            # 请求数：并发的 2 倍，至少 8 个
    for R in 1 2; do
      TAG="round${R}_${ILEN}_1024_c${C}"
      echo "[bench] $TAG"
      # 本镜像的 SGLang 压测客户端（0.5.12）在 image 数据集上只接受 sglang 系列后端，
      # 对 OpenAI 兼容的 vLLM 服务需要在**进程内**把后端映射到 sglang-oai-chat 语义
      # （服务端自己套 chat template）。适配脚本不改公共 site-packages。
      if [ "$BENCH_MODE" = image ] && [ "$FRAMEWORK" = vllm ]; then
        RUNNER=(python3 "$REPO_DIR/scripts/bench_serving_vllm_image.py")
      else
        RUNNER=(python -m "$BENCH")
      fi
      "${RUNNER[@]}" \
        --backend "$FRAMEWORK" --host 127.0.0.1 --port "$PORT" \
        --model "$SERVED_NAME" --tokenizer "$MODEL_PATH" "${DATA[@]}" \
        --random-input-len "$ILEN" --random-output-len 1024 --random-range-ratio 1 \
        --max-concurrency "$C" --num-prompts "$N" --seed 1 \
 \
        --output-file "bench/round${R}.jsonl" > "bench/${TAG}.log" 2>&1 \
        || echo "[bench] FAIL $TAG，见 bench/${TAG}.log；不支持的档位写进报告「注意事项」"
    done
  done
done

# 第二次的结果汇总成 Markdown 表，直接贴进报告「7. 性能参考」
python - <<'EOF' | tee bench/summary.md
import json
cols = ["random_input_len", "random_output_len", "max_concurrency",
        "completed", "duration",
        "request_throughput", "output_throughput", "total_throughput",
        "mean_ttft_ms", "p99_ttft_ms",
        "mean_tpot_ms", "p99_tpot_ms", "mean_e2e_latency_ms"]
print("| " + " | ".join(cols) + " |")
print("|" + " --- |" * len(cols))
for line in open("bench/round2.jsonl", encoding="utf-8"):
    r = json.loads(line)
    print("| " + " | ".join(f"{r.get(c, '-'):.2f}" if isinstance(r.get(c), float) else str(r.get(c, "-")) for c in cols) + " |")
EOF

echo "[bench] done; 提交 bench/round1.jsonl、bench/round2.jsonl、bench/*.log 与 bench/summary.md"
