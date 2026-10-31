#!/usr/bin/env bash
# 服务起来后先跑它：几秒内确认接口可用、输出正常，再跑耗时的精度与压测。
#
# 覆盖验收标准 §5 的全部服务级检查项：
#   健康检查 / 模型列表 / 单轮图文 / 多轮图文 / 流式 / 四类任务 / 多图 / 异常输入
#
# 与官方 smoke.sh 的差异：官方是纯文本问答；本模型是多模态，
# 因此以「图文请求」为最小验证单元，并额外覆盖四类任务与异常输入。
set -uo pipefail   # 不加 -e：某一项失败记录下来，继续跑其余项
source ./env.sh
cd "$REPO_DIR"

URL="http://127.0.0.1:$PORT"
mkdir -p samples
IMG="${WORK_DIR}/samples/images/official_affordance.jpg"
[ -f "$IMG" ] || IMG="$REPO_DIR/samples/images/official_affordance.jpg"

# 图片按官方 qwen_vl_utils 规范化后以 PNG 上传，保证服务端看到的像素
# 与官方推理管线一致（报告第 3.3 节 / 报告 09 第 4 节）。
python3 "$REPO_DIR/scripts/make_data_url.py" "$IMG" > /tmp/robo_img_url.txt 2>/dev/null || true
IMG_URL=""
if [ -s /tmp/robo_img_url.txt ]; then
  IMG_URL="$(cat /tmp/robo_img_url.txt)"
fi
if [ -z "$IMG_URL" ]; then
  echo "== 图片数据 URL 生成失败，改用原图（会与官方像素不一致）"
  base64 -w0 "$IMG" > /tmp/robo_img_url_b64.txt
  printf 'data:image/jpeg;base64,' > /tmp/robo_img_url.txt
  cat /tmp/robo_img_url_b64.txt >> /tmp/robo_img_url.txt
  IMG_URL="$(cat /tmp/robo_img_url.txt)"
fi

echo "== 健康检查"
curl -sf --noproxy "*" "$URL/health" -o /dev/null -w "HTTP %{http_code}\n"

echo "== 模型列表"
curl -sf --noproxy "*" "$URL/v1/models"

echo
echo "== 单轮图文"
python3 - "$SERVED_NAME" /tmp/robo_img_url.txt /tmp/robo_payload.json <<'PY'
import json, sys
name, url_file, out = sys.argv[1], sys.argv[2], sys.argv[3]
url = open(url_file, encoding="utf-8").read().strip()
payload = {"model": name, "temperature": 0, "max_tokens": 256,
           "messages": [{"role": "user", "content": [
               {"type": "image_url", "image_url": {"url": url}},
               {"type": "text", "text": "What is shown in this image?"}]}]}
open(out, "w", encoding="utf-8").write(json.dumps(payload))
PY
curl -sf --noproxy "*" "$URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @/tmp/robo_payload.json

echo
echo "== 多轮图文（第二轮问「杯子里是什么？」）"
python3 - "$SERVED_NAME" /tmp/robo_img_url.txt /tmp/robo_payload2.json <<'PY'
import json, sys
name, url_file, out = sys.argv[1], sys.argv[2], sys.argv[3]
url = open(url_file, encoding="utf-8").read().strip()
payload = {"model": name, "temperature": 0, "max_tokens": 256,
           "messages": [
               {"role": "user", "content": [
                   {"type": "image_url", "image_url": {"url": url}},
                   {"type": "text", "text": "What is shown in this image?"}]},
               {"role": "assistant", "content": "A white mug on a table."},
               {"role": "user", "content": "What colour is the mug?"}]}
open(out, "w", encoding="utf-8").write(json.dumps(payload))
PY
curl -sf --noproxy "*" "$URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @/tmp/robo_payload2.json

echo
echo "== 流式（末行应为 data: [DONE]）"
python3 - "$SERVED_NAME" /tmp/robo_img_url.txt /tmp/robo_payload3.json <<'PY'
import json, sys
name, url_file, out = sys.argv[1], sys.argv[2], sys.argv[3]
url = open(url_file, encoding="utf-8").read().strip()
payload = {"model": name, "temperature": 0, "max_tokens": 256, "stream": True,
           "messages": [{"role": "user", "content": [
               {"type": "image_url", "image_url": {"url": url}},
               {"type": "text", "text": "用一句话介绍国产异构加速器。"}]}]}
open(out, "w", encoding="utf-8").write(json.dumps(payload))
PY
curl -sf --noproxy "*" "$URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @/tmp/robo_payload3.json | tail -n 3

echo
echo "== 四类任务 + 多图 + 异常输入"
python3 "$REPO_DIR/scripts/smoke_service.py" --url "$URL" \
  --model "$SERVED_NAME" --out-dir "$WORK_DIR/outputs/phase5_smoke"

echo "== smoke done"
