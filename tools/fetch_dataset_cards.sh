#!/usr/bin/env bash
# 用 huggingface_hub 取三个数据集的 README + RoboBrain2.5 模型卡（curl 直连 raw 会 401）
set -o pipefail
W=/root/private_data/workspace/robo
# shellcheck disable=SC1091
source "$W/tools/env_net.sh" >/dev/null 2>&1
export HF_HOME="$W/cache/hf"
export HF_HUB_CACHE="$W/cache/hf/hub"
mkdir -p "$W/third_party/upstream_eval/cards" "$W/logs"

python3 - "$W" <<'PY' 2>&1 | tee -a "$W/logs/upstream_cards.log"
import os, sys, traceback
from huggingface_hub import hf_hub_download
W = sys.argv[1]
OUT = os.path.join(W, "third_party/upstream_eval/cards")

targets = [
    ("nyu-visionx/CV-Bench", "dataset"),
    ("BAAI/RefSpatial-Bench", "dataset"),
    ("JingkunAn/TraceSpatial-Bench", "dataset"),
    ("BAAI/RoboBrain2.5-8B-NV", "model"),
]
for repo, kind in targets:
    print("\n============", repo, "(%s)" % kind, flush=True)
    try:
        p = hf_hub_download(repo_id=repo, filename="README.md", repo_type=kind,
                            cache_dir=os.environ.get("HF_HUB_CACHE"))
        txt = open(p, encoding="utf-8").read()
        name = repo.replace("/", "__") + ".md"
        dst = os.path.join(OUT, name)
        open(dst, "w", encoding="utf-8").write(txt)
        print("saved:", dst, len(txt), "chars", flush=True)
        # 打印与评测/提示词/答案相关的行
        for i, line in enumerate(txt.splitlines()):
            low = line.lower()
            if any(k in low for k in ("prompt", "evaluat", "metric", "accuracy",
                                      "iou", "answer", "question", "format",
                                      "bench", "split", "subset", "score")):
                print("  %4d| %s" % (i + 1, line[:220]))
    except Exception as e:
        print("FAILED:", type(e).__name__, str(e)[:300], flush=True)
        traceback.print_exc(limit=1)
PY
