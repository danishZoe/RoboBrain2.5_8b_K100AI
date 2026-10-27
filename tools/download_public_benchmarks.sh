#!/usr/bin/env bash
# 下载三个上游公开评测集（CV-Bench / RefSpatial-Bench / TraceSpatial-Bench）
# 纪律：只下载公开数据；不伪造官方赛事评测集；缓存与产物全部落在工作目录内。
set -o pipefail
W=/root/private_data/workspace/robo
# shellcheck disable=SC1091
source "$W/tools/env_net.sh" >/dev/null 2>&1

export HF_HOME="$W/cache/hf"
export HF_HUB_CACHE="$W/cache/hf/hub"
export XDG_CACHE_HOME="$W/cache"
export TMPDIR="$W/tmp"
mkdir -p "$HF_HOME" "$W/third_party/benchmarks" "$W/tmp"

echo "===== 环境 ====="
python3 - <<'PY'
import sys
try:
    import huggingface_hub as H
    print("huggingface_hub", H.__version__)
except Exception as e:
    print("huggingface_hub 不可用:", e)
try:
    import datasets
    print("datasets", datasets.__version__)
except Exception as e:
    print("datasets 不可用:", e)
PY
echo "HF_HOME=$HF_HOME"

REPOS=(
  "nyu-visionx/CV-Bench"
  "BAAI/RefSpatial-Bench"
  "JingkunAn/TraceSpatial-Bench"
)

python3 - "$W" "${REPOS[@]}" <<'PY' 2>&1 | tee -a "$W/logs/bench_download.log"
import json, os, sys, time, traceback
W = sys.argv[1]
repos = sys.argv[2:]
from huggingface_hub import snapshot_download

manifest = {"schema": "robo.public-benchmark-inventory.v1",
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "note": ("上游公开评测集，用于补充精度证据；"
                     "**非赛事官方验收数据集**，不能替代官方 98% 门槛。"),
            "datasets": []}

for repo in repos:
    name = repo.split("/")[-1]
    dest = os.path.join(W, "third_party/benchmarks", name)
    print("\n=== downloading", repo, "->", dest, flush=True)
    rec = {"repo_id": repo, "local_dir": dest}
    t0 = time.time()
    try:
        p = snapshot_download(repo_id=repo, repo_type="dataset",
                              local_dir=dest,
                              cache_dir=os.environ.get("HF_HUB_CACHE"),
                              max_workers=4)
        rec["status"] = "ok"
        rec["snapshot_path"] = p
        print("   ok in %.1fs" % (time.time() - t0), flush=True)
    except Exception as e:
        rec["status"] = "failed"
        rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:400])
        print("   FAILED:", rec["error"], flush=True)
        traceback.print_exc(limit=2)
    # 统计体积与文件
    files, total = [], 0
    for root, _dirs, fns in os.walk(dest) if os.path.isdir(dest) else []:
        for fn in fns:
            fp = os.path.join(root, fn)
            try:
                sz = os.path.getsize(fp)
            except OSError:
                continue
            total += sz
            files.append({"path": os.path.relpath(fp, dest), "bytes": sz})
    files.sort(key=lambda x: -x["bytes"])
    rec["file_count"] = len(files)
    rec["total_bytes"] = total
    rec["largest_files"] = files[:12]
    print("   files=%d  bytes=%.1f MB" % (len(files), total / 1024 ** 2), flush=True)
    for f in files[:8]:
        print("     %10.2f MB  %s" % (f["bytes"] / 1024 ** 2, f["path"]), flush=True)
    manifest["datasets"].append(rec)

out = os.path.join(W, "samples/public_benchmark_inventory.json")
json.dump(manifest, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\nmanifest ->", out)
PY

echo
echo "===== 落盘结果 ====="
ls -l "$W/third_party/benchmarks/" 2>/dev/null
du -sh "$W/third_party/benchmarks"/* 2>/dev/null
