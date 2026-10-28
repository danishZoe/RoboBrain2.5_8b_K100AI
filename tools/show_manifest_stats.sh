#!/usr/bin/env bash
# 从 manifest 提取 CV-Bench 图像尺寸分布（协议限制 L1 的证据）
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
python3 - <<'PY'
import json
d = json.load(open("outputs/bench_public/manifest/data_manifest.json", encoding="utf-8"))
cv = d["datasets"]["cvbench"]
dec = cv["decode"]
print("CV-Bench 条目数:", cv["n"], " 唯一ID:", cv["unique_question_ids"],
      " 解码失败:", dec["failed"])
print("分组:", cv["groups"])
hist = dec["size_histogram"]
print("\n图像尺寸分布（Top 15）:")
for k, v in list(hist.items())[:15]:
    w, h = k.split("x")
    px = int(w) * int(h)
    print("  %-12s %5d  (%.2f MP) %s" % (k, v, px / 1e6,
          "  > 501760 (transformers 基线 max_pixels)" if px > 501760 else ""))
tot = sum(hist.values())
over = dec["max_pixels_over_transformers_budget"]
print("\n超过 transformers 基线 max_pixels=501760 的图像: %d / %d = %.1f%%" %
      (over, tot, 100.0 * over / tot))
print("最大像素图:", max(((int(k.split('x')[0]) * int(k.split('x')[1])), k)
                    for k in hist), "(px, size)")
print("\nRefSpatial:", {k: {kk: vv for kk, vv in v.items()
                        if kk in ("n", "unique_ids", "rgb_decode_failed",
                                  "mask_decode_failed", "rgb_mask_size_mismatch")}
                        for k, v in d["datasets"]["refspatial"]["splits"].items()})
ts = d["datasets"]["tracespatial"]
print("\nTraceSpatial:", {k: ts[k] for k in ("n", "unique_ids", "duplicate_ids")},
      "| decode_failed:", len(ts["decode_failed"]),
      "| size mismatch:", len(ts["image_depth_mask_size_mismatch"]))
print("  分组:", ts["groups"])
print("\n文件 SHA256:")
for f in cv["files"]:
    print("  %s  %s" % (f["sha256"][:16], f["path"]))
print("  %s  %s" % (ts["trajectory_dataset_json"]["sha256"][:16],
                    ts["trajectory_dataset_json"]["path"]))
PY
