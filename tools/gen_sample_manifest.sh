#!/usr/bin/env bash
# =============================================================================
# 生成提交包的样例清单 submission/samples/MANIFEST.md
#   - 逐个样例的 sha256 与字节数
#   - 与 samples/upstream_provenance.json 里记录的 upstream 校验值对拍
#   - 结果写进 MANIFEST.md（校验通过与否都写，不掩盖）
#
# 用途：验收标准 §9 要求「固定样例及输出：样例来源、校验值、原文、解析值、
#       坐标/轨迹叠加图」。来源与校验值在 upstream_provenance.json，
#       本脚本把两者合成提交包里的一份可复核清单。
#
# 用法: bash tools/gen_sample_manifest.sh
# =============================================================================
set -uo pipefail
W=/root/private_data/workspace/robo
cd "$W" || exit 1

SUB="$W/submission/samples"
mkdir -p "$SUB"
cp -f "$W/samples/upstream_provenance.json" "$SUB/upstream_provenance.json"

python3 - "$W" "$SUB" <<'PY'
import hashlib, json, os, sys

W, SUB = sys.argv[1], sys.argv[2]
prov_path = os.path.join(W, "samples", "upstream_provenance.json")
prov = json.load(open(prov_path, encoding="utf-8"))
recorded = {i["local"]: i for i in prov.get("images", [])}

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

rows, mismatch = [], []
checked = 0
for name in sorted(os.listdir(os.path.join(SUB, "images"))):
    rel = "samples/images/" + name
    p = os.path.join(SUB, "images", name)
    d = sha256(p)
    rec = recorded.get(rel)
    if rec:
        checked += 1
        ok = "一致" if rec.get("sha256") == d else "**不一致**"
        if rec.get("sha256") != d:
            mismatch.append(rel)
        up = rec.get("upstream_path", "-")
        head = "是" if rec.get("upstream_head_sha256_match") else "否"
    else:
        ok, up, head = "（无 upstream 记录）", "-", "-"
    rows.append((rel, os.path.getsize(p), d, up, ok, head))

for name in ("tasks_official.json", "tasks_multi_image.json"):
    p = os.path.join(SUB, name)
    if os.path.exists(p):
        rows.append(("samples/" + name, os.path.getsize(p), sha256(p), "-", "本地固定样例", "-"))

out = []
out.append("## 固定样例清单（来源 + 校验值）\n")
out.append("上游仓库：`%s` @ `%s`（%s）" % (
    prov["upstream"]["repo"], prov["upstream"]["commit"][:12], prov["upstream"]["commit_date"]))
out.append("提示词/正则一致性：`%s`（%s）\n" % (
    prov.get("inference_py_parity", {}).get("result", "-"),
    prov.get("inference_py_parity", {}).get("note", "")))
out.append("| 文件 | 字节 | sha256 | upstream 路径 | 本地校验 | 与 upstream HEAD 一致 |")
out.append("|---|---|---|---|---|---|")
for rel, size, d, up, ok, head in rows:
    out.append("| `%s` | %d | `%s` | `%s` | %s | %s |" % (rel, size, d, up, ok, head))
out.append("")
out.append("**校验结论**：%s" % (
    "%d 个图像样例与 upstream 记录的 sha256 **逐位一致**，样例未被改动。" % checked
    if not mismatch else
    "**%d 个样例校验不一致**：%s —— 需复查后再提交。" % (len(mismatch), ", ".join(mismatch))))
out.append("")
out.append("配套产物：")
out.append("")
out.append("- 任务提示词与解析正则：`submission/scripts/robo_infer.py`（逐字复制自 upstream `inference.py:110-124`）")
out.append("- 逐题原文与解析值：`outputs/phase3_baseline/`、`outputs/phase5_smoke/`")
out.append("- 坐标/轨迹叠加图：`outputs/annotated/`（白底叠加 bbox、point、trajectory）")
out.append("- 服务侧同口径复跑：`outputs/vllm_*`、`outputs/phase5_*`")
out.append("")

path = os.path.join(SUB, "MANIFEST.md")
with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("[manifest] wrote %s (%d entries, %d mismatch)" % (path, len(rows), len(mismatch)))
for r in rows:
    print("   %-42s %s" % (r[0], r[2][:16] + "..."))
PY
