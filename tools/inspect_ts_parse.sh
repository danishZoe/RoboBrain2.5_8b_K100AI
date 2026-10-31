#!/usr/bin/env bash
# 检查 TraceSpatial 官方变体 2D 预测的原始输出：为什么我们的解析器判 parse_failed
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONPATH="$WS/envs/pylibs_core:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

python3 - <<'PY'
import json, os, sys
sys.path.insert(0, "/root/private_data/workspace/robo/submission/bench_public")
import common as C

for root, label in (("/root/private_data/workspace/robo/outputs/bench_public", "基线"),
                     ("/root/private_data/workspace/robo/outputs/bench_public_noneager", "候选")):
    for variant in ("official", "upstream"):
        for task in ("2D", "3D"):
            pp = os.path.join(root, "tracespatial", variant, "predictions", "%s.jsonl" % task)
            if not os.path.exists(pp):
                print("%s %s/%s: 缺 predictions" % (label, variant, task)); continue
            recs = [json.loads(l) for l in open(pp, encoding="utf-8") if l.strip()]
            st = {}
            for r in recs:
                st[r.get("terminal_state")] = st.get(r.get("terminal_state"), 0) + 1
            empty = sum(1 for r in recs if not (r.get("raw_output") or "").strip())
            # 解析器能否解析
            okparse = 0
            for r in recs:
                try:
                    pts = C.parse_points_1000(r.get("raw_output") or "", "trajectory")
                    if pts is not None and len(pts) > 0:
                        okparse += 1
                except Exception:
                    pass
            print("%s %-8s %-3s: n=%d states=%s raw_empty=%d 我们的解析器可解析=%d" % (
                label, variant, task, len(recs), st, empty, okparse))
            # 官方输入 jsonl 里的 model_prediction
            oj = os.path.join(root, "tracespatial", variant, "outputs",
                              "%s_%s.jsonl" % (C.__dict__.get("_x", "") or
                                              ("RoboTracer_RoboBrain2.5-8B-NV" if variant == "official"
                                               else "RoboTracer_RoboBrain2.5-8B-NV_upstream"), task))
            if os.path.exists(oj):
                lines = [json.loads(l) for l in open(oj, encoding="utf-8") if l.strip()]
                nempty = sum(1 for a in lines if not (a.get("model_prediction") or "").strip())
                print("        官方输入 %s: n=%d model_prediction 为空=%d" % (
                    os.path.basename(oj), len(lines), nempty))
                if task == "2D" and variant == "official":
                    for a in lines[:2]:
                        mp = (a.get("model_prediction") or "")
                        print("        --- 样例 model_prediction（前 400 字符）---")
                        print("        " + mp[:400].replace("\n", "\n        "))
            break   # 每组只看基线/候选的 official，upstream 另起
PY
