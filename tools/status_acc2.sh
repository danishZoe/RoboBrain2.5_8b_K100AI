#!/usr/bin/env bash
# 状态速查：TraceSpatial / RefSpatial / CV-Bench 现有判分结果
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONPATH="$WS/envs/pylibs_core:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

python3 - <<'PY'
import json, os

KEYS = ("metrics_2d", "start_in_mask", "end_in_bbox", "metrics_3d",
        "start_distance", "end_distance", "no_collision_rate",
        "start_success_rate", "end_success_rate", "overall_success_rate")

def dump_trace(root, label):
    sp = os.path.join(root, "tracespatial", "summary.json")
    print("\n########## %s ##########" % label)
    if not os.path.exists(sp):
        print("  (缺 %s)" % sp); return
    d = json.load(open(sp, encoding="utf-8"))
    for v, vin in d.get("variants", {}).items():
        for t, tin in vin.get("tasks", {}).items():
            sc = tin.get("scores")
            print("  %-9s / %-3s  n=%s  terminal=%s" % (
                v, t, tin.get("n"), tin.get("terminal_states")))
            if not sc:
                print("      -> 尚无判分结果")
                continue
            for k in KEYS:
                if sc.get(k) is not None:
                    print("      %-22s %s" % (k, sc[k]))
            print("      collision_warnings=%s skip_depthless=%s open3d=%s" % (
                sc.get("collision_warnings"), sc.get("skip_depthless"),
                sc.get("open3d_available")))

dump_trace("/root/private_data/workspace/robo/outputs/bench_public", "基线 eager")
dump_trace("/root/private_data/workspace/robo/outputs/bench_public_noneager", "候选 非eager")

print("\n########## 已有 3D 分片判分产物 ##########")
for root in ("/root/private_data/workspace/robo/outputs/bench_public",
             "/root/private_data/workspace/robo/outputs/bench_public_noneager"):
    p = os.path.join(root, "tracespatial")
    if not os.path.isdir(p):
        continue
    for v in sorted(os.listdir(p)):
        sp = os.path.join(p, v, "scores_3dchunk_3D.json")
        if os.path.exists(sp):
            print("  ", sp)

print("\n########## RefSpatial 现状 ##########")
for root, label in (("/root/private_data/workspace/robo/outputs/bench_public", "基线"),
                     ("/root/private_data/workspace/robo/outputs/bench_public_noneager", "候选")):
    sp = os.path.join(root, "refspatial", "summary.json")
    if not os.path.exists(sp):
        print("  %s: (缺)" % label); continue
    d = json.load(open(sp, encoding="utf-8"))
    for v, vin in d.get("variants", {}).items():
        for s, sin in vin.get("splits", {}).items():
            o = sin.get("scores", {}).get("official_cli", {})
            print("  %s %-16s %-12s acc=%s evaluated=%s/%s" % (
                label, v, s, o.get("official_accuracy"),
                o.get("official_evaluated"), o.get("official_total")))
PY
