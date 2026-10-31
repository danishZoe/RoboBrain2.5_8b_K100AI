#!/usr/bin/env bash
# 查 TraceSpatial 2D 分片判分的 stderr：官方脚本为什么没输出指标
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONWARNINGS=ignore
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

python3 - <<'PY'
import json, os
P = "/root/private_data/workspace/robo/outputs/bench_public/tracespatial/official/scores_3dchunk_2D.json"
d = json.load(open(P, encoding="utf-8"))
cs = d["per_chunk"]
print("分片数:", len(cs))
c = cs[0]
print("chunk0 returncode:", c.get("returncode"))
print("chunk0 stdout_len(未存): 见下")
print("chunk0 stderr_tail:")
print(c.get("stderr_tail"))
print("chunk0 keys:", sorted(c.keys()))
print("chunk0 counts:", c.get("counts"))
print("chunk0 metrics_2d:", c.get("metrics_2d"))
print("chunk0 start_in_mask:", c.get("start_in_mask"))

# 直接手工跑一次官方 2D CLI 看 stdout
print("\n" + "=" * 70)
print("手工跑一次官方 2D CLI（单分片目录）")
print("=" * 70)
import subprocess, sys
WS = "/root/private_data/workspace/robo"
D = os.path.join(WS, "tmp/ts3d/bench_public_tracespatial_official_2D/c0000")
print("工作目录:", D, os.path.isdir(D))
if os.path.isdir(D):
    oj = os.path.join(D, "outputs")
    print("outputs 目录内容:", os.listdir(oj) if os.path.isdir(oj) else "缺")
    if os.path.isdir(oj):
        for f in os.listdir(oj):
            p = os.path.join(oj, f)
            print("  ", f, os.path.getsize(p), "bytes")
            with open(p, encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if i >= 1:
                        break
                    print("    首行前 200 字符:", line[:200])
    env = dict(os.environ)
    env["PYTHONPATH"] = (WS + "/envs/pylibs_o3d:" + WS + "/envs/pylibs:" + WS +
                         "/envs/pylibs_core:" + WS + "/third_party/upstream_eval/RoboTracer/Evaluation")
    env["PYTHONWARNINGS"] = "ignore"
    r = subprocess.run([sys.executable,
                        WS + "/third_party/upstream_eval/RoboTracer/Evaluation/summarize_acc.py",
                        "--model_name", "RoboTracer_RoboBrain2.5-8B-NV",
                        "--task_name", "2D"],
                       cwd=D, env=env, capture_output=True, text=True, timeout=600)
    print("\n--- returncode:", r.returncode)
    print("--- stdout ---")
    print(r.stdout[-3000:])
    print("--- stderr ---")
    print(r.stderr[-3000:])
PY
