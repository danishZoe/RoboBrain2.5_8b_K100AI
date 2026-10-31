#!/usr/bin/env bash
# 最小复现：Popen(poll 循环) + communicate 是否丢 stdout
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
export PYTHONWARNINGS=ignore
set +u
source /opt/dtk/env.sh >/dev/null 2>&1
set -u

python3 - <<'PY'
import os, subprocess, sys, glob

WS = "/root/private_data/workspace/robo"
D = os.path.join(WS, "tmp/ts3d/tracespatial_official_2D/c0000")
print("工作目录:", D, "存在:", os.path.isdir(D))
if not os.path.isdir(D):
    cands = sorted(glob.glob(os.path.join(WS, "tmp/ts3d/*/c0000")))
    print("候选:", cands[:5])
    if cands:
        D = cands[0]
print("使用:", D)
print("outputs:", os.listdir(os.path.join(D, "outputs")) if os.path.isdir(os.path.join(D, "outputs")) else "缺")

env = dict(os.environ)
env["PYTHONPATH"] = (WS + "/envs/pylibs_o3d:" + WS + "/envs/pylibs:" + WS +
                     "/envs/pylibs_core:" + WS + "/third_party/upstream_eval/RoboTracer/Evaluation")
env["PYTHONWARNINGS"] = "ignore"
cmd = [sys.executable,
       WS + "/third_party/upstream_eval/RoboTracer/Evaluation/summarize_acc.py",
       "--model_name", "RoboTracer_RoboBrain2.5-8B-NV", "--task_name", "2D"]

print("\n=== 方式A：Popen + poll 循环 + communicate（我调度器当前的做法）===")
p = subprocess.Popen(cmd, cwd=D, env=env, stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE, text=True)
while p.poll() is None:
    pass
out, err = p.communicate()
print("rc =", p.returncode)
print("stdout 长度 =", len(out))
print("stdout 内容 >>>")
print(out)
print("stderr 长度 =", len(err))

print("\n=== 方式B：subprocess.run ===")
r = subprocess.run(cmd, cwd=D, env=env, capture_output=True, text=True, timeout=900)
print("rc =", r.returncode)
print("stdout 长度 =", len(r.stdout))
print("stdout 内容 >>>")
print(r.stdout)
print("stderr 长度 =", len(r.stderr))
PY
