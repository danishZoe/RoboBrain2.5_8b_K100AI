#!/usr/bin/env bash
# 查看稳定性检查的逐题差异幅度
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
python3 - <<'PY'
import json, os
p = "/root/private_data/workspace/robo/outputs/bench_public/stability/stability.json"
d = json.load(open(p, encoding="utf-8"))
for k in ("refspatial", "tracespatial", "cvbench"):
    s = d.get(k)
    if not s:
        continue
    print("\n===== %s  n=%s 一致=%s 变化率=%s 全OK=%s =====" %
          (k, s["n"], s["identical_raw"], s["change_ratio"], s["all_states_ok"]))
    for r in s.get("samples", []):
        a = (r.get("run1") or "")[:90]
        b = (r.get("run2") or "")[:90]
        same = r.get("identical_raw")
        flag = "同" if same else ">>>"
        print("  %s %-28s run1=%s" % (flag, r.get("sample_id"), a))
        if not same:
            print("      %-28s run2=%s" % ("", b))
            print("      parsed相同=%s" % r.get("identical_parsed"))
PY
