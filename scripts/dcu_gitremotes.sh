#!/bin/bash
W=/root/private_data/workspace/robo
cd "$W" || exit 1
echo "=== git remotes of upstream_eval repos ==="
for d in third_party/upstream_eval/*/; do
  [ -d "$d/.git" ] || continue
  echo "--- $d"
  git -C "$d" remote -v 2>/dev/null | head -2
  git -C "$d" log --oneline -1 2>/dev/null
  git -C "$d" rev-parse HEAD 2>/dev/null
done
echo
echo "=== FlagEvalMM structure ==="
ls third_party/upstream_eval/FlagEvalMM 2>/dev/null
echo "--- tasks dir ---"
ls third_party/upstream_eval/FlagEvalMM/tasks 2>/dev/null | head -30
echo "--- model_configs ---"
ls third_party/upstream_eval/FlagEvalMM/model_configs 2>/dev/null | head -20
echo "--- CV-Bench related files inside FlagEvalMM ---"
find third_party/upstream_eval/FlagEvalMM -iname "*cv*bench*" 2>/dev/null | head -20
echo "=== RoboTracer Evaluation ==="
ls third_party/upstream_eval/RoboTracer/Evaluation 2>/dev/null | head -20
echo "=== git status of FlagEvalMM (dirty?) ==="
git -C third_party/upstream_eval/FlagEvalMM status --short 2>/dev/null | head -20
