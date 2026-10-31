#!/bin/bash
W=/root/private_data/workspace/robo
cd "$W/third_party/upstream_eval/FlagEvalMM/tasks/cv_bench" || exit 1
echo "=== DCU's cv_bench task files ==="
ls -la
echo "=== sha256 (DCU) ==="
sha256sum *.py
echo "=== diff process.py vs evaluate.py summary ==="
wc -l *.py
echo "=== process.py head 80 ==="
head -80 process.py
