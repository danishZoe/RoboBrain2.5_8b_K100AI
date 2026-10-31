#!/bin/bash
W=/root/private_data/workspace/robo
cd "$W" || exit 1
echo "=== locate cv-bench tooling ==="
find . -maxdepth 3 -name "run_cv_bench.py" -o -maxdepth 3 -name "score_3d_chunked.py" -o -maxdepth 3 -name "common.py" 2>/dev/null | head -20
echo "--- dirs containing cvbench files ---"
find . -maxdepth 3 -name "*cv_bench*" -o -maxdepth 3 -name "*cvbench*" 2>/dev/null | head -30
echo "=== third_party tree ==="
ls -la third_party 2>/dev/null | head -20
echo "=== any official eval dir ==="
ls -la third_party/* 2>/dev/null | head -40
echo "=== modelscope dataset cache? ==="
ls -la ~/.cache/modelscope 2>/dev/null | head -10
find / -maxdepth 6 -type d -name "*cv-bench*" 2>/dev/null | head -10
find / -maxdepth 7 -type d -name "*cv_bench*" 2>/dev/null | head -10
