#!/usr/bin/env bash
# 1) open3d 安装进度 / 2) FlagEvalMM 里 cv_bench 任务目录结构
set -o pipefail
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1

echo "===== A) envs/pylibs 现状 ====="
ls -la envs/pylibs 2>/dev/null | head -n 30
echo "--- 是否有 open3d ---"
ls -d envs/pylibs/open3d* 2>/dev/null || echo "no open3d dir yet"
du -sh envs/pylibs 2>/dev/null

echo
echo "===== B) pip 进程 ====="
ps -ef | grep -E 'pip install' | grep -v grep | head -n 5 || echo "no pip running"

echo
echo "===== C) FlagEvalMM cv_bench 目录 ====="
FE=third_party/upstream_eval/FlagEvalMM
ls "$FE" | head -n 30
echo "--- 找 cv_bench ---"
find "$FE" -maxdepth 4 -iname '*cv_bench*' -o -maxdepth 4 -iname '*cvbench*' 2>/dev/null | head -n 20

echo
echo "===== D) FlagEvalMM 评测入口文件 ====="
find "$FE" -maxdepth 3 -name 'evaluate*.py' 2>/dev/null | head -n 30
find "$FE" -maxdepth 3 -name '*.md' -path '*cv*' 2>/dev/null | head -n 10

echo
echo "===== E) CV-Bench README 全文（前 100 行）====="
head -n 100 third_party/benchmarks/CV-Bench/README.md
