#!/usr/bin/env bash
WS=/root/private_data/workspace/robo
cd "$WS" || exit 1
echo "=== tools/ 规模 ==="
du -sh tools
ls -1 tools | wc -l
echo
echo "=== submission README 里引用的 tools/* 路径 ==="
grep -oE 'tools/[A-Za-z0-9_./-]+' submission/README.md | sort -u
echo
echo "=== submission/bench_public 内容 ==="
ls -1 submission/bench_public
