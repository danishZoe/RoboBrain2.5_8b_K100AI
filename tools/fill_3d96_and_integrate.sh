#!/usr/bin/env bash
set -uo pipefail
WS=/root/private_data/workspace/robo
cd "$WS"

echo "=== 把 96 题兜底 3D 产物补进 bench 根目录（生产判分若已完成则不覆盖）==="
for V in official upstream; do
  for SRC in bench_public bench_public_noneager; do
    SRCF="$WS/outputs/bench_public_3d96/$SRC/tracespatial/$V/scores_3dchunk_3D.json"
    DSTF="$WS/outputs/$SRC/tracespatial/$V/scores_3dchunk_3D.json"
    if [ ! -f "$SRCF" ]; then
      echo "  缺源：$SRCF"
      continue
    fi
    if [ -f "$DSTF" ]; then
      echo "  已存在（不覆盖）：$DSTF"
      continue
    fi
    cp "$SRCF" "$DSTF"
    echo "  复制：$SRCF -> $DSTF"
  done
done

echo
echo "=== 重跑集成 ==="
bash tools/integrate_ts_results.sh 2>&1 | tail -30
