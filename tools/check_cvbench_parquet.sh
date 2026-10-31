#!/usr/bin/env bash
set -o pipefail
W=/root/private_data/workspace/robo
echo "===== CV-Bench parquet 结构（图像是否内嵌） ====="
python3 - <<'PY'
import pyarrow.parquet as pq
for f in ("test_2d.parquet", "test_3d.parquet"):
    p = "/root/private_data/workspace/robo/third_party/benchmarks/CV-Bench/" + f
    t = pq.read_table(p)
    print(f, "rows =", t.num_rows)
    print("  cols:", t.schema.names)
    r = t.slice(0, 1).to_pylist()[0]
    for k, v in r.items():
        s = str(v)
        print("    %-16s %s" % (k, s[:110]))
PY

echo
echo "===== RefSpatial data parquet 结构 ====="
python3 - <<'PY'
import pyarrow.parquet as pq, os
B = "/root/private_data/workspace/robo/third_party/benchmarks/RefSpatial-Bench/data"
for f in sorted(os.listdir(B)):
    t = pq.read_table(os.path.join(B, f))
    print(f, "rows =", t.num_rows, "cols =", t.schema.names)
    r = t.slice(0, 1).to_pylist()[0]
    for k, v in r.items():
        print("    %-12s %s" % (k, str(v)[:110]))
    break
PY

echo
echo "===== 三个目录的文件/体积总账 ====="
for d in CV-Bench RefSpatial-Bench TraceSpatial-Bench; do
  printf '%-22s files=%-6s %s\n' "$d" \
    "$(find "$W/third_party/benchmarks/$d" -type f -not -path '*/.cache/*' | wc -l)" \
    "$(du -sh "$W/third_party/benchmarks/$d" | cut -f1)"
done
echo "总计: $(du -sh "$W/third_party/benchmarks" | cut -f1)"
