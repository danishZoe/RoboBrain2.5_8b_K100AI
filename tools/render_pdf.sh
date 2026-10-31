#!/usr/bin/env bash
# 把赛事 PDF 的关键页渲染成 PNG，便于肉眼读取「官方材料支持清单」
set -o pipefail
source /root/private_data/workspace/robo/tools/env_net.sh

W=/root/private_data/workspace/robo
PDF="$W/cache/guoxin_guomo.pdf"
OUT="$W/outputs/pdf_pages"
PREFIX="$W/envs/pylibs"
mkdir -p "$OUT"

if [ ! -d "$PREFIX/pypdfium2" ]; then
  echo "=== 安装 pypdfium2 到隔离前缀 ==="
  pip install -q --no-cache-dir --target "$PREFIX" pypdfium2 2>&1 | tail -3
fi

PYTHONPATH="$PREFIX" python3 - "$PDF" "$OUT" <<'PY'
import sys, os
import pypdfium2 as pdfium

pdf_path, out_dir = sys.argv[1], sys.argv[2]
pdf = pdfium.PdfDocument(pdf_path)
# 渲染 1-based 页码 5,6,7（官方材料支持清单及其后）
for pno in (5, 6, 7):
    if pno > len(pdf):
        continue
    page = pdf[pno - 1]
    bmp = page.render(scale=2.2)
    img = bmp.to_pil()
    fp = os.path.join(out_dir, f"page{pno:02d}.png")
    img.save(fp)
    print("wrote", fp, img.size)
PY
