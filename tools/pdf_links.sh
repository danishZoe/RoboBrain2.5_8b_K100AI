#!/usr/bin/env bash
# 从赛事 PDF 里抽出所有链接注解（/URI），定位「官方材料支持清单」的下载入口
set -o pipefail
source /root/private_data/workspace/robo/tools/env_net.sh

W=/root/private_data/workspace/robo
PDF="$W/cache/guoxin_guomo.pdf"
mkdir -p "$W/cache" "$W/tmp"

if [ ! -f "$PDF" ]; then
  echo "PDF 不在预期路径，请先 scp 进来"
  exit 1
fi

# pypdf 装到工作目录内的隔离前缀，不污染公共 site-packages
PREFIX="$W/envs/pylibs"
if [ ! -d "$PREFIX/pypdf" ]; then
  echo "=== 安装 pypdf 到隔离前缀 ==="
  pip install -q --no-cache-dir --target "$PREFIX" pypdf 2>&1 | tail -3
fi

PYTHONPATH="$PREFIX" python3 - "$PDF" <<'PY'
import sys
from pypdf import PdfReader

path = sys.argv[1]
r = PdfReader(path)
print("pages:", len(r.pages))
seen = set()
for i, page in enumerate(r.pages, 1):
    annots = page.get("/Annots") or []
    for a in annots:
        try:
            obj = a.get_object()
            act = obj.get("/A")
            uri = None
            if act and act.get("/URI"):
                uri = act.get("/URI")
            elif obj.get("/URI"):
                uri = obj.get("/URI")
            if not uri:
                continue
            uri = str(uri)
            if uri in seen:
                continue
            seen.add(uri)
            # 提取链接所在页面的文字片段，便于判断这个链接属于哪一列
            txt = (page.extract_text() or "").replace("\n", " ")
            print(f"[p{i}] {uri}")
            print(f"      page-text-head: {txt[:110]}")
        except Exception as e:
            pass
print("total unique URIs:", len(seen))
PY
