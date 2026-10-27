#!/usr/bin/env bash
# 用 HTTP Range 只读 ZIP 中央目录，检查 aiter wheel 是否含 aiter/attention 子包
# （不下载整个 wheel，不安装到任何环境）
source /root/private_data/workspace/robo/tools/env_net.sh

W=/root/private_data/workspace/robo
TMP="$W/tmp/pypi_probe"
mkdir -p "$TMP"

PYTHONPATH= python3 - "$TMP" <<'PY'
import json, os, re, sys, urllib.request, urllib.parse, zipfile, io

TMP = sys.argv[1]
BASE = "https://pypi.sourcefind.cn/release/dtk2604"

def get(url, headers=None, timeout=60):
    req = urllib.request.Request(url, headers=headers or {})
    return urllib.request.urlopen(req, timeout=timeout)

def simple_index(pkg):
    """devpi 的 simple 页面"""
    out = {}
    for path in (f"{BASE}/{pkg}/", f"{BASE}/+simple/{pkg}/", f"{BASE}/+simple/{pkg}"):
        try:
            with get(path) as r:
                html = r.read().decode("utf-8", "replace")
            for href, fname in re.findall(r'href="([^"]+)"[^>]*>([^<]+)</a>', html):
                out[urllib.parse.urljoin(path, href)] = fname
            if out:
                return out
        except Exception as e:
            last = e
    print(f"  [simple index {pkg}] 失败: {last}")
    return out

def list_zip_names(url):
    """只取 ZIP 尾部，解析中央目录里的文件清单"""
    try:
        with get(url, headers={"Range": "bytes=0-0"}) as r:
            total = r.headers.get("Content-Range")
            size = int(total.split("/")[-1]) if total else int(r.headers.get("Content-Length", 0))
    except Exception as e:
        return None, None, f"HEAD/range 失败: {e!r}"

    tail_len = min(size, 4 * 1024 * 1024)
    start = max(0, size - tail_len)
    try:
        with get(url, headers={"Range": f"bytes={start}-{size-1}"}) as r:
            blob = r.read()
    except Exception as e:
        return None, size, f"range 读取失败: {e!r}"

    # 找 EOCD
    idx = blob.rfind(b"PK\x05\x06")
    if idx < 0:
        return None, size, "未找到 EOCD"
    import struct
    (cd_count, cd_size, cd_off) = struct.unpack("<HII", blob[idx+10:idx+20])
    # 中央目录可能不在我们取的这一段里
    if cd_off >= start:
        # 中央目录起点在已取区间内
        rel = cd_off - start
        cd = blob[rel:rel+cd_size]
    else:
        try:
            with get(url, headers={"Range": f"bytes={cd_off}-{cd_off+cd_size-1}"}) as r:
                cd = r.read()
        except Exception as e:
            return None, size, f"中央目录读取失败: {e!r}"
    names = re.findall(b"PK\x01\x02.{42}(.{0,120}?)(?:\x00|\r)", cd, re.S)
    out = []
    for n in names:
        try:
            out.append(n.decode("utf-8"))
        except Exception:
            pass
    return out, size, None

for pkg in ("aiter", "flashinfer-python"):
    print(f"\n{'='*70}\n### {pkg}\n{'='*70}")
    idx = simple_index(pkg)
    if not idx:
        continue
    # 只看和当前栈匹配的候选：das.opt1 + torch290；aiter 另列 das185 参考
    for url, fname in sorted(idx.items(), key=lambda kv: kv[1]):
        if not (fname.endswith(".whl") or fname.endswith(".tar.gz")):
            continue
        if pkg == "aiter" and "torch290" not in fname and "das185" not in fname:
            continue
        if pkg == "flashinfer-python" and "0.6.1" not in fname and "0.7.2" not in fname:
            continue
        print(f"\n--- {fname}")
        names, size, err = list_zip_names(url)
        if err:
            print(f"    {err}")
            continue
        print(f"    wheel 大小: {size/1024/1024:.1f} MiB   条目数: {len(names)}")
        if pkg == "aiter":
            att = [n for n in names if n.startswith("aiter/attention/")]
            print(f"    ★ aiter/attention/ 条目数: {len(att)}")
            for n in att[:12]:
                print("       ", n)
            for key in ("paged_attention_ragged", "flash_attn_varlen_func", "unified_attention"):
                hit = [n for n in names if key in n]
                print(f"    {key}: {len(hit)} 个 -> {hit[:3]}")
            top = sorted({n.split('/')[1] for n in names if n.startswith('aiter/') and n.count('/') >= 2})
            print("    aiter 顶层子模块:", top)
        else:
            print(f"    条目数 {len(names)}，样例: {names[:5]}")
PY
