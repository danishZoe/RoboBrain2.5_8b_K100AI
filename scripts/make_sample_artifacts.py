#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_sample_artifacts.py - 生成验收标准 §9 要求的「固定样例及输出」材料。

§9 要求固定样例材料包含：**样例来源、校验值、原文、解析值、坐标/轨迹叠加图**。

本脚本从服务级探针结果（serving 路径的真实输出）生成：
  submission/samples/parsed_outputs.json  —— 每个样例的原文 + 解析值 + 结束原因
  submission/samples/annotated/<id>.png   —— 坐标/框/轨迹的叠加图

叠加图直接复用 `scripts/robo_infer.py` 的 `draw_on_image()`，
其坐标系与上游 `draw_on_image()` 完全一致（0~1000 归一化 → 绝对像素），
因此图上位置与官方示例可比。解析正则同样复用 `robo_infer.parse_answer()`，
避免「图是另一套正则画的」这种口径分裂。

用法:
    python3 scripts/make_sample_artifacts.py \
        --probe outputs/phase5_probe_canonical/probe_result.json \
        --out-dir submission/samples
"""
import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def ensure_dtk_runtime():
    """robo_infer 会 import torch，而 torch 需要 DTK 运行时（libgalaxyhip.so.5）。
    若当前 shell 未加载 /opt/dtk/env.sh，就用 bash 重新 exec 一次。"""
    try:
        import torch  # noqa: F401
        return
    except ImportError:
        pass
    dtk = "/opt/dtk/env.sh"
    if os.path.exists(dtk):
        cmd = ('set +u; source %s; set -u; exec %s "%s" %s'
               % (dtk, sys.executable, os.path.abspath(__file__),
                  " ".join('"%s"' % a for a in sys.argv[1:])))
        os.execvp("bash", ["bash", "-c", cmd])
    raise SystemExit(
        "[artifacts] torch 无法导入且未找到 /opt/dtk/env.sh；"
        "请在加载 DTK 运行时的环境里执行本脚本。"
    )


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="outputs/phase5_probe_canonical/probe_result.json")
    ap.add_argument("--out-dir", default="submission/samples")
    ap.add_argument("--repo-root", default=ROOT)
    args = ap.parse_args()

    if not os.path.exists(args.probe):
        print("[artifacts] probe result missing: %s" % args.probe)
        return 2
    ensure_dtk_runtime()
    data = json.load(open(args.probe, encoding="utf-8"))
    records = data.get("records") or []
    if not records:
        print("[artifacts] no records in %s" % args.probe)
        return 2

    from robo_infer import draw_on_image, parse_answer

    out_dir = args.out_dir
    ann_dir = os.path.join(out_dir, "annotated")
    os.makedirs(ann_dir, exist_ok=True)

    items = []
    for rec in records:
        sid = rec.get("sample_id")
        task = rec.get("task")
        text = rec.get("text") or ""
        # 解析一律用上游同款正则；不另立口径
        parsed = parse_answer(text, task)
        img_rel = (rec.get("images") or [None])[0]
        entry = {
            "sample_id": sid,
            "task": task,
            "prompt": rec.get("prompt"),
            "image": img_rel,
            "raw_output": text,
            "parsed": parsed,
            "finish_reason": rec.get("finish_reason"),
            "gen_tokens": rec.get("gen_tokens"),
            "seconds": rec.get("seconds"),
        }
        # 叠加图
        if img_rel:
            img_abs = os.path.join(args.repo_root, img_rel)
            if os.path.exists(img_abs):
                ann_rel = os.path.join("annotated", "%s.png" % sid)
                ann_abs = os.path.join(out_dir, ann_rel)
                try:
                    draw_on_image(
                        img_abs, ann_abs,
                        points=[tuple(p) for p in parsed.get("points", [])],
                        boxes=[list(b) for b in parsed.get("boxes", [])],
                        trajectories=[[tuple(p) for p in t] for t in parsed.get("trajectories", [])],
                    )
                    entry["annotated_overlay"] = ann_rel
                    entry["annotated_overlay_sha256"] = sha256(ann_abs)
                except Exception as e:  # noqa: BLE001
                    entry["annotated_overlay_error"] = "%s: %s" % (type(e).__name__, e)
        items.append(entry)

    payload = {
        "source_probe": args.probe,
        "backend": data.get("backend"),
        "url": data.get("url"),
        "coordinate_convention": "0~1000 归一化（与上游 draw_on_image 一致）",
        "parser": "scripts/robo_infer.py::parse_answer（与官方正则逐字一致）",
        "samples": items,
    }
    out_json = os.path.join(out_dir, "parsed_outputs.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    n_ann = sum(1 for it in items if it.get("annotated_overlay"))
    print("[artifacts] wrote %s (%d samples)" % (out_json, len(items)))
    print("[artifacts] overlays: %d/%d -> %s" % (n_ann, len(items), ann_dir))
    for it in items:
        keys = ",".join(sorted(it["parsed"].keys())) or "-"
        print("   %-22s task=%-10s parsed=%-14s overlay=%s"
              % (it["sample_id"], it["task"], keys,
                 os.path.basename(it.get("annotated_overlay", "-"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
