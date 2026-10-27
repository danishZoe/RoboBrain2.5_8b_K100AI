#!/usr/bin/env python3
"""Phase 4 controlled experiment harness (transformers backend).

Every experiment changes exactly ONE main variable against the frozen baseline and
re-runs the same fixed samples with the same measurement method, so results are
directly comparable with outputs/phase3_baseline/baseline_raw.json.

Variables (use only what you intend to change):
  --attn-impl   default | sdpa | flash_attention_2 | eager
  --max-pixels  cap on the visual-token budget (processor level)
  --min-pixels  floor on the visual-token budget (processor level)

Correctness is re-checked every run: the parsed structure and the raw answer are
written out so a config change can be compared against the baseline answer, not
just against a non-empty output.

Usage:
  python3 scripts/tune.py --tag attn_fa2 --attn-impl flash_attention_2 \
      --tasks samples/tasks_official.json --repeat 3 --out-dir outputs/phase4
"""

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import agg, run_once  # noqa: E402


def apply_pixel_budget(processor, max_pixels, min_pixels):
    """Cap / floor the visual-token budget at the processor level.

    Qwen2VLImageProcessor(Fast) derives min/max pixels from the ``size`` dict and
    also exposes explicit attributes; set both so the fast path and the slow path
    agree.
    """
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        raise RuntimeError("processor has no image_processor")
    changed = {}
    if max_pixels:
        ip.max_pixels = int(max_pixels)
        changed["max_pixels"] = ip.max_pixels
    if min_pixels:
        ip.min_pixels = int(min_pixels)
        changed["min_pixels"] = ip.min_pixels
    size = getattr(ip, "size", None)
    if isinstance(size, dict):
        if max_pixels:
            size["longest_edge"] = int(max_pixels)
        if min_pixels:
            size["shortest_edge"] = int(min_pixels)
        changed["size"] = dict(size)
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=os.environ.get("MODEL_PATH"))
    ap.add_argument("--tasks", default=os.path.join(
        os.environ.get("WORK_DIR", "."), "samples", "tasks_official.json"))
    ap.add_argument("--image-dir", default=os.path.join(
        os.environ.get("WORK_DIR", "."), "samples", "images"))
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--attn-impl", default="default",
                    choices=["default", "sdpa", "flash_attention_2", "eager"])
    ap.add_argument("--max-pixels", type=int, default=0)
    ap.add_argument("--min-pixels", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="outputs/phase4")
    ap.add_argument("--device-map", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.tasks, encoding="utf-8") as f:
        tasks = json.load(f)

    print("=" * 72)
    print(f"EXPERIMENT: {args.tag}")
    print(f"  attn_impl = {args.attn_impl}")
    print(f"  max_pixels= {args.max_pixels or 'default'}  min_pixels= {args.min_pixels or 'default'}")
    print("=" * 72)

    kw = {}
    if args.attn_impl != "default":
        kw["attn_implementation"] = args.attn_impl

    t0 = time.time()
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path, dtype="auto", device_map=args.device_map, **kw)
    except Exception as e:
        print(f"[FATAL] model load with attn_implementation={args.attn_impl} failed: "
              f"{type(e).__name__}: {e}")
        with open(os.path.join(args.out_dir, f"{args.tag}_FAILED.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"tag": args.tag, "attn_impl": args.attn_impl,
                       "error": f"{type(e).__name__}: {e}"}, f, ensure_ascii=False, indent=2)
        return 2
    processor = AutoProcessor.from_pretrained(args.model_path)
    model.eval()
    device = next(model.parameters()).device
    load_s = time.time() - t0

    pixel_cfg = apply_pixel_budget(processor, args.max_pixels, args.min_pixels)

    class _RB:
        pass
    rb = _RB()
    rb.model, rb.processor, rb.device = model, processor, device

    print(f"loaded in {load_s:.1f}s on {device}; pixel budget = {pixel_cfg}")

    records = []
    for t in tasks:
        image = os.path.join(args.image_dir, t["image"])
        mnt = int(t.get("max_new_tokens", 128))
        do_sample = bool(t.get("do_sample", False))
        for r in range(args.repeat):
            torch.cuda.reset_peak_memory_stats()
            rec = run_once(rb, t["prompt"], image, t["task"], mnt, do_sample, 0.7)
            rec.update({"sample_id": t["id"], "task": t["task"], "repeat": r + 1,
                        "warmup": (r == 0),
                        "peak_vram_reserved_gib": round(torch.cuda.max_memory_reserved()/1024**3, 3)})
            records.append(rec)
            print(f"  {t['id']:22s} rep{r+1} ptok={rec['prompt_tokens']:4d} "
                  f"gtok={rec['gen_tokens']:3d} ttft={rec['ttft_seconds']:.4f} "
                  f"tot={rec['total_seconds']:.4f} dec={rec['decode_tok_per_s']} "
                  f"vram={rec['peak_vram_reserved_gib']}", flush=True)

    steady = [r for r in records if not r["warmup"]]
    summary = {
        "tag": args.tag,
        "attn_impl": args.attn_impl,
        "pixel_budget": pixel_cfg,
        "model_load_seconds": round(load_s, 2),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0),
        "steady_state": {
            "ttft_seconds": agg([r["ttft_seconds"] for r in steady]),
            "total_seconds": agg([r["total_seconds"] for r in steady]),
            "decode_tok_per_s": agg([r["decode_tok_per_s"] for r in steady]),
            "e2e_tok_per_s": agg([r["e2e_tok_per_s"] for r in steady]),
            "peak_vram_reserved_gib": agg([r["peak_vram_reserved_gib"] for r in steady]),
            "prompt_tokens": agg([r["prompt_tokens"] for r in steady]),
        },
        "answers": {r["sample_id"]: r["answer"] for r in records if not r["warmup"]},
    }

    out = os.path.join(args.out_dir, f"{args.tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=2)

    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    print("artifact:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
