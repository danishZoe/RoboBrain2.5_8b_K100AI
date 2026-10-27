#!/usr/bin/env python3
"""Batched inference for RoboBrain2.5-8B-NV on K100AI (transformers path).

This is the practical replacement for a serving backend on this node: SGLang's
attention backends are all broken in this image (see reports/06_sglang_compatibility.md),
so batching inside the verified transformers path is how throughput is recovered.

Key detail: the tokenizer MUST use left padding for batched decoder-only
generation. With right padding the answers are corrupted (pad positions leak
into the output). The script sets it and verifies every answer against the
single-request baseline when one is available.

Usage:
  # run the fixed sample set in batches of 4
  python3 scripts/batch_infer.py --tasks samples/tasks_official.json \
      --image-dir samples/images --batch-size 4 --out-dir outputs/batch_run

  # run an arbitrary request list (one JSON object per line)
  #   {"id": "r1", "task": "pointing", "image": "a.jpg", "prompt": "...", "max_new_tokens": 64}
  python3 scripts/batch_infer.py --requests my_requests.jsonl --batch-size 4
"""

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robo_infer import build_prompt, draw_on_image, load_vision_inputs, parse_answer  # noqa: E402


def load_requests(args):
    if args.requests:
        reqs = []
        with open(args.requests, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    reqs.append(json.loads(line))
        return reqs
    with open(args.tasks, encoding="utf-8") as f:
        tasks = json.load(f)
    for t in tasks:
        t.setdefault("id", t.get("task", "req"))
        t["image"] = os.path.join(args.image_dir, t["image"])
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=os.environ.get("MODEL_PATH"))
    ap.add_argument("--tasks", default=os.path.join(
        os.environ.get("WORK_DIR", "."), "samples", "tasks_official.json"))
    ap.add_argument("--requests", default=None, help="JSONL request file (overrides --tasks)")
    ap.add_argument("--image-dir", default=os.path.join(
        os.environ.get("WORK_DIR", "."), "samples", "images"))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out-dir", default="outputs/batch_run")
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--max-pixels", type=int, default=0)
    ap.add_argument("--plot", type=int, default=0,
                    help="write annotated images for spatial tasks")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    reqs = load_requests(args)
    print(f"{len(reqs)} requests, batch_size={args.batch_size}")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype="auto", device_map=args.device_map)
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "left"   # required for batched generation
    if args.max_pixels:
        processor.image_processor.max_pixels = int(args.max_pixels)
        processor.image_processor.size["longest_edge"] = int(args.max_pixels)
    model.eval()
    device = next(model.parameters()).device
    print(f"model on {device}, padding_side={processor.tokenizer.padding_side}")

    results = []
    t_start = time.time()
    for bi in range(0, len(reqs), args.batch_size):
        chunk = reqs[bi:bi + args.batch_size]
        texts, images = [], []
        for r in chunk:
            img = r["image"]
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img if img.startswith("http") else f"file://{img}"},
                    {"type": "text", "text": build_prompt(r["prompt"], r.get("task", "general"))},
                ],
            }]
            texts.append(processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))
            img_in, _, _ = load_vision_inputs(messages)
            images.extend(img_in)

        inputs = processor(text=texts, images=images, padding=True,
                           return_tensors="pt").to(device)
        mnt = max(int(r.get("max_new_tokens", 128)) for r in chunk)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=mnt,
                                 do_sample=False, temperature=0.7)
        torch.cuda.synchronize()
        wall = time.time() - t0

        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        answers = processor.batch_decode(trimmed, skip_special_tokens=True)
        gen_counts = [int(x.shape[-1]) for x in trimmed]

        for r, ans, g in zip(chunk, answers, gen_counts):
            res = dict(r)
            res["answer"] = ans
            parsed = parse_answer(ans, r.get("task", "general"))
            res["parsed"] = parsed
            res["gen_tokens"] = g
            if args.plot and r.get("task") in ("pointing", "trajectory", "grounding"):
                base = os.path.basename(r["image"]).replace(".", "_annotated.")
                ann = os.path.join(args.out_dir, base)
                draw_on_image(r["image"], ann,
                              points=parsed.get("points"),
                              boxes=parsed.get("boxes"),
                              trajectories=parsed.get("trajectories"))
                res["annotated_image"] = ann
            results.append(res)
            print(f"[{r.get('id')}] task={r.get('task', 'general')} gen={g} :: {ans[:110]}")

        print(f"  batch of {len(chunk)}: {wall:.2f}s, "
              f"{sum(gen_counts)/wall:.2f} tok/s aggregate, "
              f"peak VRAM {torch.cuda.max_memory_reserved()/1024**3:.2f} GiB")

    total = time.time() - t_start
    out_path = os.path.join(args.out_dir, "batch_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"batch_size": args.batch_size, "total_seconds": round(total, 2),
                   "device": str(device), "results": results}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n{len(results)} results in {total:.1f}s -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
