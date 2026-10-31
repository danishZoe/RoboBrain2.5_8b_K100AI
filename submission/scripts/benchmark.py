#!/usr/bin/env python3
"""Phase 3 baseline: fixed-sample functional + performance measurement.

Measures, per sample and per repeat:
  * prompt tokens / generated tokens
  * time to first token (TTFT, real streaming measurement)
  * end-to-end latency, decode throughput (tok/s), end-to-end throughput
  * peak VRAM (allocated / reserved)

The first repeat of every sample is treated as warm-up and excluded from the
aggregate statistics; load time is measured separately. Async device execution
is handled with explicit ``torch.cuda.synchronize()`` calls.

Usage:
  python3 scripts/benchmark.py --tasks samples/tasks.json --repeat 3 \
      --out-dir outputs/phase3_baseline
"""

import argparse
import json
import os
import statistics
import sys
import threading
import time

import torch
from transformers import TextIteratorStreamer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robo_infer import build_prompt, load_vision_inputs  # noqa: E402


def run_once(rb, prompt, image, task, max_new_tokens, do_sample, temperature):
    """One request with real streaming, so TTFT is observable."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{image}"},
            {"type": "text", "text": build_prompt(prompt, task)},
        ],
    }]
    chat_text = rb.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, _ = load_vision_inputs(messages)
    inputs = rb.processor(text=[chat_text], images=image_inputs, videos=video_inputs,
                          padding=True, return_tensors="pt").to(rb.device)

    streamer = TextIteratorStreamer(
        rb.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        streamer=streamer,
    )

    torch.cuda.synchronize()
    t0 = time.time()
    thread = threading.Thread(target=rb.model.generate, kwargs={**inputs, **gen_kwargs})
    thread.start()

    ttft = None
    chunks = []
    for chunk in streamer:
        if ttft is None:
            torch.cuda.synchronize()
            ttft = time.time() - t0
        chunks.append(chunk)
    thread.join()
    torch.cuda.synchronize()
    total = time.time() - t0

    answer = "".join(chunks)
    gen_tokens = len(rb.processor.tokenizer(answer, add_special_tokens=False)["input_ids"]) \
        if answer.strip() else 0

    return {
        "answer": answer,
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "gen_tokens": gen_tokens,
        "ttft_seconds": round(ttft, 4) if ttft is not None else None,
        "total_seconds": round(total, 4),
        "decode_tok_per_s": round(gen_tokens / (total - ttft), 3)
        if (ttft is not None and total > ttft) else None,
        "e2e_tok_per_s": round(gen_tokens / total, 3) if total > 0 else None,
    }


def agg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "median": round(statistics.median(vals), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=os.environ.get("MODEL_PATH"))
    ap.add_argument("--tasks", default=os.path.join(
        os.environ.get("WORK_DIR", "."), "samples", "tasks.json"))
    ap.add_argument("--image-dir", default=os.path.join(
        os.environ.get("WORK_DIR", "."), "samples", "images"))
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out-dir", default="outputs/phase3_baseline")
    ap.add_argument("--device-map", default="cuda:0")
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.tasks, encoding="utf-8") as f:
        tasks = json.load(f)

    print("=" * 72)
    print("PHASE 3 BASELINE - RoboBrain2.5-8B-NV / K100AI")
    print("=" * 72)
    print(f"tasks: {args.tasks} ({len(tasks)} samples)  repeat: {args.repeat}")

    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype="auto", device_map=args.device_map)
    processor = AutoProcessor.from_pretrained(args.model_path)
    model.eval()
    device = next(model.parameters()).device
    load_s = time.time() - t0
    print(f"model loaded in {load_s:.1f}s on {device}")

    class _RB:
        pass
    rb = _RB()
    rb.model, rb.processor, rb.device = model, processor, device

    print(f"vram after load: {torch.cuda.memory_reserved()/1024**3:.2f} GiB")

    records = []
    for t in tasks:
        image = os.path.join(args.image_dir, t["image"])
        mnt = int(t.get("max_new_tokens", 128))
        do_sample = bool(t.get("do_sample", False))
        print(f"\n=== {t['id']} | task={t['task']} | max_new_tokens={mnt} "
              f"| do_sample={do_sample} ===", flush=True)
        for r in range(args.repeat):
            torch.cuda.reset_peak_memory_stats()
            rec = run_once(rb, t["prompt"], image, t["task"], mnt, do_sample, 0.7)
            rec.update({
                "sample_id": t["id"], "task": t["task"],
                "repeat": r + 1, "warmup": (r == 0),
                "peak_vram_alloc_gib": round(torch.cuda.max_memory_allocated()/1024**3, 3),
                "peak_vram_reserved_gib": round(torch.cuda.max_memory_reserved()/1024**3, 3),
            })
            records.append(rec)
            print(f"  rep{r + 1}: ttft={rec['ttft_seconds']}s total={rec['total_seconds']}s "
                  f"gen={rec['gen_tokens']} tok  decode={rec['decode_tok_per_s']} tok/s "
                  f"peakVRAM={rec['peak_vram_reserved_gib']}GiB", flush=True)
            print(f"  answer: {rec['answer'][:200]!r}", flush=True)

    steady = [r for r in records if not r["warmup"]]
    summary = {
        "model_load_seconds": round(load_s, 2),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0),
        "samples": len(tasks),
        "repeats": args.repeat,
        "steady_state": {
            "ttft_seconds": agg([r["ttft_seconds"] for r in steady]),
            "total_seconds": agg([r["total_seconds"] for r in steady]),
            "decode_tok_per_s": agg([r["decode_tok_per_s"] for r in steady]),
            "e2e_tok_per_s": agg([r["e2e_tok_per_s"] for r in steady]),
            "peak_vram_reserved_gib": agg([r["peak_vram_reserved_gib"] for r in steady]),
        },
    }

    with open(os.path.join(args.out_dir, "baseline_raw.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=2)

    lines = ["# Phase 3 baseline (transformers backend, single HCU)", "",
             f"- device: `{summary['device_name']}` (`{summary['device']}`)",
             f"- model load: {summary['model_load_seconds']} s",
             f"- samples: {summary['samples']}, repeats: {summary['repeats']} "
             f"(first repeat = warm-up, excluded from aggregates)", "",
             "## steady-state aggregate", ""]
    for k, v in summary["steady_state"].items():
        if v:
            lines.append(f"- **{k}**: mean {v['mean']} (min {v['min']} / max {v['max']}, n={v['n']})")
    lines += ["", "## per-sample (steady state)", "",
              "| sample | task | prompt tok | gen tok | TTFT s | total s | decode tok/s | peak VRAM GiB |",
              "|---|---|---|---|---|---|---|---|"]
    for r in steady:
        lines.append(f"| {r['sample_id']} | {r['task']} | {r['prompt_tokens']} | {r['gen_tokens']} | "
                     f"{r['ttft_seconds']} | {r['total_seconds']} | {r['decode_tok_per_s']} | "
                     f"{r['peak_vram_reserved_gib']} |")
    md = "\n".join(lines) + "\n"
    with open(os.path.join(args.out_dir, "baseline_summary.md"), "w", encoding="utf-8") as f:
        f.write(md)

    print("\n" + "=" * 72)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 72)
    print("artifacts:", os.path.join(args.out_dir, "baseline_raw.json"),
          os.path.join(args.out_dir, "baseline_summary.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
