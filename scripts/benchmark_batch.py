#!/usr/bin/env python3
"""Phase 4: batched inference on the VERIFIED transformers path.

Motivation (from the phase-3 baseline): single-stream decode is pinned at
~11.8 tok/s regardless of sample, which points at per-step weight streaming
(16.3 GiB of bf16 weights read once per decode step) rather than at compute.
If that is the bottleneck, running N requests as ONE padded batch should
amortise the weight read and raise AGGREGATE throughput, while per-request
latency stays roughly constant.

This script measures exactly that, on the same fixed samples and with the same
aggregation rules as the baseline:

  * batch_size sweep (1 / 2 / 4 / 8)
  * wall time for the whole batch
  * aggregate tok/s (sum of generated tokens / batch wall time)
  * per-request latency and per-request tok/s
  * peak VRAM

Correctness is re-checked: every sample's answer is compared with the baseline
answer, because batching changes padding and can change numerics.

Usage:
  python3 scripts/benchmark_batch.py --batch-sizes 1 2 4 8 --repeat 3 \
      --tasks samples/tasks_official.json --out-dir outputs/phase4_batch
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robo_infer import build_prompt, load_vision_inputs  # noqa: E402


def build_inputs(rb, tasks, image_dir, max_pixels=0):
    """Tokenise a list of samples into one padded batch."""
    texts, images = [], []
    for t in tasks:
        image = os.path.join(image_dir, t["image"])
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image}"},
                {"type": "text", "text": build_prompt(t["prompt"], t["task"])},
            ],
        }]
        chat = rb.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        img_inputs, _, _ = load_vision_inputs(messages)
        texts.append(chat)
        images.extend(img_inputs)

    inputs = rb.processor(text=texts, images=images, padding=True,
                          return_tensors="pt").to(rb.device)
    return inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=os.environ.get("MODEL_PATH"))
    ap.add_argument("--tasks", default=os.path.join(
        os.environ.get("WORK_DIR", "."), "samples", "tasks_official.json"))
    ap.add_argument("--image-dir", default=os.path.join(
        os.environ.get("WORK_DIR", "."), "samples", "images"))
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--max-pixels", type=int, default=0)
    ap.add_argument("--out-dir", default="outputs/phase4_batch")
    ap.add_argument("--device-map", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.tasks, encoding="utf-8") as f:
        tasks = json.load(f)

    print("=" * 72)
    print("PHASE 4 BATCH SWEEP - transformers path, verified backend")
    print("=" * 72)

    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype="auto", device_map=args.device_map)
    processor = AutoProcessor.from_pretrained(args.model_path)
    # Batched decoder-only generation MUST left-pad: with right padding the model
    # continues from the pad positions and the answers get corrupted (verified:
    # right padding produced "Human:" / pad-token spam in the bs=4 outputs).
    # The processor owns the tokenizer, so set it there.
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        print(f"tokenizer padding_side = {processor.tokenizer.padding_side}")
    model.eval()
    device = next(model.parameters()).device
    load_s = time.time() - t0
    print(f"model loaded in {load_s:.1f}s on {device}")

    if args.max_pixels:
        ip = processor.image_processor
        ip.max_pixels = int(args.max_pixels)
        ip.size["longest_edge"] = int(args.max_pixels)
        print(f"visual budget capped at {args.max_pixels} px")

    class _RB:
        pass
    rb = _RB()
    rb.model, rb.processor, rb.device = model, processor, device

    records = []
    for bs in args.batch_sizes:
        print(f"\n=== batch_size {bs} ===", flush=True)
        for rep in range(args.repeat):
            batch = tasks[:bs] if bs <= len(tasks) else (
                tasks + tasks * (bs // len(tasks) + 1))[:bs]
            inputs = build_inputs(rb, batch, args.image_dir)
            prompt_len = int(inputs["input_ids"].shape[1])

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=128,
                                     do_sample=False, temperature=0.7)
            torch.cuda.synchronize()
            wall = time.time() - t0

            trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
            answers = processor.batch_decode(trimmed, skip_special_tokens=True)
            gen_counts = [int(x.shape[-1]) for x in trimmed]
            total_gen = sum(gen_counts)

            rec = {
                "batch_size": bs,
                "repeat": rep + 1,
                "warmup": rep == 0,
                "prompt_len_padded": prompt_len,
                "wall_seconds": round(wall, 4),
                "gen_tokens_total": total_gen,
                "gen_tokens_per_request": gen_counts,
                "aggregate_tok_per_s": round(total_gen / wall, 3) if wall else None,
                "per_request_seconds": round(wall / bs, 4),
                "per_request_tok_per_s": round(total_gen / wall / bs, 3) if wall else None,
                "peak_vram_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
                "answers": answers[:len(tasks)],
            }
            records.append(rec)
            print(f"  bs={bs} rep{rep+1}: wall={rec['wall_seconds']:.3f}s "
                  f"gen={total_gen} agg={rec['aggregate_tok_per_s']} tok/s "
                  f"per-req={rec['per_request_seconds']:.3f}s "
                  f"vram={rec['peak_vram_reserved_gib']}GiB", flush=True)

    steady = [r for r in records if not r["warmup"]]
    summary = {"model_load_seconds": round(load_s, 2), "device": str(device),
               "device_name": torch.cuda.get_device_name(0),
               "max_pixels": args.max_pixels or "default",
               "per_batch_size": {}}
    for bs in args.batch_sizes:
        sub = [r for r in steady if r["batch_size"] == bs]
        summary["per_batch_size"][str(bs)] = {
            "wall_seconds_mean": round(statistics.mean(r["wall_seconds"] for r in sub), 4),
            "aggregate_tok_per_s_mean": round(
                statistics.mean(r["aggregate_tok_per_s"] for r in sub), 3),
            "per_request_seconds_mean": round(
                statistics.mean(r["per_request_seconds"] for r in sub), 4),
            "per_request_tok_per_s_mean": round(
                statistics.mean(r["per_request_tok_per_s"] for r in sub), 3),
            "peak_vram_reserved_gib_max": max(r["peak_vram_reserved_gib"] for r in sub),
            "n": len(sub),
        }

    base = summary["per_batch_size"].get("1", {}).get("aggregate_tok_per_s_mean")
    summary["scaling_vs_bs1"] = {
        bs: round(v["aggregate_tok_per_s_mean"] / base, 3)
        for bs, v in summary["per_batch_size"].items()
    } if base else {}

    # ---- correctness: batched answers must match the single-request baseline ----
    baseline = {}
    bp = os.path.join(os.environ.get("WORK_DIR", "."),
                      "outputs", "phase3_baseline", "baseline_raw.json")
    if os.path.isfile(bp):
        with open(bp, encoding="utf-8") as f:
            braw = json.load(f)
        for rec in braw.get("records", []):
            if not rec.get("warmup"):
                baseline.setdefault(rec["sample_id"], rec["answer"])

    correctness = []
    for bs in args.batch_sizes:
        sub = [r for r in steady if r["batch_size"] == bs]
        for r in sub:
            for sid, ans in zip([t["id"] for t in tasks], r["answers"]):
                ref = baseline.get(sid)
                correctness.append({
                    "batch_size": bs, "sample_id": sid,
                    "matches_baseline": (ref is not None and ans.strip() == ref.strip()),
                })
    bad = [c for c in correctness if not c["matches_baseline"]]
    summary["correctness"] = {
        "compared": len(correctness),
        "mismatched": len(bad),
        "mismatch_detail": bad[:20],
    }

    out = os.path.join(args.out_dir, "batch_sweep.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=2)

    lines = ["# 第四阶段：批处理扩展（transformers 路径，已验证后端）", "",
             f"- 设备：`{summary['device_name']}`，模型加载 {summary['model_load_seconds']} s",
             f"- 视觉预算：{summary['max_pixels']}", "",
             "| batch | 批耗时 s | 聚合 tok/s | 相对 bs=1 | 单请求时延 s | 单请求 tok/s | 峰值显存 GiB |",
             "|---|---|---|---|---|---|---|"]
    for bs in args.batch_sizes:
        v = summary["per_batch_size"][str(bs)]
        lines.append(f"| {bs} | {v['wall_seconds_mean']} | {v['aggregate_tok_per_s_mean']} | "
                     f"×{summary['scaling_vs_bs1'].get(bs, '-')} | "
                     f"{v['per_request_seconds_mean']} | {v['per_request_tok_per_s_mean']} | "
                     f"{v['peak_vram_reserved_gib_max']} |")
    with open(os.path.join(args.out_dir, "batch_sweep.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))
    print("artifact:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
