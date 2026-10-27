#!/usr/bin/env python3
"""Functional validation of the four RoboBrain2.5 task heads on K100AI.

For every sample in the task list this script:
  1. builds the official task prompt (verbatim upstream templates),
  2. runs one end-to-end image+text request on the HCU,
  3. parses the model output with the official regexes,
  4. writes the annotated image (point / bbox / trajectory) when the task is spatial,
  5. records whether the output was *parseable*, not merely non-empty.

The last point is what separates "the model ran" from "the model produced a
usable answer for this task".

Usage:
  python3 scripts/run_tasks.py --tasks samples/tasks_official.json \
      --out-dir outputs/tasks_official
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robo_infer import RoboBrain, parse_answer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=os.environ.get("MODEL_PATH"))
    ap.add_argument("--tasks", default=os.path.join(
        os.environ.get("WORK_DIR", "."), "samples", "tasks_official.json"))
    ap.add_argument("--image-dir", default=os.path.join(
        os.environ.get("WORK_DIR", "."), "samples", "images"))
    ap.add_argument("--out-dir", default="outputs/tasks_official")
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--do-sample", type=int, default=-1,
                    help="-1: follow the per-sample default in the task file")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.tasks, encoding="utf-8") as f:
        tasks = json.load(f)

    rb = RoboBrain(args.model_path, device_map=args.device_map)
    print(f"\nloaded in {rb.load_seconds:.1f}s on {rb.device}\n")

    results = []
    for t in tasks:
        image = os.path.join(args.image_dir, t["image"])
        mnt = int(t.get("max_new_tokens", 128))
        do_sample = bool(t.get("do_sample", False)) if args.do_sample < 0 else bool(args.do_sample)

        t0 = time.time()
        r = rb.infer(t["prompt"], [image], task=t["task"], max_new_tokens=mnt,
                     do_sample=do_sample, temperature=0.7,
                     plot=True, out_dir=args.out_dir)
        r["sample_id"] = t["id"]
        r["expectation"] = t.get("expect")
        r["wall_seconds"] = round(time.time() - t0, 3)

        parsed = parse_answer(r["answer"], t["task"])
        counts = {k: len(v) for k, v in parsed.items()}
        r["parsed_counts"] = counts
        r["parseable"] = bool(counts) if t["task"] != "general" else bool(r["answer"].strip())
        r["image"] = image
        results.append(r)

        print("=" * 72)
        print(f"[{r['sample_id']}] task={r['task']}  parseable={r['parseable']}  "
              f"counts={counts}")
        print("PROMPT:", r["model_prompt"])
        print("ANSWER:", r["answer"])
        if r.get("annotated_image"):
            print("ANNOTATED:", r["annotated_image"])
        print("=" * 72, flush=True)

    summary = {
        "model_load_seconds": round(rb.load_seconds, 2),
        "device": str(rb.device),
        "samples": len(results),
        "parseable": sum(1 for r in results if r["parseable"]),
        "results": results,
    }
    out = os.path.join(args.out_dir, "task_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + json.dumps(
        {k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False, indent=2))
    print("artifact:", out)
    print("peak vram reserved: %.2f GiB" % (torch.cuda.max_memory_reserved() / 1024**3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
