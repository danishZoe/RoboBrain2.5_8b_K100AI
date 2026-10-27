#!/usr/bin/env python3
"""Phase 2 acceptance: minimal end-to-end inference on the K100AI (HCU) device.

Acceptance criteria (AGENTS.md phase 2):
  * model + processor load, and the HCU actually takes part in the computation
  * at least one representative image+text request finishes end to end with a
    task-relevant, non-empty answer
  * the very same script/config runs twice without device errors, OOM, NaN/Inf
    or persistently empty output

The script itself is idempotent: run it with --repeat 2 to satisfy the
"same script twice" requirement in one go, or invoke it twice from the shell.
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robo_infer import RoboBrain, load_vision_inputs  # noqa: E402


def check_device(model):
    """Prove the HCU participates: parameters live on cuda:0 and a matmul lands there."""
    devs = {str(p.device) for p in model.model.parameters()}
    assert devs == {"cuda:0"}, f"unexpected parameter devices: {devs}"
    a = torch.randn(512, 512, device="cuda:0", dtype=torch.bfloat16)
    b = torch.randn(512, 512, device="cuda:0", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    c = a @ b
    torch.cuda.synchronize()
    assert torch.isfinite(c).all().item(), "NaN/Inf detected in device matmul"
    return {
        "name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "param_devices": sorted(devs),
        "vram_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=os.environ.get("MODEL_PATH"))
    ap.add_argument("--image", default=os.path.join(
        os.environ.get("WORK_DIR", "."), "samples", "images", "red_hat.png"))
    ap.add_argument("--prompt", default="What is shown in this image?")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--out-dir", default="outputs/phase2")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print("=" * 64)
    print("PHASE 2 ACCEPTANCE - RoboBrain2.5-8B-NV on K100AI")
    print("=" * 64)
    print(f"model : {args.model_path}")
    print(f"image : {args.image}")
    print(f"repeat: {args.repeat}")

    t0 = time.time()
    rb = RoboBrain(args.model_path, device_map="cuda:0")
    load_s = time.time() - t0

    dev_info = check_device(rb)
    print("\n[device check]", json.dumps(dev_info, ensure_ascii=False))

    results = []
    for i in range(args.repeat):
        print(f"\n--- run {i + 1}/{args.repeat} ---", flush=True)
        r = rb.infer(
            args.prompt, [args.image], task="general",
            max_new_tokens=args.max_new_tokens, do_sample=False, temperature=0.7,
            plot=False,
        )
        r["run_index"] = i + 1
        r["device"] = dev_info
        r["model_load_seconds"] = round(load_s, 2)
        results.append(r)
        print(f"answer  : {r['answer'][:400]}")
        print(f"tokens  : prompt={r['prompt_tokens']} gen={r['gen_tokens']}")
        print(f"e2e     : {r['e2e_seconds']}s")

    # ---- verdict -------------------------------------------------------
    failures = []
    for r in results:
        if not r["answer"].strip():
            failures.append(f"run{r['run_index']}: empty answer")
        if r["gen_tokens"] <= 0:
            failures.append(f"run{r['run_index']}: no token generated")
    peak = torch.cuda.max_memory_reserved() / 1024**3

    verdict = {
        "model_load_seconds": round(load_s, 2),
        "device": dev_info,
        "runs": results,
        "peak_vram_reserved_gib": round(peak, 2),
        "failures": failures,
        "passed": not failures,
    }
    out = os.path.join(args.out_dir, "phase2_result.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 64)
    print("VERDICT:", "PASS" if verdict["passed"] else "FAIL " + str(failures))
    print(f"peak vram (reserved): {peak:.2f} GiB")
    print("result file:", out)
    print("=" * 64)
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
