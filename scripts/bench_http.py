#!/usr/bin/env python3
"""Phase 4: measure an OpenAI-compatible serving backend (SGLang / vLLM) with the
same fixed samples and the same aggregation rules as the transformers baseline
(first repeat = warm-up, excluded), so the numbers are directly comparable.

Single-request latency and concurrent throughput are measured separately, because
concurrent runs change the effective batch and therefore per-request latency.

Usage:
  python3 scripts/bench_http.py --backend sglang --port 30000 --tag sglang_base \
      --tasks samples/tasks_official.json --repeat 3 --concurrency 1 4 8
"""

import argparse
import base64
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import urllib.request


def to_data_url(path):
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


def build_prompt(text, task):
    """Official RoboBrain2.5 task templates (verbatim upstream)."""
    if task == "pointing":
        return (f"{text}. Please provide its 2D coordinates. Your answer should be "
                f"formatted as a tuple, i.e. [(x, y)], where the tuple contains the x "
                f"and y coordinates of a point satisfying the conditions above.")
    if task == "trajectory":
        return (f"Please predict 3D end-effector-centric waypoints to complete the task "
                f'successfully. The task is "{text}". Your answer should be formatted as '
                f"a list of tuples, i.e., [(x1, y1, d1), (x2, y2, d2), ...], where each "
                f"tuple contains the x and y coordinates and the depth of the point.")
    if task == "grounding":
        return (f"Please provide the bounding box coordinate of the region this sentence "
                f"describes: {text}.")
    return text


def post_json(url, payload, timeout=900):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def wait_ready(base_url, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=10) as r:
                return True, time.time() - t0
        except Exception:
            time.sleep(5)
    return False, time.time() - t0


def one_request(base_url, model, sample, image_path, max_new_tokens, do_sample):
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": to_data_url(image_path)}},
                {"type": "text", "text": build_prompt(sample["prompt"], sample["task"])},
            ],
        }],
        "max_tokens": max_new_tokens,
        "temperature": 0.7 if do_sample else 0.0,
        "top_p": 0.8 if do_sample else 1.0,
    }
    t0 = time.time()
    r = post_json(f"{base_url}/v1/chat/completions", payload)
    total = time.time() - t0
    usage = r.get("usage", {}) or {}
    answer = r["choices"][0]["message"]["content"]
    gen = usage.get("completion_tokens") or 0
    return {
        "total_seconds": round(total, 4),
        "prompt_tokens": usage.get("prompt_tokens"),
        "gen_tokens": gen,
        "tok_per_s": round(gen / total, 3) if total else None,
        "answer": answer,
    }


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"n": len(vals), "mean": round(statistics.mean(vals), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4),
            "median": round(statistics.median(vals), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="sglang", choices=["sglang", "vllm"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--model", default="robobrain2.5-8b-nv")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--tasks", default=os.path.join(
        os.environ.get("WORK_DIR", "."), "samples", "tasks_official.json"))
    ap.add_argument("--image-dir", default=os.path.join(
        os.environ.get("WORK_DIR", "."), "samples", "images"))
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--out-dir", default="outputs/phase4")
    ap.add_argument("--wait-timeout", type=int, default=1800)
    args = ap.parse_args()

    tag = args.tag or args.backend
    base_url = f"http://{args.host}:{args.port}"
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.tasks, encoding="utf-8") as f:
        tasks = json.load(f)

    print(f"[{args.backend}] waiting for {base_url} ...", flush=True)
    ready, waited = wait_ready(base_url, timeout=args.wait_timeout)
    if not ready:
        print("SERVER NOT READY")
        return 1
    print(f"[{args.backend}] ready after {waited:.1f}s", flush=True)

    records = []
    answers = {}
    for conc in args.concurrency:
        print(f"\n=== concurrency {conc} ===", flush=True)
        for s in tasks:
            image = os.path.join(args.image_dir, s["image"])
            mnt = int(s.get("max_new_tokens", 128))
            do_sample = bool(s.get("do_sample", False))
            for rep in range(args.repeat):
                warmup = (rep == 0)
                t0 = time.time()
                if conc == 1:
                    r = one_request(base_url, args.model, s, image, mnt, do_sample)
                    rec = {"total_seconds": r["total_seconds"],
                           "prompt_tokens": r["prompt_tokens"],
                           "gen_tokens": r["gen_tokens"],
                           "tok_per_s": r["tok_per_s"],
                           "batch_seconds": round(time.time() - t0, 4),
                           "answer": r["answer"]}
                else:
                    with ThreadPoolExecutor(max_workers=conc) as ex:
                        futs = [ex.submit(one_request, base_url, args.model, s, image,
                                           mnt, do_sample) for _ in range(conc)]
                        burst = [f.result() for f in futs]
                    rec = {
                        "total_seconds": round(statistics.mean(b["total_seconds"] for b in burst), 4),
                        "prompt_tokens": burst[0]["prompt_tokens"],
                        "gen_tokens": sum(b["gen_tokens"] for b in burst),
                        "batch_seconds": round(time.time() - t0, 4),
                        "answer": burst[0]["answer"],
                    }
                    rec["tok_per_s"] = round(rec["gen_tokens"] / rec["batch_seconds"], 3)
                rec.update({"sample_id": s["id"], "task": s["task"], "concurrency": conc,
                            "repeat": rep + 1, "warmup": warmup})
                records.append(rec)
                if not warmup:
                    answers.setdefault(s["id"], rec["answer"])
                print(f"  {s['id']:22s} conc={conc} rep{rep+1} "
                      f"ptok={rec['prompt_tokens']} gtok={rec['gen_tokens']} "
                      f"total={rec['total_seconds']} batch={rec['batch_seconds']} "
                      f"tok/s={rec['tok_per_s']}", flush=True)

    steady = [r for r in records if not r["warmup"]]
    summary = {
        "tag": tag,
        "backend": args.backend,
        "endpoint": base_url,
        "samples": len(tasks),
        "repeats": args.repeat,
        "concurrency_levels": args.concurrency,
        "answers": answers,
    }
    for conc in args.concurrency:
        sub = [r for r in steady if r["concurrency"] == conc]
        summary[f"concurrency_{conc}"] = {
            "total_seconds_mean": agg([r["total_seconds"] for r in sub]),
            "batch_seconds_mean": agg([r["batch_seconds"] for r in sub]),
            "tok_per_s_mean": agg([r["tok_per_s"] for r in sub]),
            "gen_tokens_mean": agg([r["gen_tokens"] for r in sub]),
        }

    out = os.path.join(args.out_dir, f"{tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=2)

    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "answers"},
                             ensure_ascii=False, indent=2))
    print("artifact:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
