#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Send ONE OpenAI-compatible chat request with an image to a local serving
backend (vLLM / SGLang) and verify the answer is a valid RoboBrain output.

Usage:
  python3 scripts/probe_http_request.py --backend vllm --port 8000 \
      --task pointing --prompt "the affordance area for holding the cup" \
      --image samples/images/official_affordance.jpg

  python3 scripts/probe_http_request.py --backend vllm --port 8000 \
      --tasks samples/tasks_official.json --image-dir samples/images \
      --repeat 2 --out-dir outputs/probe_vllm

The script is intentionally self-contained (stdlib + requests) so it can be
re-run without the workspace venv. It writes raw responses to --out-dir so
that every measurement stays auditable.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time

import requests

# Official prompt templates, verbatim from FlagOpen/RoboBrain2.5/inference.py
TASK_TEMPLATES = {
    "general": "{text}",
    "grounding": (
        "Please provide the bounding box coordinate of the region this sentence "
        "describes: {text}."
    ),
    "pointing": (
        "{text}. Please provide its 2D coordinates. Your answer should be "
        "formatted as a tuple, i.e. [(x, y)], where the tuple contains the x and "
        "y coordinates of a point satisfying the conditions above."
    ),
    "trajectory": (
        "Please predict 3D end-effector-centric waypoints to complete the task "
        'successfully. The task is "{text}". Your answer should be formatted as a '
        "list of tuples, i.e., [(x1, y1, d1), (x2, y2, d2), ...], where each tuple "
        "contains the x and y coordinates and the depth of the point."
    ),
}

RE_TRAJ = re.compile(r"(\d+)\s*,\s*(\d+)\s*,\s*([+-]?\d+\.\d+)")
RE_POINT = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
RE_BOX = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_adapter import normalize_bbox  # noqa: E402
from answer_adapter import parse_answer as _parse_answer  # noqa: E402
from image_canonical import canonicalize_to_data_url  # noqa: E402

# Canonicalise the uploaded image so the server sees the same pixels as the
# official upstream pipeline. Flipped off with --no-canonical for A/B tests.
CANONICAL = True


def parse_answer(task: str, text: str):
    """Official regexes, applied after the ``<bbox>`` format bridge.

    The vLLM backend emits ``<bbox>x1 y1 x2 y2</bbox>`` while the official
    regex only accepts ``[x1, y1, x2, y2]``; see scripts/answer_adapter.py.
    """
    return _parse_answer(text, task)


def build_prompt(task: str, prompt: str) -> str:
    tpl = TASK_TEMPLATES.get(task, TASK_TEMPLATES["general"])
    return tpl.format(text=prompt)


def encode_image(path: str) -> str:
    """Base64 PNG data URL.

    By default the image is first passed through the upstream
    `qwen_vl_utils` loader so the server receives exactly the pixels the
    official pipeline feeds the model (see scripts/image_canonical.py).
    Use --no-canonical to upload the original file instead.
    """
    if not CANONICAL:
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        with open(path, "rb") as f:
            return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"
    return canonicalize_to_data_url(path)


def one_request(url: str, model: str, task: str, prompt: str, image_paths,
                max_tokens: int, timeout: int = 180):
    """Send one chat completion. `image_paths` may be a single path or a list."""
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    content = [{"type": "image_url", "image_url": {"url": encode_image(p)}}
               for p in image_paths]
    content.append({"type": "text", "text": build_prompt(task, prompt)})
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
    }
    t0 = time.perf_counter()
    r = requests.post(url, json=body, timeout=timeout)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    gen = usage.get("completion_tokens") or 0
    return {
        "text": text,
        "text_normalized": normalize_bbox(text),
        "usage": usage,
        "seconds": round(dt, 4),
        "gen_tokens": gen,
        "tok_per_s": round(gen / dt, 4) if dt > 0 and gen else None,
        "parsed": parse_answer(task, text),
        "finish_reason": data["choices"][0].get("finish_reason"),
        "num_images": len(image_paths),
        "raw": data,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="vllm", choices=["vllm", "sglang"])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default="robobrain2.5-8b-nv")
    ap.add_argument("--task", default="pointing")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--image", default="")
    ap.add_argument("--images", default="",
                    help="comma-separated image files (multi-image case)")
    ap.add_argument("--tasks", default="", help="samples/tasks_official.json")
    ap.add_argument("--image-dir", default="")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--override-max-tokens", type=int, default=0,
                    help="use this budget for every sample (official upstream "
                         "inference.py uses max_new_tokens=768)")
    ap.add_argument("--out-dir", default="outputs/probe_http")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--no-canonical", action="store_true",
                    help="upload the original image instead of applying the "
                         "upstream qwen_vl_utils preprocessing")
    args = ap.parse_args()

    global CANONICAL
    CANONICAL = not args.no_canonical

    url = (f"http://{args.host}:{args.port}/v1/chat/completions"
           if args.backend == "vllm"
           else f"http://{args.host}:{args.port}/generate")

    os.makedirs(args.out_dir, exist_ok=True)

    jobs = []
    if args.tasks:
        with open(args.tasks, encoding="utf-8") as f:
            spec = json.load(f)
        # accept both a top-level list and {"samples": [...]}
        samples = spec["samples"] if isinstance(spec, dict) else spec
        for s in samples:
            # a sample may carry "image" (single) or "images" (list -> multi-image case)
            imgs = s.get("images")
            if not imgs:
                one = s.get("image") or s.get("image_path") or s.get("file")
                imgs = [one] if one else []
            paths = [os.path.join(args.image_dir, n) for n in imgs]
            jobs.append((s.get("id") or s.get("sample_id") or imgs[0],
                         s.get("task", "general"),
                         s.get("prompt") or s.get("text") or "",
                         paths,
                         int(args.override_max_tokens or
                             s.get("max_new_tokens") or args.max_tokens)))
    elif args.images:
        paths = [os.path.join(args.image_dir, n) if args.image_dir else n
                 for n in args.images.split(",") if n.strip()]
        jobs.append(("adhoc_multi", args.task, args.prompt, paths,
                     args.max_tokens))
    else:
        jobs.append(("adhoc", args.task, args.prompt, [args.image],
                     args.max_tokens))

    records = []
    for rep in range(args.repeat):
        for sid, task, prompt, img_paths, max_tok in jobs:
            print(f"[{args.backend}] rep{rep} {sid} task={task} "
                  f"images={len(img_paths)}", flush=True)
            try:
                res = one_request(url, args.model, task, prompt, img_paths,
                                  max_tok, args.timeout)
            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {e}", flush=True)
                records.append({"sample_id": sid, "task": task, "error":
                                f"{type(e).__name__}: {e}"})
                continue
            rec = {"sample_id": sid, "task": task, "repeat": rep,
                   "prompt": prompt, "images": img_paths, **res}
            records.append(rec)
            print(f"  -> {res['text'][:160]}", flush=True)
            print(f"     PARSED {res['parsed']}  {res['seconds']}s "
                  f"{res['tok_per_s']} tok/s", flush=True)

    out = os.path.join(args.out_dir, "probe_result.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"backend": args.backend, "url": url, "records": records},
                  f, ensure_ascii=False, indent=2)
    ok = sum(1 for r in records if "error" not in r)
    parsed_ok = sum(1 for r in records
                    if "error" not in r and r["parsed"] and
                    (r["parsed"].get("points") or r["parsed"].get("bbox")
                     or r["parsed"].get("text")))
    print(f"\nREQUESTS_OK={ok}/{len(records)}  PARSED={parsed_ok}/{len(records)}")
    print(f"wrote {out}")
    return 0 if ok == len(records) else 1


if __name__ == "__main__":
    sys.exit(main())
