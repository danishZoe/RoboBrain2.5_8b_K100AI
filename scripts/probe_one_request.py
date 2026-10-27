#!/usr/bin/env python3
"""Issue exactly one image+text request against a running OpenAI-compatible server
and save the raw response. Used by probe_attn_backend.sh to validate a backend
end to end (not just "the port is open")."""

import argparse
import base64
import json
import os
import time
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30010)
    ap.add_argument("--model", default="robobrain2.5-8b-nv")
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default="What is shown in this image?")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.image, "rb") as f:
        data_url = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

    payload = {
        "model": args.model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": args.prompt},
            ],
        }],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
    }
    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            body = json.loads(r.read().decode())
    except Exception as e:
        print(f"  REQUEST FAILED: {type(e).__name__}: {e}")
        return 1
    dt = time.time() - t0
    answer = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {}) or {}
    print(f"  OK  {dt:.2f}s  prompt_tokens={usage.get('prompt_tokens')} "
          f"completion_tokens={usage.get('completion_tokens')}")
    print(f"  ANSWER: {answer[:300]}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"seconds": round(dt, 3), "usage": usage, "answer": answer},
                      f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
