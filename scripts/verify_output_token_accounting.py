#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_output_token_accounting.py

Why
---
The official benchmark client reports two different output-token numbers:

* ``total_output_tokens``            -> planned length (or the server's
  ``usage.completion_tokens`` when the server emits usage in the stream)
* ``total_output_tokens_retokenized`` -> re-encoding of the *decoded* text

On the 16384-token tiers these two differ by up to 22.6% (131,072 vs 101,416),
while the vLLM engine's own counter shows ~99.7% of the planned tokens were
really generated. So the gap is *not* early EOS.

This script isolates the cause with a controlled pair:

  A) "gibberish" prompt - random token ids decoded to text, i.e. what the
     official client's random dataset produces;
  B) "natural" prompt   - ordinary English prose of the same length.

For each we compare ``usage.completion_tokens`` (what the engine generated)
with ``len(tokenizer.encode(text))`` (how many tokens the same text costs when
re-read) and measure how repetitive the produced text is.

Interpretation (both requests, same server, same settings):
  * A: engine tokens == planned, re-encoded length much smaller -> the model
    answers a gibberish prompt with highly repetitive text, which re-encodes
    into fewer tokens. The official ``output_throughput`` column stays valid.
  * B: engine tokens == re-encoded length -> no discrepancy for normal text.

Usage:
    python3 scripts/verify_output_token_accounting.py \
        --url http://127.0.0.1:30000 --model robobrain2.5-8b-nv \
        --tokenizer $MODEL_PATH --out outputs/phase5_token_accounting/result.json
"""
import argparse
import base64
import json
import os
import random
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "envs", "pylibs"))

PARAGRAPH = (
    "The service must keep the resident set within the memory budget while "
    "serving concurrent multimodal requests. Each request carries one image "
    "and a long textual context, and the engine schedules prefills and "
    "decodes in the same batch. Under such a workload the decode phase is "
    "usually bound by memory bandwidth, because every decoding step has to "
    "read the whole key value cache of every resident sequence. "
)

REPEAT_WORDS = ("the model produces a repetitive continuation that collapses when the "
                "same text is encoded again by the tokenizer, so the retokenized count "
                "is smaller than the number of tokens that were actually generated. ")


def build_payload(url, model, image_path, prompt, max_tokens):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img_b64}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_completion_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": False,
    }


def post(url, payload):
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.loads(r.read().decode())


def shingles(text, k=40):
    if len(text) <= k:
        return [text]
    return [text[i:i + k] for i in range(0, len(text) - k, k)]


def repetition(text, k=40):
    sh = shingles(text, k)
    if not sh:
        return 0.0
    return 1.0 - len(set(sh)) / float(len(sh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--model", default="robobrain2.5-8b-nv")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--image", default="samples/images/official_affordance.jpg")
    ap.add_argument("--input-tokens", type=int, default=16384)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="outputs/phase5_token_accounting/result.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rng = random.Random(args.seed)

    # --- prompt family A: random token ids decoded to text (official random dataset style)
    vocab_size = len(tok)
    ids = [rng.randrange(0, vocab_size) for _ in range(args.input_tokens)]
    gibberish = tok.decode(ids, skip_special_tokens=True)

    # --- prompt family B: natural prose of comparable token length
    natural = ""
    while len(tok.encode(natural, add_special_tokens=False)) < args.input_tokens:
        natural += PARAGRAPH
    natural_ids = len(tok.encode(natural, add_special_tokens=False))
    # the server rejects prompts whose text+vision+output exceed max_model_len;
    # trim natural prose to the same length as family A
    natural = tok.decode(
        tok.encode(natural, add_special_tokens=False)[: args.input_tokens],
        skip_special_tokens=True,
    )

    results = []
    for name, prompt in (("gibberish_random_tokens", gibberish), ("natural_prose", natural)):
        payload = build_payload(args.url, args.model, args.image, prompt, args.max_tokens)
        t0 = __import__("time").perf_counter()
        resp = post(args.url, payload)
        dt = __import__("time").perf_counter() - t0
        text = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
        gen = usage.get("completion_tokens")
        re_enc = len(tok.encode(text, add_special_tokens=False))
        rec = dict(
            family=name,
            prompt_text_tokens=len(tok.encode(prompt, add_special_tokens=False)),
            engine_completion_tokens=gen,
            retokenized_len=re_enc,
            ratio=round(re_enc / gen, 4) if gen else None,
            usage=usage,
            repeat_ratio_40gram=round(repetition(text), 4),
            text_len_chars=len(text),
            wall_s=round(dt, 2),
            finish_reason=resp["choices"][0].get("finish_reason"),
            text_head=text[:300],
            text_tail=text[-300:],
        )
        results.append(rec)
        print("[%s] prompt_tokens=%d engine_completion_tokens=%s retokenized=%d ratio=%s "
              "repeat40=%s chars=%d %.1fs" % (
                  name, rec["prompt_text_tokens"], gen, re_enc, rec["ratio"],
                  rec["repeat_ratio_40gram"], rec["text_len_chars"], rec["wall_s"]))
        print("    head: %s" % rec["text_head"].replace("\n", " ")[:200])
        print("    tail: %s" % rec["text_tail"].replace("\n", " ")[:200])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"image": args.image, "input_tokens": args.input_tokens,
                   "max_tokens": args.max_tokens, "natural_prompt_tokens": natural_ids,
                   "results": results}, f, ensure_ascii=False, indent=2)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
