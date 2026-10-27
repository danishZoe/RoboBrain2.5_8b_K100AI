#!/usr/bin/env python3
"""Determinism probe for the grounding head.

Open question: the transformers baseline answered grounding with a bare
"[0, 309, 620, 989]" (18 tokens) while a fresh transformers process answered
with a long chain of thought ending in "[20, 290, 480, 780]" (120 tokens).
Same prompt, same pixels, greedy decoding. Either the two code paths differ,
or the greedy decision at the very first token is a near-tie that flips.

This script answers it directly:
  * runs the identical request N times inside one process,
  * prints the top-5 first-token probabilities (is the decision a near-tie?),
  * prints a fingerprint of the generated text.

Run it several times as SEPARATE processes to measure cross-process
stability. Every run appends one line to logs/determinism_probe.jsonl.

Usage (remote):
  python3 scripts/probe_determinism.py --runs 5
"""

import argparse
import json
import os
import time

import torch
from PIL import Image

WORK = os.environ.get("WORK_DIR", "/root/private_data/workspace/robo")
MODEL = os.environ.get("MODEL_PATH")
IMG = os.path.join(WORK, "samples", "images", "official_grounding.jpg")
PROMPT = ("Please provide the bounding box coordinate of the region this "
          "sentence describes: the person wearing a red hat.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(WORK, "logs",
                                                   "determinism_probe.jsonl"))
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL)
    tok = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype="auto", device_map="cuda:0")
    model.eval()

    messages = [{"role": "user", "content": [
        {"type": "image", "image": "file://" + IMG},
        {"type": "text", "text": PROMPT},
    ]}]
    chat = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    image = Image.open(IMG).convert("RGB")
    inputs = processor(text=[chat], images=[image], return_tensors="pt").to("cuda")

    pid = os.getpid()
    print("[probe] pid=%d n_prompt=%d" % (pid, inputs.input_ids.shape[1]),
          flush=True)

    rows = []
    for i in range(args.runs):
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=768, do_sample=False,
                return_dict_in_generate=True, output_scores=True)
        ids = out.sequences[0][inputs.input_ids.shape[1]:]
        text = processor.batch_decode(
            [ids], skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        probs = torch.softmax(out.scores[0][0].float(), dim=-1)
        top = torch.topk(probs, 5)
        top5 = [{"token": tok.convert_ids_to_tokens([int(t)])[0],
                 "prob": round(float(p), 6),
                 "id": int(t)} for t, p in zip(top.indices, top.values)]
        row = {
            "pid": pid,
            "iter": i,
            "n_new_tokens": int(ids.shape[0]),
            "text": text,
            "first_token_top5": top5,
            "wall_seconds": round(time.time() - t0, 3),
        }
        rows.append(row)
        print("[iter %d] pid=%d ntok=%d first=%r(%.4f) second=%r(%.4f) text=%r"
              % (i, pid, row["n_new_tokens"], top5[0]["token"],
                 top5[0]["prob"], top5[1]["token"], top5[1]["prob"],
                 text[:120]), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("[probe] appended %d rows to %s" % (len(rows), args.out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
