#!/usr/bin/env python3
"""Close the grounding first-token tie.

Findings so far:
  * transformers (fresh process) -> "The person wearing a red hat is ..."
  * vLLM                        -> "To determine the bounding box for ..."
  * recorded baseline           -> "[0, 309, 620, 989]"
  * first-token top-2 probabilities are identical (0.4056 / 0.4056)

So the very first token is a numerical near-tie and the whole generation
diverges from there. This script measures, in one process:

  1. the actual model dtype (bf16 would quantise tied logits to the same
     value, making the argmax tie-break decide the output),
  2. whether the two image-loading paths (raw PIL vs qwen_vl_utils
     process_vision_info, the latter used by scripts/robo_infer.py) yield
     bit-identical pixel_values,
  3. the top-10 first-token probabilities and raw logits for both paths.

Usage (remote):
  python3 scripts/probe_tie.py
"""

import json
import os
import sys
import time

import torch
from PIL import Image

WORK = os.environ.get("WORK_DIR", "/root/private_data/workspace/robo")
MODEL = os.environ.get("MODEL_PATH")
IMG = os.path.join(WORK, "samples", "images", "official_grounding.jpg")
PROMPT = ("Please provide the bounding box coordinate of the region this "
          "sentence describes: the person wearing a red hat.")


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor

    sys.path.insert(0, os.path.join(WORK, "scripts"))
    processor = AutoProcessor.from_pretrained(MODEL)
    tok = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype="auto", device_map="cuda:0")
    model.eval()
    p = next(model.parameters())
    print("[dtype] param dtype =", p.dtype, "| device =", p.device)
    try:
        print("[dtype] model.dtype =", model.dtype)
    except Exception:  # noqa: BLE001
        pass

    messages = [{"role": "user", "content": [
        {"type": "image", "image": "file://" + IMG},
        {"type": "text", "text": PROMPT},
    ]}]
    chat = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)

    pil_image = Image.open(IMG).convert("RGB")
    inputs_pil = processor(text=[chat], images=[pil_image], return_tensors="pt")

    # --- path B: the qwen_vl_utils route used by scripts/robo_infer.py ----
    qu_inputs = None
    try:
        from qwen_vl_utils import process_vision_info
        imgs, vids = process_vision_info(messages)
        print("[qwen_vl_utils] images=%s videos=%s"
              % ([getattr(i, "size", None) for i in (imgs or [])], vids))
        if imgs:
            qu_inputs = processor(text=[chat], images=imgs, videos=vids,
                                  padding=True, return_tensors="pt")
    except Exception as exc:  # noqa: BLE001
        print("[qwen_vl_utils] unavailable:", type(exc).__name__, exc)

    if qu_inputs is not None:
        same_pv = torch.equal(inputs_pil["pixel_values"], qu_inputs["pixel_values"])
        same_ids = torch.equal(inputs_pil["input_ids"], qu_inputs["input_ids"])
        maxdiff = float((inputs_pil["pixel_values"].float()
                         - qu_inputs["pixel_values"].float()).abs().max())
        print("[pixels] identical=%s  input_ids identical=%s  max|diff|=%.6f"
              % (same_pv, same_ids, maxdiff))
        print("[pixels] pil shape=%s  qu shape=%s"
              % (tuple(inputs_pil["pixel_values"].shape),
                 tuple(qu_inputs["pixel_values"].shape)))
    else:
        print("[pixels] qwen_vl_utils path unavailable, PIL only")

    watch = ["[", "The", "To"]
    watch_ids = {s: tok.encode(s, add_special_tokens=False)[0] for s in watch}
    print("[watch] token ids:", watch_ids)

    results = {}
    for label, inputs in (("PIL", inputs_pil),
                          ("QWEN_VL_UTILS", qu_inputs)):
        if inputs is None:
            continue
        inputs = inputs.to("cuda")
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=768, do_sample=False,
                                 return_dict_in_generate=True,
                                 output_scores=True)
        ids = out.sequences[0][inputs["input_ids"].shape[1]:]
        text = processor.batch_decode(
            [ids], skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        logits = out.scores[0][0].float()
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, 10)
        rows = [{"token": tok.convert_ids_to_tokens([int(i)])[0],
                 "id": int(i), "prob": round(float(p), 6),
                 "logit": round(float(logits[int(i)]), 6)}
                for p, i in zip(top.values, top.indices)]
        watched = []
        for s, tid in watch_ids.items():
            watched.append({"surface": s, "id": int(tid),
                            "prob": round(float(probs[tid]), 6),
                            "logit": round(float(logits[tid]), 6)})
        print("\n===== path %s =====" % label)
        print("  n_new_tokens:", int(ids.shape[0]))
        print("  text:", repr(text[:220]))
        for r in rows:
            print("   top: %-12s id=%-6d p=%.6f logit=%.6f"
                  % (r["token"], r["id"], r["prob"], r["logit"]))
        for r in watched:
            print("   watch %-6s id=%-6d p=%.6f logit=%.6f"
                  % (r["surface"], r["id"], r["prob"], r["logit"]))
        results[label] = {"text": text, "n_new_tokens": int(ids.shape[0]),
                           "top10": rows, "watched": watched}

    # --- third path: the exact robo_infer helper --------------------------
    try:
        from robo_infer import RoboBrain, build_prompt
        rb = RoboBrain(MODEL, device_map="cuda:0")
        out = rb.infer("the person wearing a red hat", [IMG], task="grounding",
                       max_new_tokens=64, do_sample=False, temperature=0.7)
        print("\n===== robo_infer.RoboBrain =====")
        print("  answer:", repr(out["answer"]))
        print("  prompt_tokens:", out["prompt_tokens"],
              "gen_tokens:", out["gen_tokens"])
        results["robo_infer"] = {"answer": out["answer"],
                                 "prompt_tokens": out["prompt_tokens"],
                                 "gen_tokens": out["gen_tokens"]}
    except Exception as exc:  # noqa: BLE001
        print("\n===== robo_infer.RoboBrain ===== failed:", type(exc).__name__, exc)
        results["robo_infer"] = {"error": "%s: %s" % (type(exc).__name__, exc)}

    path = os.path.join(WORK, "logs", "probe_tie.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n[dump]", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
