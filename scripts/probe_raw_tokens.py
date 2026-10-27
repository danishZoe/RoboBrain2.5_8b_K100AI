#!/usr/bin/env python3
"""Token-level probe of the transformers (official) path.

Purpose: locate why the transformers baseline answers grounding with a bare
"[0, 309, 620, 989]" while the vLLM server answers with a long chain of
thought ending in "<bbox>0 298 622 998</bbox>", even though both sides report
identical prompt token counts and both decode greedily.

This script records, for the grounding sample only:
  * the image-processor budget actually in effect (max_pixels / size / resample),
  * the chat text rendered by the processor and by the tokenizer (are they equal?),
  * the raw prompt token ids,
  * the raw generated token ids and their string forms,
  * the answer decoded with and without special tokens.

All outputs are written under $WORK_DIR/tmp/ so they can be diffed later
against the vLLM side (which returns token strings via logprobs).

Usage (remote):
  python3 scripts/probe_raw_tokens.py
"""

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
TMP = os.path.join(WORK, "tmp")
os.makedirs(TMP, exist_ok=True)


def dump(name, obj):
    path = os.path.join(TMP, name)
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(obj, str):
            f.write(obj)
        else:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    print("[dump]", path)


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL)
    tok = processor.tokenizer
    ip = processor.image_processor
    print("[proc] class:", type(ip).__name__)
    print("[proc] size:", getattr(ip, "size", None))
    print("[proc] max_pixels:", getattr(ip, "max_pixels", None),
          "min_pixels:", getattr(ip, "min_pixels", None))
    print("[proc] do_resize:", getattr(ip, "do_resize", None),
          "resample:", getattr(ip, "resample", None),
          "use_fast:", getattr(processor, "use_fast", None))
    print("[proc] image_mean:", getattr(ip, "image_mean", None))
    print("[proc] load %.1fs" % (time.time() - t0))

    pc_path = os.path.join(MODEL, "preprocessor_config.json")
    if os.path.exists(pc_path):
        print("[proc] preprocessor_config.json:",
              json.dumps(json.load(open(pc_path, encoding="utf-8")),
                         ensure_ascii=False))

    # Does the tokenizer know the bbox markers as special tokens?
    for s in ["<bbox>", "</bbox>", "<|box_start|>", "<|box_end|>",
              "[0, 309, 620, 989]", "<|vision_start|>"]:
        ids = tok.encode(s, add_special_tokens=False)
        print("[tok] %-22r -> %s %s" % (s, ids, tok.convert_ids_to_tokens(ids)))

    messages = [{"role": "user", "content": [
        {"type": "image", "image": "file://" + IMG},
        {"type": "text", "text": PROMPT},
    ]}]

    chat = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    chat_tok = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)
    print("[chat] processor-render == tokenizer-render ?", chat == chat_tok)
    print("[chat] repr:", repr(chat))
    dump("grounding_chat_official.txt", chat)

    image = Image.open(IMG).convert("RGB")
    inputs = processor(text=[chat], images=[image], return_tensors="pt")
    print("[inputs] input_ids:", tuple(inputs.input_ids.shape),
          "pixel_values:", tuple(inputs.pixel_values.shape))
    ids0 = inputs.input_ids[0].tolist()
    print("[inputs] n_prompt_tokens:", len(ids0))
    print("[inputs] tokens:", tok.convert_ids_to_tokens(ids0)[:60])
    dump("grounding_prompt_ids_transformers.json", ids0)

    print("[torch] cuda available:", torch.cuda.is_available(),
          "count:", torch.cuda.device_count())
    try:
        print("[torch] device 0:", torch.cuda.get_device_name(0))
    except Exception as exc:  # pragma: no cover
        print("[torch] device name failed:", exc)

    inputs = inputs.to("cuda")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype="auto", device_map="cuda:0")
    print("[model] loaded %.1fs" % (time.time() - t0))

    t1 = time.time()
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=768, do_sample=False)
    trimmed = gen[0][inputs.input_ids.shape[1]:]
    print("[gen] n_new_tokens:", int(trimmed.shape[0]),
          "wall %.2fs" % (time.time() - t1))

    gen_ids = trimmed.tolist()
    toks = tok.convert_ids_to_tokens(gen_ids)
    print("[gen] token strings:", toks)
    dump("grounding_gen_tokens_transformers.json",
         {"ids": gen_ids, "tokens": toks})

    dec_skip = processor.batch_decode(
        [trimmed], skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0]
    dec_keep = processor.batch_decode(
        [trimmed], skip_special_tokens=False,
        clean_up_tokenization_spaces=False)[0]
    print("[gen] decode(skip_special=True) :", repr(dec_skip))
    print("[gen] decode(skip_special=False):", repr(dec_keep))
    dump("grounding_raw_decode_transformers.txt",
         "skip_special_true: " + repr(dec_skip) + "\n\n"
         "skip_special_false: " + repr(dec_keep) + "\n")

    # Does the processor expose a bbox-aware post-processor?
    fn = getattr(processor, "post_process_image_text_to_text", None)
    print("[proc] has post_process_image_text_to_text:", fn is not None)
    if fn is not None:
        try:
            out = fn(dec_skip)
            print("[proc] post_process(dec_skip) ->", repr(out))
        except Exception as exc:
            print("[proc] post_process failed:", type(exc).__name__, exc)
        try:
            out = fn([dec_skip])
            print("[proc] post_process([dec_skip]) ->", repr(out))
        except Exception as exc:
            print("[proc] post_process(list) failed:", type(exc).__name__, exc)

    print("[done] total %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
