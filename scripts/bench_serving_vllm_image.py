#!/usr/bin/env python3
"""Run the SGLang benchmark client against an OpenAI-compatible vLLM server
with the **image** dataset.

Why this wrapper is needed
--------------------------
The SGLang benchmark client shipped in this image (``sglang 0.5.12``) exposes
the multimodal dataset through ``sglang.benchmark.datasets.image`` and refuses
any backend outside::

    supported_backends = ["sglang", "sglang-native", "sglang-oai-chat"]
    if backend not in supported_backends:
        raise ValueError("Image dataset only supports backends: ...")

so ``--backend vllm --dataset-name image`` aborts before sending a single
request, even though the vLLM backend in the same client *does* build
OpenAI-style multimodal payloads (image_url entries + text).

The three accepted backends differ only in **who applies the chat template**:

* ``sglang`` / ``sglang-native``: ``/generate`` does not apply it, so the
  client sends the fully rendered template.
* ``sglang-oai-chat``: the server applies it, so the client sends raw text.

A vLLM OpenAI-compatible server (``/v1/chat/completions``) behaves exactly
like ``sglang-oai-chat``: it applies the chat template itself. This wrapper
therefore maps the ``vllm`` backend onto ``sglang-oai-chat`` for the dataset
builder only, and changes nothing else.

Nothing in the shared environment is modified: the substitution happens in
this process, at runtime.

Usage:
    python3 bench_serving_vllm_image.py --backend vllm --host 127.0.0.1 \
        --port 30000 --model robobrain2.5-8b-nv --tokenizer "$MODEL_PATH" \
        --dataset-name image --image-count 1 --image-resolution 1080p \
        --random-input-len 4096 --random-output-len 1024 --random-range-ratio 1 \
        --max-concurrency 8 --num-prompts 16 --seed 1 \
        --disable-ignore-eos --output-file bench/round2.jsonl
"""

import runpy
import sys

from sglang.benchmark.datasets import image as image_mod

_ORIG = image_mod.create_mm_data_row


def _patched(text_prompt, images, images_base64, output_len, processor,
             backend, *args, **kwargs):
    if backend == "vllm":
        # An OpenAI-compatible /v1/chat/completions server applies the chat
        # template itself -> same semantics as sglang-oai-chat.
        backend = "sglang-oai-chat"
    return _ORIG(text_prompt, images, images_base64, output_len, processor,
                 backend, *args, **kwargs)


image_mod.create_mm_data_row = _patched
print("[bench-wrapper] vllm image backend mapped to sglang-oai-chat semantics",
      flush=True)

if __name__ == "__main__":
    sys.argv[0] = "sglang.bench_serving"
    runpy.run_module("sglang.bench_serving", run_name="__main__")
