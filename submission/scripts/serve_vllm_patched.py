#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vLLM launcher with a minimal, in-workspace compatibility shim.

Nothing in site-packages is modified by this script.

--------------------------------------------------------------------------------
Root cause this shim fixes
--------------------------------------------------------------------------------
vLLM 0.15.1 loads a multimodal checkpoint through
``vllm/model_executor/models/qwen3.py::Qwen3ForCausalLM.load_weights``::

    skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None)

For RoboBrain2.5-8B-NV (Qwen3VLForConditionalGeneration) ``self.config`` is the
*text* config. transformers 5.3.0 keeps ``tie_word_embeddings`` on the
*composite* config only, so the attribute lookup on the text config raises::

    AttributeError: 'Qwen3VLTextConfig' object has no attribute 'tie_word_embeddings'

and the engine core dies during checkpoint loading ("Engine core initialization
failed").

The shim re-attaches the field to the text-config class, using the value from
the model's own ``config.json`` and cross-checking it against the weight index
(a stored ``lm_head.weight`` tensor is the ground truth for "not tied").
The equivalent upstream-style diff is kept in
``patches/0001-vllm-qwen3-tie-word-embeddings.patch`` for the submitted repo.

Usage (identical flags to ``vllm serve``)::

    python3 scripts/serve_vllm_patched.py \
        --model "$MODEL_PATH" --served-model-name robobrain2.5-8b-nv \
        --host 127.0.0.1 --port 8000 --dtype bfloat16 \
        --gpu-memory-utilization 0.85 --max-model-len 8192
"""
from __future__ import annotations

import json
import os
import runpy
import sys


def resolve_tie_word_embeddings(model_path: str) -> bool:
    """Return the model's real tie_word_embeddings value.

    config.json gives the declared value; the safetensors index gives the
    ground truth (a stored ``lm_head.weight`` means the head is NOT tied).
    """
    tie = False
    cfg_path = os.path.join(model_path, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            tie = bool(json.load(f).get("tie_word_embeddings", False))

    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path, encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})
        has_lm_head = any("lm_head" in name for name in weight_map)
        # Ground truth: a stored lm_head.weight tensor means the head is NOT tied,
        # so tie_word_embeddings must be False and vLLM must load the tensor.
        tie = not has_lm_head

    return tie


def apply_shim(model_path: str) -> bool:
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

    tie = resolve_tie_word_embeddings(model_path)
    if not hasattr(Qwen3VLTextConfig, "tie_word_embeddings"):
        Qwen3VLTextConfig.tie_word_embeddings = tie
        print(f"[shim] Qwen3VLTextConfig.tie_word_embeddings = {tie} "
              f"(model={model_path})", flush=True)
    else:
        print(f"[shim] Qwen3VLTextConfig already exposes tie_word_embeddings; "
              f"no change (declared={tie})", flush=True)
    return tie


def _model_from_argv(argv: list[str]) -> str | None:
    for i, a in enumerate(argv):
        if a == "--model" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--model="):
            return a.split("=", 1)[1]
    return os.environ.get("MODEL_PATH")


def main() -> int:
    argv = sys.argv[1:]
    model = _model_from_argv(argv)

    # vLLM spawns its EngineCore worker with VLLM_WORKER_MULTIPROC_METHOD=spawn,
    # so an in-process monkey-patch is NOT inherited by the worker. Export the
    # shim through sitecustomize so every interpreter (parent + children) applies
    # it before any model code runs.
    shim_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "envs", "vllm_shim")
    if os.path.isdir(shim_dir):
        os.environ["ROBO_VLLM_TIE_SHIM"] = "1"
        if model:
            os.environ["ROBO_MODEL_PATH"] = model
        existing = os.environ.get("PYTHONPATH", "")
        parts = [p for p in existing.split(os.pathsep) if p]
        if shim_dir not in parts:
            os.environ["PYTHONPATH"] = os.pathsep.join([shim_dir] + parts)
        print(f"[launcher] sitecustomize shim dir -> {shim_dir}", flush=True)

    if model:
        apply_shim(model)
    else:
        print("[shim] no --model / MODEL_PATH found; shim skipped", flush=True)

    # Execute the stock vLLM OpenAI server entry point unchanged.
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
