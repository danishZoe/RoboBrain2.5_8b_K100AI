"""Auto-imported by EVERY Python interpreter that has this directory on PYTHONPATH.

That includes the parent vLLM process AND every ``multiprocessing`` child:
vLLM 0.15.1 uses ``VLLM_WORKER_MULTIPROC_METHOD=spawn`` by default, so a
monkey-patch applied only in the parent process is NOT inherited by the
EngineCore worker — the patch has to live in ``sitecustomize``.

Activated only when ``ROBO_VLLM_TIE_SHIM=1``.

Why the shim exists
--------------------
``vllm/model_executor/models/qwen3.py::Qwen3ForCausalLM.load_weights`` reads::

    skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None)

For RoboBrain2.5-8B-NV (``Qwen3VLForConditionalGeneration``) ``self.config`` is
the *text* config. transformers 5.3.0 keeps ``tie_word_embeddings`` on the
*composite* config only, so the lookup raises::

    AttributeError: 'Qwen3VLTextConfig' object has no attribute 'tie_word_embeddings'

and the engine core dies while loading the checkpoint.

The value is taken from the model's own ``config.json`` and cross-checked against
the safetensors index: a stored ``lm_head.weight`` tensor means the LM head is
**not** tied, so ``tie_word_embeddings`` must be ``False`` and vLLM must load it.

Nothing in site-packages is modified.
"""
import os


def _apply_tie_shim():
    import json

    model_path = os.environ.get("ROBO_MODEL_PATH") or os.environ.get("MODEL_PATH") or ""
    tie = False
    if model_path:
        cfg_path = os.path.join(model_path, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                tie = bool(json.load(f).get("tie_word_embeddings", False))
        idx_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(idx_path):
            with open(idx_path, encoding="utf-8") as f:
                weight_map = json.load(f).get("weight_map", {})
            # ground truth: a stored lm_head tensor means the head is NOT tied
            tie = not any("lm_head" in name for name in weight_map)

    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

    if not hasattr(Qwen3VLTextConfig, "tie_word_embeddings"):
        Qwen3VLTextConfig.tie_word_embeddings = tie
    return tie


if os.environ.get("ROBO_VLLM_TIE_SHIM") == "1":
    try:
        _tie = _apply_tie_shim()
        print(f"[sitecustomize] Qwen3VLTextConfig.tie_word_embeddings = {_tie}",
              flush=True)
    except Exception as exc:  # never break the interpreter
        print(f"[sitecustomize] tie shim skipped: {exc}", flush=True)


# This file shadows the distribution's own sitecustomize on PYTHONPATH, so run it.
try:
    import importlib.util as _ilu

    _dist_sitecustomize = "/usr/lib/python3.10/sitecustomize.py"
    if os.path.exists(_dist_sitecustomize):
        _spec = _ilu.spec_from_file_location("_dist_sitecustomize",
                                             _dist_sitecustomize)
        if _spec and _spec.loader:
            _mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
except Exception:
    pass
