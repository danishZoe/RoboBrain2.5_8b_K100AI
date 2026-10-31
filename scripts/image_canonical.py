#!/usr/bin/env python3
"""Client-side image canonicalisation.

Why
---
The official upstream pipeline (`third_party/RoboBrain2.5/inference.py`) loads
images through `qwen_vl_utils.process_vision_info` -> `fetch_image`, which
applies `smart_resize(..., factor=28)` and a PIL **BICUBIC** resize before the
processor ever sees the image. The processor then applies a *second* resize
to a multiple of 32 (`patch_size 16 * merge_size 2`).

A serving stack that uploads the original file therefore feeds the model
**different pixels** than the official pipeline:

    official path : 426x640 --PIL bicubic--> 420x644 --torchvision bicubic--> 416x640
    serving path  : 426x640 --------------- torchvision bicubic ---------> 416x640

Measured on this node, the two pixel tensors differ by up to 0.47 on the
normalised [-1, 1] scale, and that difference flips the grounding head's
first-token decision (see reports/09_vllm_adaptation.md).

Fix
---
Pre-resolve the image with the very same `qwen_vl_utils` loader before
uploading it, using a lossless encoding (PNG) so the pixels survive the
round trip. The server then performs only the second resize and reproduces
the official pixels exactly:

    official  vs canonicalised : max|diff| = 0.000000
    official  vs raw upload    : max|diff| = 0.470588

This changes only the input preprocessing, never the weights, the sampling
parameters or the output text.
"""

import io
import math
import os

from PIL import Image

# Mirrors qwen_vl_utils.vision_process defaults.
SPATIAL_MERGE_SIZE = 2
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
DEFAULT_PATCH_SIZE = 14  # qwen_vl_utils.fetch_image default

try:
    from qwen_vl_utils.vision_process import fetch_image as _qwen_fetch_image
except Exception:  # noqa: BLE001 - fall back to the local re-implementation
    _qwen_fetch_image = None


def _round_by_factor(number, factor):
    return round(number / factor) * factor


def _smart_resize(height, width, factor, min_pixels=None, max_pixels=None):
    """Local copy of qwen_vl_utils.vision_process.smart_resize."""
    max_pixels = (max_pixels if max_pixels is not None
                  else IMAGE_MAX_TOKEN_NUM * factor ** 2)
    min_pixels = (min_pixels if min_pixels is not None
                  else IMAGE_MIN_TOKEN_NUM * factor ** 2)
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _to_rgb(image):
    if image.mode == "RGBA":
        white = Image.new("RGB", image.size, (255, 255, 255))
        white.paste(image, mask=image.split()[3])
        return white
    return image.convert("RGB")


def canonicalize_image(image: Image.Image, patch_size: int = DEFAULT_PATCH_SIZE):
    """Return the image after the upstream `qwen_vl_utils` resize step."""
    if _qwen_fetch_image is not None:
        # Feed the already-open PIL object through the upstream loader so the
        # algorithm cannot drift from the official one.
        return _qwen_fetch_image({"image": image}, image_patch_size=patch_size)
    factor = patch_size * SPATIAL_MERGE_SIZE
    image = _to_rgb(image)
    width, height = image.size
    h_bar, w_bar = _smart_resize(height, width, factor=factor)
    return image.resize((w_bar, h_bar))  # PIL default resample = BICUBIC


def canonicalize_file(path: str, patch_size: int = DEFAULT_PATCH_SIZE) -> Image.Image:
    with Image.open(path) as im:
        return canonicalize_image(im.copy(), patch_size=patch_size)


def canonicalize_bytes(data: bytes, patch_size: int = DEFAULT_PATCH_SIZE) -> bytes:
    """Canonicalise raw image bytes and return lossless PNG bytes."""
    with Image.open(io.BytesIO(data)) as im:
        canonical = canonicalize_image(im.copy(), patch_size=patch_size)
    buf = io.BytesIO()
    canonical.save(buf, format="PNG")
    return buf.getvalue()


def canonicalize_to_data_url(path: str, patch_size: int = DEFAULT_PATCH_SIZE) -> str:
    """Return a base64 PNG data URL with the official preprocessing applied."""
    import base64
    png = canonicalize_bytes(open(path, "rb").read(), patch_size=patch_size)
    return "data:image/png;base64," + base64.b64encode(png).decode()


if __name__ == "__main__":
    import base64
    import sys
    for p in sys.argv[1:]:
        img = canonicalize_file(p)
        out = os.path.splitext(p)[0] + "_canonical.png"
        img.save(out)
        print("%s -> %s size=%s bytes=%d" % (p, out, img.size, os.path.getsize(out)))
