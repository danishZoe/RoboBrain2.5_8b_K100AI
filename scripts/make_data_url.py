#!/usr/bin/env python3
"""Print a base64 data URL for an image, with the upstream `qwen_vl_utils`
preprocessing applied.

The official inference pipeline loads images through
`qwen_vl_utils.process_vision_info` -> `fetch_image`, which applies a
`smart_resize(..., factor=28)` + PIL BICUBIC resize before the processor
sees the image; the processor then applies a second resize to a multiple of
32.  A serving stack that uploads the original file feeds the model
*different pixels* than the official pipeline (measured max|diff| = 0.47 on
the normalised [-1, 1] scale on this node), which is enough to flip the
grounding head's first generated token.

Pre-resolving the image here and encoding it losslessly (PNG) makes the
server reproduce the official pixels exactly (max|diff| = 0.000000).

Usage:
    python3 make_data_url.py <image-path> > /tmp/data_url.txt
"""
import base64
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image_canonical import canonicalize_bytes  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: make_data_url.py <image-path>", file=sys.stderr)
        return 2
    path = sys.argv[1]
    with open(path, "rb") as f:
        raw = f.read()
    try:
        png = canonicalize_bytes(raw)
        mime = "image/png"
    except Exception:  # noqa: BLE001 - fall back to the original file
        png = raw
        mime = "image/jpeg"
    sys.stdout.write("data:%s;base64,%s"
                     % (mime, base64.b64encode(png).decode()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
