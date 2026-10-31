#!/usr/bin/env python3
"""Output-format adapter between the RoboBrain2.5 backends and the official
parser.

Why this exists
---------------
The official parser (`third_party/RoboBrain2.5/inference.py`, lines 110-124)
extracts a grounding box with a full-text regex:

    box_pattern = r'\\[\\s*(\\d+)\\s*,\\s*(\\d+)\\s*,\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\]'

i.e. it only accepts the bracket form ``[x1, y1, x2, y2]``.

Measured on this node with the same weights and the same prompt:

  * transformers backend -> ``The bounding box ... is [20, 290, 480, 780]``
    -> already matches the official regex.
  * vLLM backend         -> ``The bounding box ... is `<bbox>0 298 622
    998</bbox>```  -> does NOT match the official regex, so the official
    pipeline would score this sample as unparseable even though the model
    produced a correct box.

The two backends differ because the model itself has two answer modes for
grounding (a bare box, and a prose analysis that ends in a box), and the two
backends land in different modes; within the prose mode the two backends
also disagree on the surface form of the box.

This module performs a *format bridge only*: the four integers are copied
verbatim, nothing is rounded, clipped, reordered or invented. The raw model
text is always kept alongside the normalised text so that the original
output stays auditable.

Nothing in here changes the model, the sampling parameters or the weights.
"""

import re

# ``<bbox>x1 y1 x2 y2</bbox>`` and the Qwen special-token variant.
RE_BBOX_TAG = re.compile(
    r"<bbox>\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)"
    r"\s+(-?\d+(?:\.\d+)?)\s*</bbox>")
RE_BOX_SPECIAL = re.compile(
    r"<\|box_start\|>\s*\[?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)"
    r"\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]?"
    r"\s*<\|box_end\|>")

# Official parsing regexes, copied verbatim from upstream inference.py.
RE_TRAJ = re.compile(r"(\d+),\s*(\d+),\s*([+-]?\d+\.\d+)")
RE_POINT = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
RE_BOX = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]")


def normalize_bbox(text: str) -> str:
    """Rewrite ``<bbox>x y x y</bbox>`` into the bracket form the official
    regex expects. The numbers are copied verbatim."""
    if not text:
        return text
    out = RE_BBOX_TAG.sub(
        lambda m: "[%s, %s, %s, %s]" % (m.group(1), m.group(2), m.group(3),
                                        m.group(4)),
        text)
    out = RE_BOX_SPECIAL.sub(
        lambda m: "[%s, %s, %s, %s]" % (m.group(1), m.group(2), m.group(3),
                                        m.group(4)),
        out)
    return out


def parse_answer(text: str, task: str):
    """Parse a model answer with the official regexes.

    The official regexes are applied to the *normalised* text, because the
    vLLM backend emits the ``<bbox>`` surface form. Returns the same shape
    as scripts/robo_infer.py:parse_answer.
    """
    norm = normalize_bbox(text or "")
    if task == "trajectory":
        return {"points": [[int(a), int(b), float(c)]
                           for a, b, c in RE_TRAJ.findall(norm)]}
    if task == "grounding":
        m = RE_BOX.search(norm)
        return {"bbox": [int(v) for v in m.groups()] if m else None}
    if task in ("pointing", "affordance"):
        return {"points": [{"x": int(a), "y": int(b)}
                           for a, b in RE_POINT.findall(norm)]}
    return {"text": norm.strip()}


def answer_with_parse(text: str, task: str):
    """Return (normalised_text, parsed) keeping the raw text separately."""
    norm = normalize_bbox(text or "")
    return norm, parse_answer(text, task)
