#!/usr/bin/env python3
"""RoboBrain2.5-8B-NV unified inference - offline port of the official implementation.

Source of truth for prompts / parsing / annotation:
    https://github.com/FlagOpen/RoboBrain2.5/blob/main/inference.py
    (mirror kept under third_party/RoboBrain2.5/ ; the upstream repo is unreachable
     from the K100AI node because outbound HTTP is blocked)

Compatibility notes for the Iluvatar (天数智芯) DTK stack:
  * ``torch.cuda`` is the vendor shim for the HCU device - the official code path
    (``device_map`` / ``inputs.to("cuda")``) is kept unchanged.
  * ``qwen_vl_utils`` is not in the base image; the pure-python wheel is vendored
    into envs/pylibs and injected via PYTHONPATH (see env.sh).
  * ``cv2`` 4.10.0 is present in the image, so annotation is identical to upstream.

All model/task prompt strings below are copied verbatim from upstream so that the
fine-tuned output format keeps matching.
"""

import argparse
import json
import os
import re
import sys
import time

import cv2
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

TASKS = ("general", "pointing", "trajectory", "grounding")


# --------------------------------------------------------------------------- #
# prompt construction (verbatim from upstream inference.py)
# --------------------------------------------------------------------------- #
def build_prompt(text: str, task: str) -> str:
    if task == "pointing":
        return (
            f"{text}. Please provide its 2D coordinates. Your answer should be "
            f"formatted as a tuple, i.e. [(x, y)], where the tuple contains the x "
            f"and y coordinates of a point satisfying the conditions above."
        )
    if task == "trajectory":
        return (
            f"Please predict 3D end-effector-centric waypoints to complete the task "
            f'successfully. The task is "{text}". Your answer should be formatted as '
            f"a list of tuples, i.e., [(x1, y1, d1), (x2, y2, d2), ...], where each "
            f"tuple contains the x and y coordinates and the depth of the point."
        )
    if task == "grounding":
        return (
            f"Please provide the bounding box coordinate of the region this sentence "
            f"describes: {text}."
        )
    return text


# --------------------------------------------------------------------------- #
# output parsing (verbatim regexes from upstream inference.py)
# --------------------------------------------------------------------------- #
def parse_answer(text: str, task: str):
    if task == "trajectory":
        pts = re.findall(r"(\d+),\s*(\d+),\s*([+-]?\d+\.\d+)", text)
        return {"trajectories": [[(int(x), int(y), float(d)) for x, y, d in pts]]}
    if task == "pointing":
        pts = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)
        return {"points": [(int(x), int(y)) for x, y in pts]}
    if task == "grounding":
        boxes = re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", text)
        return {"boxes": [[int(a), int(b), int(c), int(d)] for a, b, c, d in boxes]}
    return {}


def draw_on_image(image_path, output_path, points=None, boxes=None, trajectories=None):
    """Annotation identical to upstream (relative 0~1000 coords -> absolute pixels)."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    h, w = image.shape[:2]

    def rel_to_abs(x_rel, y_rel):
        x = int(round((x_rel / 1000.0) * w))
        y = int(round((y_rel / 1000.0) * h))
        return max(0, min(w - 1, x)), max(0, min(h - 1, y))

    for point in points or []:
        x, y = rel_to_abs(point[0], point[1])
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

    for box in boxes or []:
        x1, y1 = rel_to_abs(box[0], box[1])
        x2, y2 = rel_to_abs(box[2], box[3])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for trajectory in trajectories or []:
        if not trajectory or len(trajectory) < 2:
            continue
        abs_pts = [rel_to_abs(p[0], p[1]) for p in trajectory]
        for i in range(1, len(abs_pts)):
            cv2.line(image, abs_pts[i - 1], abs_pts[i], (0, 0, 255), 2)
        cv2.circle(image, abs_pts[0], 7, (0, 255, 0), -1)
        cv2.circle(image, abs_pts[-1], 7, (255, 0, 0), -1)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, image)
    return output_path


# --------------------------------------------------------------------------- #
# vision input handling
# --------------------------------------------------------------------------- #
def load_vision_inputs(messages):
    """Use upstream qwen_vl_utils when available, else a minimal PIL fallback."""
    try:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        return image_inputs, video_inputs, "qwen_vl_utils"
    except ImportError:
        image_inputs = []
        for msg in messages:
            for item in msg.get("content", []):
                if item.get("type") == "image":
                    path = item["image"]
                    if path.startswith("http"):
                        raise RuntimeError(
                            "remote image URLs need qwen_vl_utils (not installed)"
                        )
                    image_inputs.append(Image.open(path).convert("RGB"))
        return image_inputs, [], "pil_fallback"


class RoboBrain:
    def __init__(self, model_path, device_map="cuda:0", dtype="auto",
                 max_pixels=0, min_pixels=0):
        t0 = time.time()
        print("[load] model ...", flush=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=dtype, device_map=device_map
        )
        print("[load] processor ...", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_path)
        if max_pixels or min_pixels:
            ip = self.processor.image_processor
            if max_pixels:
                ip.max_pixels = int(max_pixels)
            if min_pixels:
                ip.min_pixels = int(min_pixels)
            size = getattr(ip, "size", None)
            if isinstance(size, dict):
                if max_pixels:
                    size["longest_edge"] = int(max_pixels)
                if min_pixels:
                    size["shortest_edge"] = int(min_pixels)
            print(f"[load] visual budget: max_pixels={getattr(ip, 'max_pixels', None)} "
                  f"min_pixels={getattr(ip, 'min_pixels', None)} size={size}", flush=True)
        self.model.eval()
        self.load_seconds = time.time() - t0
        self.device = next(self.model.parameters()).device
        print(f"[load] done in {self.load_seconds:.1f}s on {self.device}", flush=True)

    def infer(self, text, images, task="general", max_new_tokens=768,
              do_sample=True, temperature=0.7, plot=False, out_dir=None):
        assert task in TASKS, f"Invalid task type: {task}"
        if task != "general":
            assert len(images) == 1, "pointing/grounding/trajectory require exactly one image"

        prompt = build_prompt(text, task)
        messages = [{
            "role": "user",
            "content": [
                *[{"type": "image",
                   "image": p if p.startswith("http") else f"file://{p}"} for p in images],
                {"type": "text", "text": prompt},
            ],
        }]

        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, backend = load_vision_inputs(messages)
        inputs = self.processor(
            text=[chat_text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        prompt_tokens = int(inputs["input_ids"].shape[1])
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=do_sample, temperature=temperature,
            )
        torch.cuda.synchronize()
        e2e = time.time() - t0

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        answer = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        gen_tokens = int(trimmed[0].shape[-1])

        parsed = parse_answer(answer, task)
        result = {
            "task": task,
            "user_prompt": text,
            "model_prompt": prompt,
            "answer": answer,
            "parsed": parsed,
            "prompt_tokens": prompt_tokens,
            "gen_tokens": gen_tokens,
            "e2e_seconds": round(e2e, 3),
            "decode_tok_per_s": round(gen_tokens / e2e, 2) if e2e > 0 else None,
            "vision_backend": backend,
        }

        if plot and task in ("pointing", "trajectory", "grounding") and images:
            out_dir = out_dir or "outputs"
            base = os.path.basename(images[0]).replace(".", "_annotated.")
            out_path = os.path.join(out_dir, base)
            draw_on_image(
                images[0], out_path,
                points=parsed.get("points"),
                boxes=parsed.get("boxes"),
                trajectories=parsed.get("trajectories"),
            )
            result["annotated_image"] = out_path
        return result


def main():
    ap = argparse.ArgumentParser(description="RoboBrain2.5-8B-NV inference")
    ap.add_argument("--model-path", default=os.environ.get(
        "MODEL_PATH", "/public/home/scnjz2s4uc/SothisAI/model/Aihub/RoboBrain2.5-8B-NV/main/RoboBrain2.5-8B-NV"))
    ap.add_argument("--task", default="general", choices=list(TASKS))
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--image", action="append", required=True,
                    help="local image path (repeatable; general task only accepts several)")
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--do-sample", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--plot", type=int, default=0)
    ap.add_argument("--max-pixels", type=int, default=0,
                    help="cap the visual-token budget (processor level); 0 = model default")
    ap.add_argument("--min-pixels", type=int, default=0)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    model = RoboBrain(args.model_path, device_map=args.device_map,
                      max_pixels=args.max_pixels, min_pixels=args.min_pixels)
    result = model.infer(
        args.prompt, args.image, task=args.task,
        max_new_tokens=args.max_new_tokens,
        do_sample=bool(args.do_sample), temperature=args.temperature,
        plot=bool(args.plot), out_dir=args.out_dir,
    )
    print("\n" + "=" * 60)
    print("MODEL PROMPT:\n" + result["model_prompt"])
    print("-" * 60)
    print("ANSWER:\n" + result["answer"])
    print("-" * 60)
    print("PARSED:", json.dumps(result["parsed"], ensure_ascii=False))
    print(f"tokens prompt={result['prompt_tokens']} gen={result['gen_tokens']} "
          f"e2e={result['e2e_seconds']}s")
    print("=" * 60)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("wrote", args.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
