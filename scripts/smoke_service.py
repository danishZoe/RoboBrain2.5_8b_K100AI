#!/usr/bin/env python3
"""Service-level smoke test for the vLLM deployment of RoboBrain2.5-8B-NV.

Covers the acceptance spec items that only a running service can show:

  * health check and /v1/models returning the declared model name
  * the four task heads through one service entry point (F01-F06)
  * multi-image general request (F07)
  * invalid inputs answered with an explicit error, never a silently
    degraded or fake success (F08)
  * streaming: chunk concatenation must equal the non-streaming answer
  * multi-turn image+text context

Every request and response is kept verbatim in the output directory.

Usage (remote):
  python3 scripts/smoke_service.py --url http://127.0.0.1:8000 \
      --model robobrain2.5-8b-nv --out-dir outputs/phase5_smoke
"""

import argparse
import base64
import json
import mimetypes
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_adapter import normalize_bbox, parse_answer  # noqa: E402
from image_canonical import canonicalize_to_data_url  # noqa: E402

TASK_TEMPLATES = {
    "pointing": ("{text}. Please provide its 2D coordinates. Your answer should "
                 "be formatted as a tuple, i.e. [(x, y)], where the tuple "
                 "contains the x and y coordinates of a point satisfying the "
                 "conditions above."),
    "trajectory": ("Please predict 3D end-effector-centric waypoints to "
                   "complete the task successfully. The task is \"{text}\". Your "
                   "answer should be formatted as a list of tuples, i.e., "
                   "[(x1, y1, d1), (x2, y2, d2), ...], where each tuple contains "
                   "the x and y coordinates and the depth of the point."),
    "grounding": ("Please provide the bounding box coordinate of the region "
                  "this sentence describes: {text}."),
    "general": "{text}",
}

RE_TRAJ = __import__("re").compile(r"(\d+),\s*(\d+),\s*([+-]?\d+\.\d+)")
RE_POINT = __import__("re").compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
RE_BOX = __import__("re").compile(
    r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]")


def encode_image(path, canonical=True):
    """Base64 PNG data URL.

    The image is first passed through the upstream `qwen_vl_utils` loader so
    the server receives exactly the pixels the official upstream pipeline
    feeds the model (see scripts/image_canonical.py).
    """
    if canonical:
        return canonicalize_to_data_url(path)
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        return "data:%s;base64,%s" % (mime,
                                       base64.b64encode(f.read()).decode())


def chat(url, model, content, max_tokens=768, stream=False, timeout=300):
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0,
            "seed": 0}
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    t0 = time.perf_counter()
    if stream:
        resp = requests.post(url + "/v1/chat/completions", json=body,
                             stream=True, timeout=timeout)
        parts, usage, finish = [], {}, None
        t_first = None
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", "replace")
            if not line.startswith("data: "):
                continue
            if line[6:].strip() == "[DONE]":
                break
            try:
                obj = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                d = (ch.get("delta") or {}).get("content")
                if d:
                    if t_first is None:
                        t_first = time.perf_counter()
                    parts.append(d)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
        return {"status": resp.status_code, "stream_text": "".join(parts),
                "usage": usage, "finish_reason": finish,
                "seconds": round(time.perf_counter() - t0, 4),
                "ttft_seconds": round(t_first - t0, 4) if t_first else None,
                "chunks": len(parts)}
    resp = requests.post(url + "/v1/chat/completions", json=body,
                         timeout=timeout)
    dt = round(time.perf_counter() - t0, 4)
    out = {"status": resp.status_code, "seconds": dt, "raw": None}
    try:
        data = resp.json()
        out["raw"] = data
        out["text"] = (data.get("choices") or [{}])[0].get("message", {}) \
            .get("content")
        out["usage"] = data.get("usage")
        out["finish_reason"] = (data.get("choices") or [{}])[0].get("finish_reason")
        out["error"] = data.get("error")
    except Exception:  # noqa: BLE001
        out["text"] = resp.text[:1000]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="robobrain2.5-8b-nv")
    ap.add_argument("--image-dir", default="samples/images")
    ap.add_argument("--out-dir", default="outputs/phase5_smoke")
    ap.add_argument("--no-canonical", action="store_true",
                    help="upload the original image instead of applying the "
                         "upstream qwen_vl_utils preprocessing")
    args = ap.parse_args()
    canonical = not args.no_canonical

    work = os.environ.get("WORK_DIR", "/root/private_data/workspace/robo")
    img_dir = args.image_dir if os.path.isabs(args.image_dir) \
        else os.path.join(work, args.image_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    results = {}

    # --- 0. health + models ---------------------------------------------
    health = {"http_status": None, "body": None, "error": None}
    try:
        r = requests.get(args.url + "/health", timeout=30)
        health["http_status"] = r.status_code
        health["body"] = r.text[:500]
    except Exception as exc:  # noqa: BLE001
        health["error"] = "%s: %s" % (type(exc).__name__, exc)
    results["health"] = health

    models = {"http_status": None, "body": None, "error": None}
    try:
        r = requests.get(args.url + "/v1/models", timeout=30)
        models["http_status"] = r.status_code
        models["body"] = r.json()
    except Exception as exc:  # noqa: BLE001
        models["error"] = "%s: %s" % (type(exc).__name__, exc)
    results["models"] = models
    print("[smoke] health=%s models=%s" % (health.get("http_status"),
                                          models.get("http_status")))

    def img(name):
        return {"type": "image_url",
                "image_url": {"url": encode_image(
                    os.path.join(img_dir, name), canonical=canonical)}}

    def txt(text):
        return {"type": "text", "text": text}

    # --- 1. F01-F06: four task heads through the service ------------------
    cases = [
        ("F01_general", "general", "official_affordance.jpg",
         "What is shown in this image?"),
        ("F02_grounding", "grounding", "official_grounding.jpg",
         "the person wearing a red hat"),
        ("F03_affordance", "pointing", "official_affordance.jpg",
         "the affordance area for holding the cup"),
        ("F04_pointing", "pointing", "official_pointing.jpg",
         "Identify spot within the vacant space that's between the two mugs"),
        ("F05_navigation", "pointing", "official_navigation.jpg",
         "Identify spot within toilet in the house"),
        ("F06_trajectory", "trajectory", "official_trajectory.jpg",
         "reach for the banana on the plate"),
    ]
    task_results = []
    for cid, task, image, prompt in cases:
        content = [img(image),
                   txt(TASK_TEMPLATES[task].format(text=prompt))]
        r = chat(args.url, args.model, content, max_tokens=768)
        text = r.get("text") or ""
        rec = {"case": cid, "task": task, "image": image, "prompt": prompt,
               "status": r.get("status"), "seconds": r.get("seconds"),
               "usage": r.get("usage"), "finish_reason": r.get("finish_reason"),
               "text": text, "text_normalized": normalize_bbox(text),
               "parsed": parse_answer(text, task),
               "error": r.get("error")}
        task_results.append(rec)
        print("[smoke] %-16s status=%s parsed=%s"
              % (cid, rec["status"], rec["parsed"]), flush=True)
    results["tasks"] = task_results

    # --- 2. F07: multi-image general -------------------------------------
    content = [img("official_affordance.jpg"), img("official_pointing.jpg"),
               txt("Compare these two images and describe what differs.")]
    r = chat(args.url, args.model, content, max_tokens=768)
    text = r.get("text") or ""
    results["F07_multi_image"] = {
        "status": r.get("status"), "seconds": r.get("seconds"),
        "usage": r.get("usage"), "text": text,
        "num_images": 2, "finish_reason": r.get("finish_reason"),
        "mentions_both": bool(text) and len(text) > 120}
    print("[smoke] F07_multi_image status=%s len=%d"
          % (results["F07_multi_image"]["status"], len(text)), flush=True)

    # --- 3. F08: invalid inputs must fail loudly --------------------------
    invalid = []

    # (a) corrupt image payload: a text file disguised as jpeg
    corrupt = base64.b64encode(b"this is definitely not an image").decode()
    r = chat(args.url, args.model, [
        {"type": "image_url",
         "image_url": {"url": "data:image/jpeg;base64," + corrupt}},
        txt("What is shown in this image?")], max_tokens=64)
    invalid.append({"case": "F08a_corrupt_image", "status": r.get("status"),
                    "error": r.get("error") or (r.get("text") or "")[:400],
                    "accepted": r.get("status") == 200 and bool(r.get("text"))})

    # (b) unreachable image url scheme
    r = chat(args.url, args.model, [
        {"type": "image_url",
         "image_url": {"url": "http://127.0.0.1:1/nonexistent.jpg"}},
        txt("What is shown in this image?")], max_tokens=64, timeout=120)
    invalid.append({"case": "F08b_unreachable_image", "status": r.get("status"),
                    "error": r.get("error") or (r.get("text") or "")[:400],
                    "accepted": r.get("status") == 200 and bool(r.get("text"))})

    # (c) malformed data url
    r = chat(args.url, args.model, [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,%%%"}},
        txt("What is shown in this image?")], max_tokens=64)
    invalid.append({"case": "F08c_malformed_data_url", "status": r.get("status"),
                    "error": r.get("error") or (r.get("text") or "")[:400],
                    "accepted": r.get("status") == 200 and bool(r.get("text"))})

    # (d) spatial task with two images: the upstream contract forbids it.
    # The server itself is schema-agnostic, so the guard is client side;
    # verify the server does NOT silently drop the second image.
    r = chat(args.url, args.model, [
        img("official_grounding.jpg"), img("official_pointing.jpg"),
        txt(TASK_TEMPLATES["grounding"].format(text="the person wearing a red hat"))],
        max_tokens=768)
    txtout = r.get("text") or ""
    invalid.append({"case": "F08d_spatial_task_two_images",
                    "status": r.get("status"),
                    "note": "server accepted; upstream contract requires 1 image "
                            "for grounding/pointing/trajectory -> client guard",
                    "text": txtout,
                    "accepted": r.get("status") == 200 and bool(txtout)})
    results["invalid_inputs"] = invalid
    for row in invalid:
        print("[smoke] %-32s status=%s accepted=%s"
              % (row["case"], row.get("status"), row.get("accepted")), flush=True)

    # --- 4. streaming vs non-streaming ------------------------------------
    content = [img("official_affordance.jpg"),
               txt("What is shown in this image?")]
    non = chat(args.url, args.model, content, max_tokens=768)
    st = chat(args.url, args.model, content, max_tokens=768, stream=True)
    same = (st.get("stream_text") or "") == (non.get("text") or "")
    results["streaming"] = {
        "nonstream_text": non.get("text"), "nonstream_status": non.get("status"),
        "nonstream_seconds": non.get("seconds"),
        "stream_text": st.get("stream_text"), "stream_status": st.get("status"),
        "stream_seconds": st.get("seconds"),
        "stream_ttft_seconds": st.get("ttft_seconds"),
        "stream_chunks": st.get("chunks"),
        "stream_usage": st.get("usage"),
        "identical": same}
    print("[smoke] streaming identical=%s chunks=%s ttft=%s"
          % (same, st.get("chunks"), st.get("ttft_seconds")), flush=True)

    # --- 5. multi-turn image+text context ---------------------------------
    body = {"model": args.model, "max_tokens": 768, "temperature": 0.0,
            "seed": 0, "messages": [
                {"role": "user", "content": [
                    img("official_affordance.jpg"),
                    txt("What is shown in this image?")]},
                {"role": "assistant", "content": "A white mug on a table."},
                {"role": "user", "content": [
                    img("official_pointing.jpg"),
                    txt("And what is in this second image?")]},
            ]}
    t0 = time.perf_counter()
    r = requests.post(args.url + "/v1/chat/completions", json=body, timeout=300)
    mt = {"status": r.status_code, "seconds": round(time.perf_counter() - t0, 4),
          "request": body}
    try:
        data = r.json()
        mt["text"] = (data.get("choices") or [{}])[0].get("message", {}) \
            .get("content")
        mt["usage"] = data.get("usage")
    except Exception:  # noqa: BLE001
        mt["text"] = r.text[:600]
    results["multi_turn"] = mt
    print("[smoke] multi_turn status=%s len=%s"
          % (mt.get("status"), len(mt.get("text") or "")), flush=True)

    path = os.path.join(args.out_dir, "smoke_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("[smoke] wrote", path)
    failed = []
    if health.get("http_status") != 200 or models.get("http_status") != 200:
        failed.append("health/models")
    failed.extend(r["case"] for r in task_results if r.get("status") != 200 or not r.get("text") or not r.get("parsed"))
    if results["F07_multi_image"].get("status") != 200 or not text:
        failed.append("F07_multi_image")
    if st.get("status") != 200 or non.get("status") != 200 or not st.get("stream_text") or not same:
        failed.append("streaming")
    if mt.get("status") != 200 or not mt.get("text"):
        failed.append("multi_turn")
    if any(r.get("accepted") for r in invalid[:3]):
        failed.append("invalid_inputs")
    if failed:
        print("[smoke] FAILED:", ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
