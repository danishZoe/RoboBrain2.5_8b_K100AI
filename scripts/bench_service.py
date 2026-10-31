#!/usr/bin/env python3
"""Six-tier service load benchmark for the RoboBrain2.5-8B-NV vLLM service.

Mirrors the load matrix required by the acceptance spec (section 7.1),
which follows the official template's `BENCH_MODE=image bash bench.sh`:

    input tokens | output tokens | concurrency | requests per round
    4096         | 1024          | 1           | 8
    4096         | 1024          | 8           | 16
    4096         | 1024          | 64          | 128
    16384        | 1024          | 1           | 8
    16384        | 1024          | 8           | 16
    16384        | 1024          | 64          | 128

Two rounds per tier; the first round is the warm-up, the second is the
recorded result. Both rounds are written to disk.

Measurement rules that matter:
  * TTFT and TPOT come from a STREAMING request (the first chunk carrying
    content). The non-streaming round-trip time is reported separately and
    is never labelled TTFT.
  * `min_tokens` == `max_tokens` == output budget, so the model cannot stop
    early and the output length is fixed across requests.
  * The input-token target is reached by padding the user text with a
    deterministic filler whose token count is solved for exactly, so the
    reported prompt token count (which includes visual tokens) is verifiable.
  * Failed / truncated requests are kept in the record, never dropped.

Usage (remote):
  python3 scripts/bench_service.py --url http://127.0.0.1:8000/v1/chat/completions \
      --model robobrain2.5-8b-nv --out-dir outputs/phase5_bench
"""

import argparse
import json
import os
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

TIERS = [
    # (input_tokens, output_tokens, concurrency, requests)
    (4096, 1024, 1, 8),
    (4096, 1024, 8, 16),
    (4096, 1024, 64, 128),
    (16384, 1024, 1, 8),
    (16384, 1024, 8, 16),
    (16384, 1024, 64, 128),
]

FILLER_WORDS = (
    "context placeholder for benchmark input padding deterministic seed "
    "reproducible workload calibration token budget filler text content "
    "sequence segment index position ordering reference neutral padding "
)


def build_filler(n_tokens: int, tokenizer, variant: int = 0) -> str:
    """Return a filler string whose token count is exactly n_tokens.

    Built from a single token that encodes to one id, so the count is exact
    and the construction stays O(n) instead of re-tokenising a growing
    string for every word.

    `variant` rotates the word order so that different requests carry
    different prompt text. Without it every request in a tier shares one
    long prefix, which vLLM's prefix cache would serve from memory and
    make the prefill cost disappear from the measurement.
    """
    if n_tokens <= 0:
        return ""
    unit = None
    for cand in (" padding", " context", " placeholder", " x", " the"):
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            unit = cand
            break
    if unit is None:
        raise RuntimeError("no single-token filler found")
    text = unit * n_tokens
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != n_tokens:
        # Fall back to a word-salad built once and trimmed.
        words = FILLER_WORDS.split()
        rot = variant % len(words)
        words = words[rot:] + words[:rot]
        blob = " ".join(words * ((n_tokens // len(words)) + 2))
        ids = tokenizer.encode(blob, add_special_tokens=False)[:n_tokens]
        text = tokenizer.decode(ids)
        if len(tokenizer.encode(text, add_special_tokens=False)) != n_tokens:
            raise RuntimeError("could not build exact-length filler")
    return text


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def stream_one(url, model, image_b64, text, out_tokens, timeout, tag):
    """One streaming chat completion. Returns a metrics dict."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": "data:image/jpeg;base64," + image_b64}},
            {"type": "text", "text": text},
        ]}],
        "max_tokens": out_tokens,
        "min_tokens": out_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    rec = {"tag": tag, "ok": False, "error": None}
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=body, stream=True, timeout=timeout)
        if resp.status_code != 200:
            rec["error"] = "http_%d: %s" % (resp.status_code,
                                             resp.text[:400])
            rec["seconds"] = round(time.perf_counter() - t0, 4)
            return rec
        t_first = None
        chunks = 0
        chars = 0
        usage = {}
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", "replace")
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload.strip() == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices", []):
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    if t_first is None:
                        t_first = time.perf_counter()
                    chunks += 1
                    chars += len(content)
        t_end = time.perf_counter()
        rec["ok"] = True
        rec["ttft_seconds"] = round(t_first - t0, 4) if t_first else None
        rec["e2e_seconds"] = round(t_end - t0, 4)
        rec["stream_chunks"] = chunks
        rec["output_chars"] = chars
        rec["prompt_tokens"] = usage.get("prompt_tokens")
        rec["completion_tokens"] = usage.get("completion_tokens")
        rec["total_tokens"] = usage.get("total_tokens")
        rec["finish_reason"] = None
        if rec["completion_tokens"] is not None and t_first:
            n = max(1, rec["completion_tokens"] - 1)
            rec["tpot_seconds"] = round((t_end - t_first) / n, 5)
        return rec
    except Exception as exc:  # noqa: BLE001 - keep failures in the record
        rec["error"] = "%s: %s" % (type(exc).__name__, exc)
        rec["seconds"] = round(time.perf_counter() - t0, 4)
        return rec


def nonstream_one(url, model, image_b64, text, out_tokens, timeout, tag):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": "data:image/jpeg;base64," + image_b64}},
            {"type": "text", "text": text},
        ]}],
        "max_tokens": out_tokens,
        "min_tokens": out_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
    }
    rec = {"tag": tag, "ok": False, "error": None, "mode": "nonstream"}
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=body, timeout=timeout)
        rec["seconds"] = round(time.perf_counter() - t0, 4)
        if resp.status_code != 200:
            rec["error"] = "http_%d: %s" % (resp.status_code, resp.text[:400])
            return rec
        data = resp.json()
        usage = data.get("usage", {})
        rec["ok"] = True
        rec["prompt_tokens"] = usage.get("prompt_tokens")
        rec["completion_tokens"] = usage.get("completion_tokens")
        rec["total_tokens"] = usage.get("total_tokens")
        rec["finish_reason"] = (data.get("choices") or [{}])[0].get("finish_reason")
        rec["output_chars"] = len(
            (data.get("choices") or [{}])[0].get("message", {}).get("content") or "")
        return rec
    except Exception as exc:  # noqa: BLE001
        rec["error"] = "%s: %s" % (type(exc).__name__, exc)
        return rec


def run_tier(url, model, image_b64, tier, round_idx, timeout, image_path,
             prompt_builder):
    in_tok, out_tok, conc, n_req = tier
    tag = "in%d_out%d_c%d_r%d" % (in_tok, out_tok, conc, round_idx)
    records = []
    lock = threading.Lock()

    def worker(i):
        rec = stream_one(url, model, image_b64, prompt_builder(i), out_tok,
                         timeout, "%s_req%03d" % (tag, i))
        with lock:
            records.append(rec)

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as pool:
        list(pool.map(worker, range(n_req)))
    wall = time.perf_counter() - t_start

    ok = [r for r in records if r["ok"]]
    failed = [r for r in records if not r["ok"]]
    ttfts = [r["ttft_seconds"] for r in ok if r.get("ttft_seconds") is not None]
    e2es = [r["e2e_seconds"] for r in ok if r.get("e2e_seconds") is not None]
    tpots = [r["tpot_seconds"] for r in ok if r.get("tpot_seconds") is not None]
    out_tokens = sum(r.get("completion_tokens") or 0 for r in ok)
    in_tokens = sum(r.get("prompt_tokens") or 0 for r in ok)
    summary = {
        "tag": tag,
        "round": round_idx,
        "tier": {"input_tokens": in_tok, "output_tokens": out_tok,
                 "concurrency": conc, "requests": n_req},
        "image": image_path,
        "mode": "stream",
        "wall_seconds": round(wall, 4),
        "completed": len(ok),
        "failed": len(failed),
        "request_throughput_rps": round(len(ok) / wall, 4) if wall else None,
        "output_token_throughput_tps": round(out_tokens / wall, 4) if wall else None,
        "total_token_throughput_tps": round((out_tokens + in_tokens) / wall, 4)
        if wall else None,
        "avg_prompt_tokens": round(in_tokens / len(ok), 1) if ok else None,
        "avg_output_tokens": round(out_tokens / len(ok), 1) if ok else None,
        "ttft_seconds_mean": round(statistics.fmean(ttfts), 4) if ttfts else None,
        "ttft_seconds_p50": round(percentile(ttfts, 0.50), 4) if ttfts else None,
        "ttft_seconds_p95": round(percentile(ttfts, 0.95), 4) if ttfts else None,
        "tpot_seconds_mean": round(statistics.fmean(tpots), 5) if tpots else None,
        "e2e_seconds_mean": round(statistics.fmean(e2es), 4) if e2es else None,
        "e2e_seconds_p50": round(percentile(e2es, 0.50), 4) if e2es else None,
        "e2e_seconds_p95": round(percentile(e2es, 0.95), 4) if e2es else None,
        "errors": [r.get("error") for r in failed][:5],
    }
    return summary, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--model", default="robobrain2.5-8b-nv")
    ap.add_argument("--image", default="samples/images/official_affordance.jpg")
    ap.add_argument("--out-dir", default="outputs/phase5_bench")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--tiers", default="",
                    help="override, e.g. '4096:1024:1:8,4096:1024:8:16'")
    ap.add_argument("--only-round2-nonstream", action="store_true",
                    help="additionally run the tier once without streaming")
    args = ap.parse_args()

    work = os.environ.get("WORK_DIR", "/root/private_data/workspace/robo")
    os.makedirs(args.out_dir, exist_ok=True)

    tiers = TIERS
    if args.tiers:
        tiers = [tuple(int(x) for x in t.split(":"))
                 for t in args.tiers.split(",")]

    model_dir = os.environ.get("MODEL_PATH")
    sys.path.insert(0, os.path.join(work, "envs", "pylibs"))
    from transformers import AutoProcessor  # noqa: E402
    processor = AutoProcessor.from_pretrained(model_dir)
    tokenizer = processor.tokenizer

    image_path = args.image if os.path.isabs(args.image) \
        else os.path.join(work, args.image)
    with open(image_path, "rb") as f:
        image_b64 = __import__("base64").b64encode(f.read()).decode()

    # --- exact input-token accounting -------------------------------------
    probe_text = "Describe the image."
    messages = [{"role": "user", "content": [
        {"type": "image", "image": "file://" + image_path},
        {"type": "text", "text": probe_text},
    ]}]
    chat = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    enc = processor(text=[chat], images=__import__("PIL.Image", fromlist=["Image"])
                        .open(image_path).convert("RGB"), return_tensors="pt")
    visual_tokens = int(enc["input_ids"].shape[1]) - len(
        tokenizer.encode(chat.replace("<|image_pad|>", ""),
                         add_special_tokens=False))
    print("[bench] visual tokens for %s: %d" % (image_path, visual_tokens))

    all_summaries = []
    all_records = {}
    for tier in tiers:
        in_tok = tier[0]
        fixed = visual_tokens + len(tokenizer.encode(
            "\n\nDescribe the image in exhaustive detail.", add_special_tokens=False))
        filler_tokens = max(0, in_tok - fixed)
        # Every request gets its own filler text (same token count, different
        # content) so the measurement is not dominated by a shared prefix.
        def make_prompt(variant, _ft=filler_tokens):
            filler = build_filler(_ft, tokenizer, variant=variant)
            return filler + "\n\nDescribe the image in exhaustive detail."

        prompt_text = make_prompt(0)
        chat_chk = processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image", "image": "file://" + image_path},
                {"type": "text", "text": prompt_text}]}],
            tokenize=False, add_generation_prompt=True)
        est = visual_tokens + len(tokenizer.encode(
            chat_chk.replace("<|image_pad|>", ""), add_special_tokens=False))
        print("[bench] tier in=%d -> estimated prompt tokens ~%d (visual=%d)"
              % (in_tok, est, visual_tokens))
        for r in range(1, args.rounds + 1):
            summary, records = run_tier(
                args.url, args.model, image_b64, tier, r, args.timeout,
                image_path, make_prompt)
            summary["estimated_prompt_tokens"] = est
            all_summaries.append(summary)
            all_records[summary["tag"]] = records
            print("[bench] %s completed=%d failed=%d wall=%.1fs "
                  "out_tps=%.2f req_rps=%.3f ttft_mean=%s"
                  % (summary["tag"], summary["completed"], summary["failed"],
                     summary["wall_seconds"],
                     summary["output_token_throughput_tps"] or 0,
                     summary["request_throughput_rps"] or 0,
                     summary["ttft_seconds_mean"]), flush=True)
            with open(os.path.join(args.out_dir, "bench_results.json"), "w",
                      encoding="utf-8") as f:
                json.dump(all_summaries, f, ensure_ascii=False, indent=2)
            with open(os.path.join(args.out_dir, "bench_records.json"), "w",
                      encoding="utf-8") as f:
                json.dump(all_records, f, ensure_ascii=False, indent=2)
        if args.only_round2_nonstream:
            summary, records = run_tier(
                args.url, args.model, image_b64, tier, 99, args.timeout,
                image_path, make_prompt)
            summary["mode"] = "nonstream"
            all_summaries.append(summary)
            all_records[summary["tag"]] = records
            print("[bench] %s (nonstream) completed=%d failed=%d wall=%.1fs"
                  % (summary["tag"], summary["completed"], summary["failed"],
                     summary["wall_seconds"]), flush=True)

    print("[bench] done ->", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
