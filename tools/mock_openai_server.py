#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal OpenAI-compatible mock server for the verify.py mechanism rehearsal.

**This is not a model.** It exists only to exercise the *mechanics* of
``verify.py``'s "baseline present" path end-to-end (subset -> per-sample
request -> judge -> verify_evidence.json) **without touching the production
vLLM service**, so that a running official benchmark is never polluted.

Behaviour: it answers every chat-completion request with the *reference*
answer of the sample whose prompt appears in the request, so the agreement
judge has something well-formed to compare against.

Usage:
    python3 mock_openai_server.py <subset.jsonl> [port]
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SUBSET = sys.argv[1]
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 39999

BY_PROMPT = {}
if os.path.exists(SUBSET):
    with open(SUBSET, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            BY_PROMPT[rec.get("prompt", "")] = rec


def _flatten(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    out.append(str(part.get("text", "")))
                elif part.get("type") in ("image_url", "image"):
                    out.append("<image>")
            elif isinstance(part, str):
                out.append(part)
        return "\n".join(out)
    return ""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # keep the rehearsal log clean
        pass

    def _send(self, code: int, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/health"):
            self._send(200, b"ok", "text/plain")
        else:
            self._send(404, b"{}")

    def do_POST(self):  # noqa: N802
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            body = {}

        text = ""
        for msg in body.get("messages", []):
            text += _flatten(msg.get("content")) + "\n"

        ref = None
        for prompt, rec in BY_PROMPT.items():
            if prompt and prompt in text:
                ref = rec.get("reference")
                break
        if ref is None:
            ref = ""
        if not isinstance(ref, str):
            ref = json.dumps(ref, ensure_ascii=False)

        payload = {
            "id": "mock-chatcmpl",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "mock"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": ref},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                      "total_tokens": 2},
        }
        self._send(200, json.dumps(payload, ensure_ascii=False).encode("utf-8"))


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print("[mock] serving %d subset prompts on 127.0.0.1:%d" %
          (len(BY_PROMPT), PORT), flush=True)
    srv.serve_forever()
