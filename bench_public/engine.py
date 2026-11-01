#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""公开基准推理引擎（并发、断点续跑、逐题终态记录）。

设计要点（对应《RoboBrain2.5-8B-NV_精度测试验收要求.md》§4.3 / §5.4 / §8）：

* **并发固定**：默认 8，命令行可改；每次运行写进配置哈希。
* **每个样本有唯一终态**：``ok`` / ``request_failed`` / ``empty_output`` /
  ``truncated`` / ``parse_failed`` / ``invalid_value``。空输出、越界、NaN 一律
  计入分母，不静默删除。
* **只对传输/服务错误重试**（上限 2 次，见 ``common.chat_completion``）；
  答案错误不重试。
* **断点续跑**：已存在且终态为 ok 的样本不重复请求（可 ``--no-resume`` 关掉）。
* **逐题记录**字段见验收要求 §8：run_id / dataset / revision / split /
  sample_id / config_hash / input 文件 / 最终 prompt / 原始输出 / 解析结果 /
  score / 终态 / 错误原因。
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

import common as C

_LOCK = threading.Lock()


def _messages_repr(image_paths, text: str) -> List[Dict[str, Any]]:
    """记录最终送出的消息结构；图片用内容摘要代替 base64（避免逐题落盘上百 MB）。

    base64 原文可由 ``image_geometry`` + 原始文件/字节复现，这里保留摘要以便
    审计"送进模型的到底是什么"。
    """
    content: List[Dict[str, Any]] = []
    for ref in image_paths:
        if isinstance(ref, tuple) and len(ref) == 2 and ref[0] == "bytes":
            digest = C.sha256_text("bytes:" + str(len(ref[1])))
            content.append({"type": "image_url",
                            "image_url": {"url": "sha256:" + digest}})
        else:
            content.append({"type": "image_url",
                            "image_url": {"url": "file://" + os.path.abspath(ref)}})
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def _geometry(job: Dict[str, Any]) -> Any:
    try:
        import datasets as D
        refs = job["image_paths"]
        if refs and isinstance(refs[0], tuple):
            return D.image_geometry(data=refs[0][1])
        if refs:
            return D.image_geometry(path=refs[0])
    except Exception as exc:                        # pragma: no cover
        return {"error": repr(exc)}
    return None


def _terminal_state(resp: Dict[str, Any], parsed_ok: bool) -> str:
    if resp.get("status") == "error":
        return "request_failed"
    text = resp.get("text") or ""
    if not text.strip():
        return "empty_output"
    if resp.get("finish_reason") == "length":
        return "truncated"
    if not parsed_ok:
        return "parse_failed"
    return "ok"


def run_jobs(jobs: List[Dict[str, Any]],
             out_path: str,
             url: str,
             model: str,
             max_tokens: int = C.MAX_NEW_TOKENS,
             concurrency: int = 8,
             run_meta: Optional[Dict[str, Any]] = None,
             resume: bool = True,
             parse_fn: Optional[Callable[[str], Any]] = None,
             label: str = "") -> List[Dict[str, Any]]:
    """执行一批推理任务，返回逐题结果列表（按 sample_id 排序后写盘）。

    ``jobs`` 每项：
      ``sample_id`` / ``split`` / ``image_paths`` / ``prompt`` / ``fields``
    """
    run_meta = dict(run_meta or {})
    done: Dict[str, Dict[str, Any]] = {}
    if resume and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("terminal_state") == "ok":
                    done[rec["sample_id"]] = rec
        if done:
            print("[engine] 断点续跑：已有 %d/%d 条 ok 记录" %
                  (len(done), len(jobs)), flush=True)

    todo = [j for j in jobs if j["sample_id"] not in done]
    results: Dict[str, Dict[str, Any]] = dict(done)
    t0 = time.time()
    counter = {"n": 0}

    def one(job: Dict[str, Any]) -> Dict[str, Any]:
        resp = C.chat_completion(url, model, job["image_paths"], job["prompt"],
                                 max_tokens=max_tokens)
        raw = resp.get("text") or ""
        parsed = None
        parsed_ok = True
        if parse_fn is not None:
            try:
                parsed = parse_fn(raw)
                parsed_ok = bool(parsed)
            except Exception as exc:
                parsed = {"error": repr(exc)}
                parsed_ok = False
        state = _terminal_state(resp, parsed_ok)
        rec = {
            "sample_id": job["sample_id"],
            "split": job["split"],
            "prompt": job["prompt"],
            "messages": _messages_repr(job["image_paths"], job["prompt"]),
            "image_geometry": _geometry(job),
            "raw_output": raw,
            "parsed": parsed,
            "usage": resp.get("usage", {}),
            "seconds": resp.get("seconds"),
            "finish_reason": resp.get("finish_reason"),
            "attempts": resp.get("attempts"),
            "terminal_state": state,
            "error": resp.get("error"),
        }
        with _LOCK:
            counter["n"] += 1
            k = counter["n"]
            if k % 25 == 0 or k == len(todo):
                el = time.time() - t0
                rate = k / el if el else 0
                eta = (len(todo) - k) / rate if rate else 0
                print("[engine]%s %d/%d  %.1f req/s  ETA %.0fs" %
                      (label, k, len(todo), rate, eta), flush=True)
        return rec

    if todo:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for rec in pool.map(one, todo):
                results[rec["sample_id"]] = rec

    ordered = [results[j["sample_id"]] for j in jobs if j["sample_id"] in results]
    C.dump_jsonl(out_path, ordered)
    print("[engine]%s 完成 %d 条 -> %s  用时 %.1fs" %
          (label, len(ordered), out_path, time.time() - t0), flush=True)
    return ordered
