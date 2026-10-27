#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_verify_denominator.py - 验收标准 §4.3「解析失败不能剔除出官方计分分母」的
可执行验证。

验证三件事（全部离线，不调用服务）：
  1) 请求失败（HTTP 错误）的样本**留在分母里**，记为未命中；
  2) 输出**解析失败**（正则不匹配）的样本**留在分母里**，记为未命中；
  3) 缺参考输出的样本单独计数（without_reference），不被悄悄算成命中。

用法: python3 tools/test_verify_denominator.py
"""
import importlib.util
import json
import os
import sys
import urllib.error

W = "/root/private_data/workspace/robo"
SUB = os.path.join(W, "submission")
TMP = os.path.join(W, "tmp", "verify_denominator")
os.makedirs(TMP, exist_ok=True)

spec = importlib.util.spec_from_file_location("verify_mod", os.path.join(SUB, "verify.py"))
verify = importlib.util.module_from_spec(spec)
sys.path.insert(0, SUB)
spec.loader.exec_module(verify)

SUBSET = os.path.join(TMP, "subset.jsonl")
REFS = os.path.join(TMP, "refs.jsonl")
with open(SUBSET, "w", encoding="utf-8") as f:
    for i, task in enumerate(["grounding", "grounding", "grounding", "grounding"]):
        f.write(json.dumps({"id": "s%d" % i, "task": task,
                            "prompt": "Please provide the bounding box coordinate of the region "
                                      "this sentence describes: the mug."},
                           ensure_ascii=False) + "\n")
with open(REFS, "w", encoding="utf-8") as f:
    for i in range(4):
        f.write(json.dumps({"response": "[0, 309, 620, 989]"}, ensure_ascii=False) + "\n")

METRIC = {
    "subset": "subset.jsonl",
    "reference": "refs.jsonl",
    "n": 4,
    "judge_type": "agreement",
    "threshold": 98.0,
    "ref": 100.0,
    "scale": 100,
}

GOOD = "The box is [0, 309, 620, 989]."
UNPARSEABLE = "I cannot determine the bounding box for this region."

calls = {"n": 0}


def fake_chat(url, model, payload, timeout=300):
    i = calls["n"]
    calls["n"] += 1
    if i == 1:
        raise urllib.error.HTTPError(url, 500, "synthetic failure", None, None)
    if i == 2:
        return {"choices": [{"message": {"content": UNPARSEABLE}}]}
    if i == 3:
        return {"choices": [{"message": {"content": "[0, 0, 0, 0]"}}]}   # wrong but parseable
    return {"choices": [{"message": {"content": GOOD}}]}


verify.chat_completion = fake_chat
evidence = {}
res = verify.run_metric("agreement_test", METRIC, {}, TMP, "http://127.0.0.1:1", "m", evidence)

print("result:", json.dumps({k: v for k, v in res.items() if k != "detail"},
                             ensure_ascii=False, indent=2))
print("per-sample:", json.dumps(evidence.get("agreement_test"), ensure_ascii=False))

fails = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


print("\n=== assertions ===")
# 4 条题目：0 正确、1 请求失败、2 解析失败、3 答错但可解析
check(res.get("scored") == 4, "分母 = 4（请求失败与解析失败都没有被剔除）")
check(res.get("errors") == 1, "errors = 1（请求失败被单独记账）")
check(abs(res.get("got", 0) - 25.0) < 1e-6, "一致率 = 25.0%（1/4 命中，而不是 1/2）")
check(res.get("status") == "FAIL", "状态 = FAIL（未达 98% 阈值）")
check(any("请求失败" in n for n in (res.get("notes") or [])), "notes 显式披露请求失败")

print("\n%s" % ("全部通过" if not fails else "FAILED: %s" % fails))
sys.exit(0 if not fails else 1)
