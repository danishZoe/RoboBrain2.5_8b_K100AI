#!/usr/bin/env python3
"""Accuracy verification against the official GPU baseline.

Official contract (PDF p.5 / p.2, acceptance spec section 6)
-------------------------------------------------------------
* The baseline file ``baselines/<MODEL_ID>.json`` is issued by the platform and
  must not be modified. It freezes the decoding config (greedy, thinking
  switch, max_tokens, dataset and question count) and the official GPU scores.
* Every indicator keeps its own unit; the degradation must not exceed 2%
  (percentage-style indicators: 2 pt; OCRBench thousand-scale: 20 points).
  For embodied / VLA models the criterion is the agreement rate of the parsed
  action or coordinates against the GPU reference output, >= 98%.
* If the baseline file or the judge is missing, the accuracy item stays
  ``BLOCKED``. It must NOT be replaced by a self-made demo set.

What this script does
---------------------
1. ``--selfcheck``            : check the scoring logic only, no service needed.
2. baseline file missing     : write ``verify_result.json`` with
   ``status=BLOCKED`` + the reason, plus an **internal** cross-backend
   consistency figure (same card, transformers direct run vs vLLM serving).
   That figure is explicitly labelled ``internal`` and is NOT the official
   accuracy verdict.
3. baseline file present     : **actually run the frozen configuration**
   against the served model, score every metric with the official-style
   judge, and decide PASS/FAIL per indicator.

   IMPORTANT: this script never reports PASS without having run the
   evaluation. If the subset file, the reference outputs or a judge cannot be
   resolved, the affected metric is reported as ``BLOCKED`` (with the precise
   reason) and the overall verdict is not PASS.

Usage:
    python verify.py --selfcheck          # check the scoring logic only
    source ../env.sh && python verify.py   # run against the service
"""

import argparse
import importlib.util
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "scripts"))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

from answer_adapter import parse_answer  # noqa: E402

# Official parsing regexes (third_party/RoboBrain2.5/inference.py:110-124).
RE_TRAJ = re.compile(r"(\d+),\s*(\d+),\s*([+-]?\d+\.\d+)")
RE_POINT = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
RE_BOX = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]")

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

# Fields that may carry the per-sample reference answer.
REF_KEYS = ("reference", "answer", "ref", "gt", "ground_truth", "label",
             "output", "target", "response")
# Fields that may carry the question text.
Q_KEYS = ("question", "text", "prompt", "instruction", "query", "input")
# Fields that may carry the image path.
IMG_KEYS = ("image", "image_path", "img", "img_path", "picture")

TOL_POINTS = {100: 2.0, 1000: 20.0}   # official wording: 百分制 2 pt / 千分制 20 pt
DEFAULT_AGREEMENT_THRESHOLD = 98.0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def baseline_path(model_id: str) -> str:
    safe = model_id.replace("/", "__")
    return os.path.join(HERE, "baselines", safe + ".json")


def score_degradation(got, ref, scale):
    """Relative degradation in percent, matching the official wording."""
    if ref is None:
        return None
    if scale == "ratio":
        # agreement rate style: lower is better for a pass rate
        return (ref - got) / ref * 100.0 if ref else None
    if ref == 0:
        return 0.0 if got == 0 else float("inf")
    return (ref - got) / ref * 100.0


def score_degradation_points(got, ref, scale):
    """Absolute degradation on the metric scale (百分制 2 pt / 千分制 20 pt)."""
    if ref is None or got is None:
        return None
    return float(ref) - float(got)


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pick_field(rec, keys):
    for k in keys:
        if k in rec and rec[k] not in (None, ""):
            return rec[k]
    return None


def material_candidate_roots(base_dir):
    """Roots that a relative material path may be resolved against.

    The official baseline file is placed in ``baselines/`` while the dataset,
    the reference outputs and the judge are normally shipped at the submission
    root.  Pinning a relative path to a single root made the accuracy item
    report a *false* BLOCKED ("subset does not exist") even when the official
    materials were present, so every relative path is resolved against all
    plausible roots in a fixed order.
    """
    roots = []
    for r in (HERE, base_dir,
              os.path.dirname(os.path.abspath(base_dir)),
              os.path.dirname(HERE)):
        if r and r not in roots:
            roots.append(r)
    return roots


def resolve_material_path(base_dir, rel, suffixes=(), extra_roots=()):
    """Resolve a (possibly relative) material path against several roots.

    Returns ``(resolved_path_or_None, tried_candidates)``.
    """
    if not rel:
        return None, []
    if os.path.isabs(rel):
        cands = [rel]
    else:
        roots = list(extra_roots) + material_candidate_roots(base_dir)
        cands, seen = [], set()
        for root in roots:
            c = os.path.normpath(os.path.join(root, rel))
            if c not in seen:
                seen.add(c)
                cands.append(c)
    tried = []
    for cand in cands:
        for suf in ("",) + tuple(suffixes):
            p = cand + suf
            tried.append(p)
            if os.path.exists(p):
                return p, tried
    return None, tried


def load_external_judge(base_dir, name):
    """Load an optional judge module shipped with the official materials."""
    if not name:
        return None
    path, _tried = resolve_material_path(base_dir, name, suffixes=(".py",))
    if path is None:
        return None
    spec = importlib.util.spec_from_file_location("official_judge", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "judge", None)
    return fn


def extract_boxed(text):
    m = re.findall(r"\\boxed\{([^}]*)\}", text or "")
    if m:
        return m[-1].strip()
    m = re.findall(r"####\s*([^\n]+)", text or "")
    if m:
        return m[-1].strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text or "")
    return nums[-1] if nums else None


def extract_choice(text):
    m = re.search(r"\b([A-D])\b", (text or "").strip())
    if m:
        return m.group(1)
    m = re.search(r"answer\D{0,5}([A-D])\b", text or "", re.IGNORECASE)
    return m.group(1).upper() if m else None


def _close(a, b, tol):
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


def parsed_equal(task, got, ref, tol=0.0):
    """Compare two parsed answers under the official coordinate contract."""
    if got is None or ref is None:
        return False
    if task == "grounding":
        g, r = got.get("bbox"), ref.get("bbox")
        if not g or not r or len(g) != len(r):
            return False
        return all(_close(a, b, tol) for a, b in zip(g, r))
    if task == "trajectory":
        g, r = got.get("points") or [], ref.get("points") or []
        if len(g) != len(r) or not g:
            return False
        for pg, pr in zip(g, r):
            if len(pg) != len(pr):
                return False
            if not all(_close(a, b, tol) for a, b in zip(pg, pr)):
                return False
        return True
    if task in ("pointing", "affordance"):
        g, r = got.get("points") or [], ref.get("points") or []
        if len(g) != len(r) or not g:
            return False
        for pg, pr in zip(g, r):
            if not _close(pg.get("x"), pr.get("x"), tol) or \
               not _close(pg.get("y"), pr.get("y"), tol):
                return False
        return True
    return False


def judge_sample(kind, task, response_text, reference, judge_fn, tol):
    """Return True when the response is judged correct for this sample."""
    if judge_fn is not None:
        out = judge_fn(response_text, reference)
        return bool(out)
    if kind == "agreement":
        return parsed_equal(task, parse_answer(response_text, task),
                            parse_answer(str(reference), task), tol)
    if kind in ("numeric", "gsm8k", "exact_match"):
        g, r = extract_boxed(response_text), extract_boxed(str(reference))
        if g is None or r is None:
            return False
        try:
            return abs(float(g) - float(r)) < 1e-6
        except ValueError:
            return str(g).strip() == str(r).strip()
    if kind in ("choice", "mmlu"):
        g, r = extract_choice(response_text), extract_choice(str(reference))
        return g is not None and r is not None and g == r
    # default: string equality after whitespace normalisation
    return (response_text or "").strip() == str(reference).strip()


def infer_kind(metric, metric_name):
    for key in ("judge_type", "metric", "type", "scoring"):
        v = metric.get(key)
        if v:
            v = str(v).lower()
            if v in ("agreement", "coordinate_agreement", "exact"):
                return "agreement"
            if v in ("numeric", "gsm8k", "accuracy", "exact_match"):
                return "numeric"
            if v in ("choice", "mmlu", "multiple_choice"):
                return "choice"
    name = metric_name.lower()
    if "agreement" in name:
        return "agreement"
    if "gsm8k" in name or "numeric" in name:
        return "numeric"
    if "mmlu" in name or "choice" in name:
        return "choice"
    return None


# --------------------------------------------------------------------------- #
# service
# --------------------------------------------------------------------------- #
def chat_completion(url, model, payload, timeout=300):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_payload(record, metric, baseline, served_name, image_data_url):
    text = pick_field(record, Q_KEYS) or ""
    task = record.get("task") or metric.get("task") or "general"
    prompt = TASK_TEMPLATES.get(task, TASK_TEMPLATES["general"]).format(text=text)
    content = []
    if image_data_url:
        content.append({"type": "image_url",
                        "image_url": {"url": image_data_url}})
    content.append({"type": "text", "text": prompt})
    payload = {
        "model": served_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": metric.get("temperature",
                                   baseline.get("extra_body", {}).get("temperature", 0)),
        "max_tokens": int(metric.get("max_tokens", 768)),
        "stream": False,
    }
    extra = dict(baseline.get("extra_body", {}) or {})
    extra.pop("temperature", None)
    payload.update(extra)
    return payload, task, prompt


def load_image_data_url(record, base_dir, extra_roots=()):
    img = pick_field(record, IMG_KEYS)
    if not img:
        return None
    path, _tried = resolve_material_path(base_dir, img, extra_roots=extra_roots)
    if path is None:
        return None
    try:
        from image_canonical import canonicalize_to_data_url
        return canonicalize_to_data_url(path)
    except Exception as exc:  # pragma: no cover - defensive
        print("[verify] image canonicalization failed for %s: %s" % (path, exc))
        return None


# --------------------------------------------------------------------------- #
# selfcheck
# --------------------------------------------------------------------------- #
def selfcheck() -> int:
    """Verify the scoring logic on synthetic data; must not touch the service."""
    ok = True
    cases = [
        ("grounding", "The box is [0, 309, 620, 989].", {"bbox": [0, 309, 620, 989]}),
        ("grounding", "box <bbox>0 298 622 998</bbox>",
         {"bbox": [0, 298, 622, 998]}),
        ("pointing", "the answer is [(901, 600)]",
         {"points": [{"x": 901, "y": 600}]}),
        ("trajectory", "[(294, 525, 0.577), (405, 350, 0.607)]",
         {"points": [[294, 525, 0.577], [405, 350, 0.607]]}),
    ]
    for task, text, expect in cases:
        got = parse_answer(text, task)
        if got != expect:
            ok = False
            print("[selfcheck] FAIL %-11s %r -> %r (expect %r)"
                  % (task, text, got, expect))
        else:
            print("[selfcheck] ok   %-11s %r -> %r" % (task, text, got))
    deg = score_degradation(98.0, 100.0, "ratio")
    print("[selfcheck] degradation 98 vs 100 = %.2f%%" % deg)
    if deg is None or abs(deg - 2.0) > 1e-6:
        ok = False
        print("[selfcheck] FAIL degradation formula")

    # point-degradation (the interpretation used by the real path)
    if abs(score_degradation_points(98.0, 100.0, 100) - 2.0) > 1e-6:
        ok = False
        print("[selfcheck] FAIL point degradation formula")
    else:
        print("[selfcheck] ok   point degradation 100->98 = 2.00 pt")

    # agreement judge (T02 embodied contract)
    ref = "the box is [10, 20, 30, 40]"
    if not judge_sample("agreement", "grounding", "box is [10, 20, 30, 40]", ref, None, 0):
        ok = False
        print("[selfcheck] FAIL agreement judge (identical box)")
    elif judge_sample("agreement", "grounding", "box is [11, 20, 30, 40]", ref, None, 0):
        ok = False
        print("[selfcheck] FAIL agreement judge (different box must not agree)")
    else:
        print("[selfcheck] ok   agreement judge (exact)")

    # numeric + choice judges
    if not judge_sample("numeric", "general", "so the answer is \\boxed{42}", "42", None, 0):
        ok = False
        print("[selfcheck] FAIL numeric judge")
    elif not judge_sample("choice", "general", "The answer is B.", "B", None, 0):
        ok = False
        print("[selfcheck] FAIL choice judge")
    else:
        print("[selfcheck] ok   numeric + choice judges")

    print("[selfcheck] %s" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# internal metric (never a substitute for the official verdict)
# --------------------------------------------------------------------------- #
def run_internal_consistency(work_dir, base_out):
    """Internal metric only: transformers direct run vs vLLM serving."""
    base_path = os.path.join(work_dir, "outputs", "phase3_baseline",
                             "baseline_raw.json")
    cand_path = os.path.join(work_dir, "outputs", "phase5_probe_canonical",
                              "probe_result.json")
    if not (os.path.exists(base_path) and os.path.exists(cand_path)):
        return None
    base = {}
    for rec in json.load(open(base_path, encoding="utf-8"))["records"]:
        base.setdefault(rec["sample_id"], rec)
    out = {}
    for rec in json.load(open(cand_path, encoding="utf-8"))["records"]:
        b = base.get(rec["sample_id"])
        if not b:
            continue
        pb = parse_answer(b["answer"], rec["task"])
        pc = parse_answer(rec["text"], rec["task"])
        out[rec["sample_id"]] = {"baseline": pb, "serving": pc,
                                 "identical": pb == pc}
    return out


# --------------------------------------------------------------------------- #
# real evaluation (baseline present)
# --------------------------------------------------------------------------- #
def run_metric(metric_name, metric, baseline, base_dir, url, served_name,
               evidence):
    """Run one official metric. Returns a result dict; never fakes a PASS."""
    res = {"metric": metric_name}
    kind = infer_kind(metric, metric_name)
    res["judge_type"] = kind or "unresolved"

    ref = metric.get("ref")
    scale = metric.get("scale", 100)
    tol_points = metric.get("tolerance",
                            TOL_POINTS.get(int(scale) if str(scale).isdigit() else 100, 2.0))
    res["tolerance_points"] = tol_points
    res["official_ref"] = ref

    subset_rel = metric.get("subset")
    if not subset_rel:
        res["status"] = "BLOCKED"
        res["reason"] = "基线文件未给出 subset（评测子集路径），无法定位题目。"
        return res
    subset_path, tried = resolve_material_path(base_dir, subset_rel)
    if subset_path is None:
        res["status"] = "BLOCKED"
        res["reason"] = ("评测子集 %s 不存在（随官方材料下发）。已尝试: %s"
                         % (subset_rel, " | ".join(tried)))
        res["tried_paths"] = tried
        return res

    try:
        records = load_jsonl(subset_path)
    except Exception as exc:
        res["status"] = "BLOCKED"
        res["reason"] = "评测子集 %s 解析失败: %s" % (subset_rel, exc)
        return res

    n = int(metric.get("n") or len(records))
    records = records[:n]

    # reference outputs
    references = None
    ref_rel = metric.get("reference") or metric.get("ref_file")
    if ref_rel:
        ref_path, tried_ref = resolve_material_path(base_dir, ref_rel)
        if ref_path is None:
            res["status"] = "BLOCKED"
            res["reason"] = ("参考输出文件 %s 不存在（随官方材料下发）。已尝试: %s"
                             % (ref_rel, " | ".join(tried_ref)))
            res["tried_paths"] = tried_ref
            return res
        references = load_jsonl(ref_path)

    judge_fn = load_external_judge(base_dir, metric.get("judge"))
    tol_coord = float(metric.get("tolerance_px", 0) or 0)

    if kind is None and judge_fn is None:
        res["status"] = "BLOCKED"
        res["reason"] = ("无法确定该指标的判分方式（基线文件未给 judge/judge_type，"
                         "且指标名无法映射到内置判分器）。请随材料提供判分器。")
        return res

    hits = 0
    scored = 0
    unparsed = 0
    failed_requests = 0
    detail = []
    for idx, rec in enumerate(records):
        reference = None
        if references is not None:
            reference = (references[idx] if isinstance(references[idx], str)
                         else pick_field(references[idx], REF_KEYS)
                         or json.dumps(references[idx], ensure_ascii=False))
        else:
            reference = pick_field(rec, REF_KEYS)
        if reference is None:
            unparsed += 1
            continue
        img_url = load_image_data_url(rec, base_dir,
                                      extra_roots=(os.path.dirname(subset_path),))
        payload, task, prompt = build_payload(rec, metric, baseline,
                                              served_name, img_url)
        try:
            resp = chat_completion(url, served_name, payload)
            text = resp["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:300]
            detail.append({"index": idx, "error": "HTTP %s: %s" % (exc.code, body)})
            # 请求失败**计入分母**：验收标准 §4.3 要求「解析失败不能剔除出官方计分分母」，
            # 同理「请求失败」也不能靠丢样本把一致率做高 —— 记为未命中。
            failed_requests += 1
            scored += 1
            continue
        except Exception as exc:
            detail.append({"index": idx, "error": repr(exc)})
            failed_requests += 1
            scored += 1
            continue
        good = judge_sample(kind, task, text, reference, judge_fn, tol_coord)
        scored += 1
        hits += 1 if good else 0
        detail.append({"index": idx, "task": task, "correct": bool(good),
                       "response": text[:400]})
    evidence[metric_name] = detail

    if scored == 0:
        res["status"] = "BLOCKED"
        res["reason"] = ("题目已加载，但没有任何一条能取到参考输出"
                          "（参考输出缺失或请求全部失败）。")
        return res

    res["scored"] = scored
    res["requested"] = n
    res["without_reference"] = unparsed
    res["errors"] = failed_requests
    # 分母完整性说明：官方分母是题目数。请求失败已计入分母（记为未命中）；
    # 「没有参考输出」的题目无法判分，单独计数并在结果里显式披露，
    # 避免出现「分母被悄悄缩小」的读数。
    notes = []
    if failed_requests:
        notes.append("%d 条请求失败，已计入分母并记为未命中" % failed_requests)
    if unparsed:
        notes.append("%d 条缺少参考输出，未参与判分（材料缺口，需随基线文件补齐）" % unparsed)
    if notes:
        res["notes"] = notes
    res["denominator_basis"] = ("有参考输出的题目数；请求失败计入分母并记为未命中，"
                                "缺参考输出的题目单独计数不参与判分")

    if kind == "agreement" or metric.get("type") == "agreement":
        got = hits / scored * 100.0
        threshold = float(metric.get("threshold", DEFAULT_AGREEMENT_THRESHOLD))
        res["got"] = round(got, 4)
        res["threshold"] = threshold
        res["degradation_points"] = score_degradation_points(got, threshold, 100)
        res["degradation_percent"] = score_degradation(got, threshold, "ratio")
        res["pass"] = got >= threshold
        res["status"] = "PASS" if res["pass"] else "FAIL"
        return res

    got = hits / scored * 100.0
    res["got"] = round(got, 4)
    if ref is None:
        res["status"] = "BLOCKED"
        res["reason"] = "基线文件未给出该指标的官方 GPU 分数（ref），无法判定劣化。"
        return res
    res["degradation_points"] = score_degradation_points(got, ref, scale)
    res["degradation_percent"] = score_degradation(got, ref, scale)
    res["pass"] = (res["degradation_points"] is not None
                   and res["degradation_points"] <= tol_points)
    res["status"] = "PASS" if res["pass"] else "FAIL"
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "verify_result.json"))
    ap.add_argument("--evidence", default=os.path.join(HERE, "verify_evidence.json"))
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()

    model_id = os.environ.get("MODEL_ID", "BAAI/RoboBrain2.5-8B-NV")
    card = os.environ.get("CARD", "K100-AI")
    framework = os.environ.get("FRAMEWORK", "vllm")
    precision = os.environ.get("PRECISION", "bf16")
    work_dir = os.environ.get("WORK_DIR", os.path.join(HERE, ".."))
    port = os.environ.get("PORT", "30000")
    served_name = os.environ.get("SERVED_NAME", "robobrain2.5-8b-nv")

    result = {
        "MODEL_ID": model_id,
        "CARD": card,
        "FRAMEWORK": framework,
        "PRECISION": precision,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    bpath = baseline_path(model_id)
    if not os.path.exists(bpath):
        result["status"] = "BLOCKED"
        result["reason"] = (
            "官方基线文件 %s 未下发；官方评测数据集与判分器同样未下发。"
            "按验收标准第 2 节，精度项必须标记为 BLOCKED，"
            "不得用自行选择的 demo 或自建 GPU 结果替代官方 98%% 验收证据。"
            % os.path.relpath(bpath, HERE))
        internal = run_internal_consistency(work_dir, None)
        if internal is not None:
            same = sum(1 for v in internal.values() if v["identical"])
            result["internal_metric"] = {
                "name": "同卡不同后端逐坐标一致性（非官方指标）",
                "note": "transformers 直跑基线 vs vLLM serving；"
                        "不能替代官方 GPU 参考输出一致率",
                "identical": same,
                "total": len(internal),
                "detail": internal,
            }
        result["passed"] = False
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("[verify] BLOCKED: %s" % result["reason"])
        print("[verify] wrote %s" % args.out)
        return 2

    # --- official baseline present: run the frozen config ------------------
    baseline = json.load(open(bpath, encoding="utf-8"))
    base_dir = os.path.dirname(os.path.abspath(bpath))
    metrics = baseline.get("metrics") or {}
    if not metrics:
        result["status"] = "BLOCKED"
        result["reason"] = ("基线文件存在但没有 metrics 段，"
                             "无法确定要跑哪些指标。")
        result["passed"] = False
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("[verify] BLOCKED: %s" % result["reason"])
        return 2

    url = "http://127.0.0.1:%s/v1/chat/completions" % port
    evidence = {}
    detail = {}
    for name, metric in metrics.items():
        print("[verify] running metric %s ..." % name)
        detail[name] = run_metric(name, metric, baseline, base_dir, url,
                                   served_name, evidence)

    blocked = [k for k, v in detail.items() if v.get("status") == "BLOCKED"]
    failed = [k for k, v in detail.items() if v.get("status") == "FAIL"]
    passed = [k for k, v in detail.items() if v.get("status") == "PASS"]

    result["detail"] = detail
    result["tolerance_percent"] = baseline.get("tolerance_percent", 2.0)
    result["summary"] = {"pass": passed, "fail": failed, "blocked": blocked}
    result["passed"] = bool(passed) and not failed and not blocked
    if blocked:
        result["status"] = "BLOCKED"
        result["reason"] = ("以下指标缺少官方材料（子集 / 参考输出 / 判分器），"
                            "按验收标准不得自建替代：%s" % ", ".join(blocked))
    elif failed:
        result["status"] = "FAIL"
        result["reason"] = "以下指标劣化超过容差：%s" % ", ".join(failed)
    else:
        result["status"] = "PASS"

    internal = run_internal_consistency(work_dir, None)
    if internal is not None:
        same = sum(1 for v in internal.values() if v["identical"])
        result["internal_metric"] = {
            "name": "同卡不同后端逐坐标一致性（非官方指标）",
            "note": "transformers 直跑基线 vs vLLM serving；"
                    "不能替代官方 GPU 参考输出一致率",
            "identical": same,
            "total": len(internal),
        }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(args.evidence, "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)
    print("[verify] %s: %s" % (result["status"],
                               result.get("reason", "all metrics within tolerance")))
    print("[verify] wrote %s (+ %s)" % (args.out, args.evidence))
    return 0 if result["passed"] else (2 if blocked else 1)


if __name__ == "__main__":
    raise SystemExit(main())
