#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""公开上游评测集（CV-Bench / RefSpatial-Bench / TraceSpatial-Bench）的共用层。

本模块只做三件事，**不做任何判分**：

1. **冻结协议**：提示词模板、坐标契约、图像预处理、解码参数 —— 全部来自上游
   权威来源，逐字引用，出处写在注释里。
2. **推理层**：按冻结提示词调用本节点的 vLLM 服务，把结果落成官方期望的格式。
3. **护栏与自检**：压测期间禁止发起额外请求；提示词与已验收的
   ``scripts/robo_infer.py`` 逐字比对。

判分一律由官方脚本完成（RoboRefer / RoboTracer / FlagEvalMM），本模块不重写、
不改阈值、不改聚合规则。

坐标契约（已核实，见 reports/15_public_bench_protocol.md）
---------------------------------------------------------
RoboBrain2.5-8B-NV 的空间任务输出是 **0~1000 相对坐标**，换算基准是
**原图尺寸**（上游 ``inference.py::draw_on_image`` 的 ``rel_to_abs`` 用
``cv2.imread`` 读到的 h/w），不是缩放后的尺寸。

依据：
  * 上游 ``third_party/RoboBrain2.5/inference.py`` 的 docstring：
    "x,y are relative (0~1000)"，``rel_to_abs`` 做 ``x/1000*w``；
  * FlagEvalMM ``flagevalmm/evaluator/point_utils.text2pts_norm1k`` 同口径；
  * 本机实测输出：affordance 图 640x426 -> ``[(901, 600)]``（像素尺度不可能）。

官方实现分歧（协议差异表见 reports/15_public_bench_protocol.md）
------------------------------------------------------------
  * RoboRefer ``summarize_acc.py`` 的 "RoboBrain" 分支用 ``is_absolute=True``
    （按**像素**读）。该分支是 **RoboBrain 2.0** 的约定（2.0 输出像素坐标）；
    RoboBrain2.5 改用 0~1000 相对坐标，直接套用会把 0~1000 当像素 → 全部越界
    → 静默判 0 分。
  * 因此本次采用：**推理侧把 0~1000 换算成原图像素**，再交给**未修改的**
    官方判分函数（``is_absolute=True`` 语义）；同时保留"不换算"的官方逐字口径
    作为对照，用于量化协议差异。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from image_canonical import canonicalize_to_data_url  # noqa: E402

WORK_DIR = os.environ.get("WORK_DIR", "/root/private_data/workspace/robo")
DEFAULT_URL = "http://127.0.0.1:30000/v1/chat/completions"
DEFAULT_MODEL = "robobrain2.5-8b-nv"

# --------------------------------------------------------------------------- #
# 冻结的生成配置（验收要求 §4.3）
# --------------------------------------------------------------------------- #
# 上游 inference.py 的 max_new_tokens=768；三个基准都用它，保证不因预算截断。
MAX_NEW_TOKENS = 768
DECODE = {"temperature": 0.0, "top_p": 1.0, "seed": 0, "max_tokens": MAX_NEW_TOKENS}
# 图像数量固定为 1（三个基准均为单图任务）
IMAGES_PER_SAMPLE = 1

# --------------------------------------------------------------------------- #
# 提示词（逐字，来源：third_party/RoboBrain2.5/inference.py）
# --------------------------------------------------------------------------- #
def strip_one_trailing_period(text: str) -> str:
    """去掉**一个**结尾句点。

    上游 pointing/trajectory 模板是 ``f"{text}. Please ..."``；RefSpatial /
    TraceSpatial 的原始 prompt 自带结尾句点，直接拼接会出现 ".."。只在
    **我们自己拼句点**的地方调用它；官方逐字提示词不调用。
    """
    t = (text or "").strip()
    if t.endswith("."):
        t = t[:-1].rstrip()
    return t


def prompt_upstream(text: str, task: str) -> str:
    """RoboBrain2.5 上游 inference.py 的提示词构造（逐字）。"""
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
    raise ValueError(task)


# --------------------------------------------------------------------------- #
# 官方评测脚本里的提示词（逐字，不做任何改写）
# --------------------------------------------------------------------------- #
def prompt_refspatial_official(prompt: str) -> str:
    """RoboRefer/Evaluation/test_benchmark.py::get_prompt 的 "RoboBrain" 分支。

    官方原文：``f"{prompt} Please provide its 2D coordinates."``（不拼 suffix）。
    """
    return f"{prompt} Please provide its 2D coordinates."


def prompt_tracespatial_official(task_name: str, object_name: str,
                                 target_location: str, prompt: str) -> str:
    """RoboTracer/Evaluation/test_benchmark.py::get_prompt 的 "RoboTracer" 分支。"""
    if task_name == "2D":
        return (f'Point the 2D object-centric visual trace for the task "{prompt}". '
                f"Your answer should be formatted as a list of tuples, i.e., "
                f"[(x1, y1), (x2, y2), ...], where each tuple contains the x and y "
                f"coordinates of the point.")
    if task_name == "3D":
        return (f'Point the 3D object-centric visual trace for the task "{prompt}". '
                f"Your answer should be formatted as a list of tuples, i.e., "
                f"[(x1, y1, d1), (x2, y2, d2), ...], where each tuple contains the x "
                f"and y coordinates of the point, and d is the depth of the point.")
    raise ValueError(f"Unsupported task: {task_name}")


def prompt_cv_bench_flag_eval_mm(question: str, choices: Sequence[str]) -> str:
    """FlagEvalMM CV-Bench 的官方 prompt 链（逐字复刻）。

    复刻自官方代码：
      * ``flagevalmm/dataset/vqa_base_dataset.py::build_prompt``：先在问题后逐行
        追加 ``"\\n" + chr(ord("A") + i) + ". " + choice``，再在**最前面**加
        ``"<image 1> "``；
      * ``flagevalmm/prompt/prompt_template.py::default_prompt_func``：
        ``f"{examples}{pre_prompt}{question}\\n{post_prompt}"`` 再 ``.strip()``，
        其中 multiple-choice 的默认 post_prompt 是
        "Answer with the option's letter from the given choices directly."；
      * ``model_zoo/vlm/robobrain/model_adapter.py::process_single_item``：
        ``qs = f"{qs}<think></think><answer>"``（与上文之间**没有**空格）。

    ``"<image 1>"`` 是框架的多模态占位符，由 ``build_interleaved_message`` 在
    拼装消息时消费（图片作为独立 content 项插在该位置）。我们的服务同样用独立
    的 image content 项送图，因此这里移除该标记 —— **送进模型的文本与内容等价**。
    """
    q = question
    for i, choice in enumerate(choices):
        q += "\n" + chr(ord("A") + i) + ". " + str(choice)
    q = "<image 1> " + q                                    # VqaBaseDataset.build_prompt
    q = q + "\n" + "Answer with the option's letter from the given choices directly."
    q = q + "<think></think><answer>"                       # RoboBrain adapter
    return q.replace("<image 1> ", "", 1)


# --------------------------------------------------------------------------- #
# 自检：与已验收的 scripts/robo_infer.py 逐字比对
# --------------------------------------------------------------------------- #
def selfcheck_prompt_parity() -> bool:
    """与已验收的 scripts/robo_infer.py 逐字比对上游提示词。"""
    try:
        import robo_infer  # noqa: E402
    except Exception as exc:                       # pragma: no cover
        print("[selfcheck] 无法导入 robo_infer: %s" % exc)
        return False
    ok = True
    for task in ("pointing", "trajectory", "grounding"):
        a = prompt_upstream("TEST TEXT", task)
        b = robo_infer.build_prompt("TEST TEXT", task)
        same = a == b
        ok = ok and same
        print("[selfcheck] %-11s upstream 提示词与 robo_infer 一致: %s" %
              (task, same))
        if not same:
            print("   ours:", a)
            print("   ref :", b)
    return ok


# --------------------------------------------------------------------------- #
# 服务客户端
# --------------------------------------------------------------------------- #
def encode_image(path: str) -> str:
    """按上游 qwen_vl_utils 预处理后转 data URL（与已验收基线同源）。

    该预处理逐像素复现上游 ``process_vision_info``（实测 max|diff|=0），
    因此"用原图尺寸把 0~1000 换算成像素"与上游 ``rel_to_abs`` 口径一致。
    """
    return canonicalize_to_data_url(path)


def encode_image_bytes(data: bytes) -> str:
    """同 encode_image，但输入是原始图像字节（CV-Bench 的图像内嵌在 parquet）。"""
    import base64
    from image_canonical import canonicalize_bytes
    png = canonicalize_bytes(data)
    return "data:image/png;base64," + base64.b64encode(png).decode()


def encode_image_ref(ref) -> str:
    """``ref`` 可以是路径字符串，也可以是 ``("bytes", raw_bytes)``。"""
    if isinstance(ref, tuple) and len(ref) == 2 and ref[0] == "bytes":
        return encode_image_bytes(ref[1])
    return encode_image(ref)


def build_messages(image_paths: Sequence[Any], text: str) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": encode_image_ref(p)}}
        for p in image_paths
    ]
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def chat_completion(url: str, model: str, image_paths: Sequence[str], text: str,
                    max_tokens: int = MAX_NEW_TOKENS,
                    timeout: int = 600,
                    max_retries: int = 2) -> Dict[str, Any]:
    """调用服务；**只对传输/服务错误**重试（预登记上限），不对"答错"重试。"""
    import requests

    body = {
        "model": model,
        "messages": build_messages(image_paths, text),
        "max_tokens": max_tokens,
        "temperature": DECODE["temperature"],
        "top_p": DECODE["top_p"],
        "seed": DECODE["seed"],
    }
    last_err = None
    for attempt in range(max_retries + 1):
        t0 = time.perf_counter()
        try:
            r = requests.post(url, json=body, timeout=timeout)
            dt = time.perf_counter() - t0
            r.raise_for_status()
            data = r.json()
            return {
                "text": data["choices"][0]["message"]["content"] or "",
                "usage": data.get("usage", {}),
                "seconds": round(dt, 4),
                "finish_reason": data["choices"][0].get("finish_reason"),
                "status": r.status_code,
                "attempts": attempt + 1,
                "error": None,
            }
        except Exception as exc:                  # 传输/服务错误
            last_err = exc
            if attempt < max_retries:
                time.sleep(2 * (attempt + 1))
    return {
        "text": "", "usage": {}, "seconds": 0.0, "finish_reason": None,
        "status": "error", "attempts": max_retries + 1,
        "error": repr(last_err),
    }


def service_alive(url: str = DEFAULT_URL) -> bool:
    import requests
    health = url.rsplit("/v1/", 1)[0] + "/health"
    try:
        r = requests.get(health, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def guard_no_official_bench() -> None:
    """官方压测运行期间禁止发起额外请求（避免污染性能测量）。"""
    try:
        out = subprocess.run(["ps", "-eo", "cmd"], capture_output=True,
                             text=True, timeout=20).stdout
    except Exception:
        return
    for line in out.splitlines():
        if "bench_serving_vllm_image.py" in line and "grep" not in line:
            raise RuntimeError(
                "检测到官方压测仍在运行（bench_serving_vllm_image.py），"
                "为避免污染性能测量，公开评测必须等压测结束后再跑。")


# --------------------------------------------------------------------------- #
# 坐标换算（显式、可审计）
# --------------------------------------------------------------------------- #
def scale_1000_to_pixels(text: str, width: int, height: int,
                         task: str = "pointing") -> str:
    """把 0~1000 相对坐标的模型输出改写成**原图像素坐标**文本。

    只做坐标换算，不改写模型输出的其它内容；原始输出保留在 ``raw_output``。
    仅用于官方判分脚本按 ``is_absolute=True``（像素）读取的场合。
    """
    if not width or not height:
        return text
    if task == "pointing":
        pts = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)
        if not pts:
            return text
        out = [f"({round(int(x) / 1000.0 * width)}, {round(int(y) / 1000.0 * height)})"
               for x, y in pts]
        return " ".join(out)
    if task == "trajectory":
        pts = re.findall(r"(\d+),\s*(\d+),\s*([+-]?\d+\.?\d+)", text)
        if not pts:
            return text
        out = [f"({round(int(x) / 1000.0 * width)}, {round(int(y) / 1000.0 * height)}, {d})"
               for x, y, d in pts]
        return " ".join(out)
    if task == "grounding":
        bs = re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", text)
        if not bs:
            return text
        out = [f"[{round(int(a) / 1000.0 * width)}, {round(int(b) / 1000.0 * height)}, "
               f"{round(int(c) / 1000.0 * width)}, {round(int(d) / 1000.0 * height)}]"
               for a, b, c, d in bs]
        return " ".join(out)
    return text


def image_size(path: str) -> Tuple[int, int]:
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def parse_points_1000(text: str, task: str = "pointing") -> List[Tuple]:
    """按上游 inference.py 的正则解析 0~1000 坐标（不改写数值）。"""
    if task == "trajectory":
        pts = re.findall(r"(\d+),\s*(\d+),\s*([+-]?\d+\.?\d+)", text)
        return [(int(x), int(y), float(d)) for x, y, d in pts]
    if task == "pointing":
        pts = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)
        return [(int(x), int(y)) for x, y in pts]
    if task == "grounding":
        bs = re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", text)
        return [tuple(int(v) for v in b) for b in bs]
    return []


# --------------------------------------------------------------------------- #
# 记录与哈希
# --------------------------------------------------------------------------- #
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_hash(payload: Dict[str, Any]) -> str:
    """配置的稳定哈希（键排序 + 无空格）。"""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def dump_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def dump_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
