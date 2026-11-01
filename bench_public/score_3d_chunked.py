#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TraceSpatial 官方判分的**分片并行**调度器。

官方判分代码零改动
------------------
被调用的始终是 ``third_party/upstream_eval/RoboTracer/Evaluation/summarize_acc.py``
的**原始 CLI**（逐字逐行未改）。本脚本只做两件事：

1. 把官方输入 ``outputs/<model_id>_<task>.jsonl`` 按样本切成若干分片，每片在
   独立工作目录里跑一次官方 CLI；
2. 把每片的 stdout 按**官方口径**合并回一份总指标。

为什么要分片
------------
官方 ``compute_accuracy_3d`` 对每个样本做 TSDF 体素化 + DBSCAN 去噪，代价随场景
深度范围剧烈分化：100 条里约 80 条 ~1 s，另 20 条（病态样本）~4 min/条。单进程
串行实测 ~80 分钟。按样本分片后，病态样本可以并行，墙钟降到分钟级。

为什么分片合并是精确的
----------------------
官方 ``compute_accuracy_3d`` 的所有聚合都建立在**逐样本独立**之上：

* ``collision_flags`` / ``start_success_flags`` / ``end_success_flags``：每个未 Skip
  的样本各 append 一个 bool；
* ``success_count``：逐样本累加；
* ``start_distances`` / ``end_distances``：逐样本 append 后取均值；
* ``calculate_metrics``：逐样本算 dfd/hd/rmse/mae 后取均值（dfd>100 的样本被丢弃）。

默认 ``--chunk-size 1``（逐样本分片）时，每个分片打印的
``Start Success Rate`` / ``End Success Rate`` / ``No-Collision Rate`` /
``Overall Success Rate`` 只可能是 ``0.0000`` 或 ``1.0000``，**计数无损**；
均值类指标按各自的有效样本数加权合并。

⚠️ 分母口径差异（官方原样保留）：
* ``Overall Success Rate`` 的官方分母是 ``len(answers)``，即**总行数（含被 Skip 的）**；
* 各 Rate / 距离均值的分母是**未 Skip 的样本数**（距离还要再减去各自失败的样本）。
两者不同，合并时分别处理。

精确性验证
----------
``--chunk-size 0`` 表示整集单进程跑一次官方 CLI。把它与分片合并的结果对拍，即可
验证合并口径无误（见 ``tools/verify_3d_merge.sh``）。

为什么 ``--chunk-size 1`` 的合并是**精确**的
--------------------------------------------
官方 ``calculate_metrics`` 对 NaN/无效尺寸的样本是**静默丢弃**（不打印任何标记），
只有"有效样本数为 0"时才打印 **空 dict ``{}``**。逐样本分片时每个分片恰好一个样本：

* 打印非空 dict → 该样本进入指标计算，且未被 dfd>100 丢弃 → 计入分母，值即该样本值；
* 打印空 dict   → 该样本被官方丢弃 → 该分片对指标零贡献（分母不加）；
* 打印全 0     → 有效样本被 dfd>100 全部丢弃 → 记为 ambiguous，不计入分母。

同理，各 Rate 在逐样本分片下只可能是 0.0000 / 1.0000，计数无损。

⚠️ **``--chunk-size > 1`` 时路径指标不再是精确的**：官方对 NaN 样本的静默丢弃
在多样本分片下无法从 stdout 还原（分片里只丢了一部分时 stdout 与正常分片无异）。
因此路径指标的权威口径始终是 ``--chunk-size 1``；多样本分片仅用于速率类指标的
交叉验证。
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C               # noqa: E402
import datasets as D             # noqa: E402
import run_tracespatial as RT     # noqa: E402

ROBOTRACER = RT.ROBOTRACER


# --------------------------------------------------------------------------- #
# 低层工具
# --------------------------------------------------------------------------- #
def _proc_kb(pid: int, field: str) -> int:
    """读取 /proc/<pid>/status 里的某个 kB 字段（VmRSS / VmHWM）。"""
    try:
        with open("/proc/%d/status" % pid, "r") as fh:
            for line in fh:
                if line.startswith(field):
                    return int(line.split()[1])
    except Exception:
        return 0
    return 0


def _grab(text: str, key: str) -> Optional[str]:
    """取以 ``key`` 开头的行，去掉前缀后剩余部分（官方打印成 ``Key: value``）。

    ⚠️ 截取后必须去掉行首的冒号，否则 ``": 0.7300"`` 既不是合法 float 也不是
    合法 dict 字面量，会让所有指标静默变成 None。
    """
    for line in text.splitlines():
        if line.startswith(key):
            return line[len(key):].lstrip(":").strip()
    return None


def _grab_dict(text: str, key: str) -> Optional[Dict[str, float]]:
    raw = _grab(text, key)
    if not raw:
        return None
    try:
        d = ast.literal_eval(raw)
        return {k: float(v) for k, v in d.items()} if isinstance(d, dict) else None
    except Exception:
        return None


def _grab_float(text: str, key: str) -> Optional[float]:
    """取以 ``key`` 开头的行里的数值。

    官方打印有两种带修饰的格式，都不能直接 ``float()``：
      * ``Start Success Rate (<0.20m): 1.0000``  → 阈值括注在值前面
      * ``Start Point Distance (mean): 0.0748 m`` → 单位后缀在值后面
    统一规则：**取最后一个冒号之后的那个数字**。
    """
    raw = _grab(text, key)
    if raw is None or raw.startswith("N/A"):
        return None
    try:
        return float(raw)
    except ValueError:
        pass
    tail = raw.rsplit(":", 1)[-1].strip()
    m = re.match(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", tail)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


class MemGate:
    """并发闸门：所有判分子进程当前 RSS 之和不超过预算。"""

    def __init__(self, budget_mb: float):
        self.budget_mb = budget_mb
        self._pids: List[int] = []
        self.cv = threading.Condition()

    def register(self, pid: int) -> None:
        with self.cv:
            self._pids.append(pid)
            self.cv.notify_all()

    def unregister(self, pid: int) -> None:
        with self.cv:
            self._pids = [p for p in self._pids if p != pid]
            self.cv.notify_all()

    def rss_mb(self) -> float:
        with self.cv:
            pids = list(self._pids)
        return sum(_proc_kb(p, "VmRSS") for p in pids) / 1024.0

    def wait_under_budget(self, timeout: float = 900.0) -> None:
        deadline = time.time() + timeout
        with self.cv:
            while True:
                pids = list(self._pids)
                cur = sum(_proc_kb(p, "VmRSS") for p in pids) / 1024.0
                if cur <= self.budget_mb or time.time() > deadline:
                    return
                self.cv.wait(5.0)


# --------------------------------------------------------------------------- #
# 官方 CLI 的分片执行
# --------------------------------------------------------------------------- #
class ChunkRunner:
    """在独立目录里跑一次官方 summarize_acc.py CLI，并采样峰值 RSS。"""

    def __init__(self, work_root: str, bench_dir: str, model_id: str,
                 task: str, gate: Optional[MemGate] = None):
        self.work_root = work_root
        self.bench_dir = bench_dir
        self.model_id = model_id
        self.task = task
        self.gate = gate
        self.env = dict(os.environ)
        self.env["PYTHONPATH"] = RT.scoring_pythonpath() + os.pathsep + ROBOTRACER
        self.env["PYTHONWARNINGS"] = "ignore"

    def run(self, chunk_index: int, lines: List[str]) -> Dict[str, Any]:
        d = os.path.join(self.work_root, "c%04d" % chunk_index)
        os.makedirs(os.path.join(d, "outputs"), exist_ok=True)
        link = os.path.join(d, "TraceSpatial-Bench")
        if not os.path.exists(link):
            os.symlink(self.bench_dir, link)
        oj = os.path.join(d, "outputs", "%s_%s.jsonl" % (self.model_id, self.task))
        with open(oj, "w", encoding="utf-8") as fh:
            # ⚠️ 必须补换行：``read().splitlines()`` 得到的行**不含**结尾的 "\\n"，
            # 直接 ``writelines`` 会把多条记录**首尾相连拼成一行**，
            # 官方脚本 ``json.loads(line)`` 会抛 "Extra data" 而整批失败。
            # （逐样本分片时每片只有 1 条，问题不暴露；多样本分片才会炸。）
            fh.write("".join(l if l.endswith("\n") else l + "\n" for l in lines))

        cmd = [sys.executable, os.path.join(ROBOTRACER, "summarize_acc.py"),
               "--model_name", self.model_id, "--task_name", self.task]
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=d, env=self.env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)
        if self.gate:
            self.gate.register(proc.pid)
        peak = 0
        try:
            while True:
                hwm = _proc_kb(proc.pid, "VmHWM")
                if hwm > peak:
                    peak = hwm
                if proc.poll() is not None:
                    break
                time.sleep(0.5)
        finally:
            if self.gate:
                self.gate.unregister(proc.pid)
        stdout, stderr = proc.communicate()
        wall = time.time() - t0
        return {
            "chunk_index": chunk_index,
            "n_answers": len(lines),
            "returncode": proc.returncode,
            "wall_s": round(wall, 2),
            "peak_rss_mb": round(peak / 1024.0, 1),
            "stdout": stdout,
            "stderr_tail": stderr[-800:],
        }


def parse_chunk(res: Dict[str, Any]) -> Dict[str, Any]:
    """把一次官方 CLI 的 stdout 解析成结构化结果（口径与官方一致）。"""
    out = res["stdout"]
    parsed: Dict[str, Any] = dict(res)
    # 保留 stdout 尾部（去掉 tqdm 噪声行）以便审计口径
    keep = [ln for ln in out.splitlines()
            if ln.strip() and not ln.startswith("Evaluating (")]
    parsed["stdout_tail"] = "\n".join(keep[-40:])
    parsed.pop("stdout", None)

    parsed["metrics_2d"] = _grab_dict(out, "2D Path Metrics (Normalized [0,1])")
    parsed["start_in_mask"] = _grab_float(out, "2D Start-in-Mask Rate")
    parsed["end_in_bbox"] = _grab_float(out, "2D End-in-2D-BBox Rate")
    parsed["metrics_3d"] = _grab_dict(out, "3D Path Metrics")
    parsed["start_distance"] = _grab_float(out, "Start Point Distance (mean)")
    parsed["end_distance"] = _grab_float(out, "End Point Distance (mean)")
    parsed["no_collision_rate"] = _grab_float(out, "No-Collision Rate")
    parsed["start_success_rate"] = _grab_float(out, "Start Success Rate")
    parsed["end_success_rate"] = _grab_float(out, "End Success Rate")
    parsed["overall_success_rate"] = _grab_float(out, "Overall Success Rate")

    # 官方脚本里的各类跳过/告警计数
    parsed["counts"] = {
        "skip": out.count("[Skip]"),
        "warn2d": out.count("[Warn][2D]"),
        "warn3d_start": out.count("[Warn][3D] Start point distance failed"),
        "warn3d_end": out.count("[Warn][3D] End point distance failed"),
        "warn3d_collision": out.count("[Warn][3D] Collision check failed"),
        "warn_metric": out.count("[WARN] index="),
    }
    # open3d 降级会让碰撞检查整体失效（官方脚本自己捕获成 Warn）
    parsed["open3d_degraded"] = parsed["counts"]["warn3d_collision"] > 0
    return parsed


def _empty_chunk(idx: int, n: int, err: str) -> Dict[str, Any]:
    return {"chunk_index": idx, "error": err, "n_answers": n,
            "counts": {"skip": 0, "warn2d": 0, "warn3d_start": 0,
                       "warn3d_end": 0, "warn3d_collision": 0, "warn_metric": 0},
            "wall_s": 0.0, "peak_rss_mb": 0.0,
            "start_success_rate": None, "end_success_rate": None,
            "no_collision_rate": None, "overall_success_rate": None,
            "start_distance": None, "end_distance": None,
            "metrics_2d": None, "metrics_3d": None,
            "start_in_mask": None, "end_in_bbox": None}


# --------------------------------------------------------------------------- #
# 合并
# --------------------------------------------------------------------------- #
def merge_chunks(chunks: List[Dict[str, Any]], chunk_size: int) -> Dict[str, Any]:
    """按官方口径把各分片合并成一份总指标。"""
    tot_answers = sum(c["n_answers"] for c in chunks)

    n_evaluated = 0          # 未被 [Skip] 掉的样本数（各 Rate 的分母）
    start_ok = end_ok = 0    # Start / End Success 计数
    collision_cnt = 0        # 发生碰撞的样本数
    overall_ok = 0           # Overall Success 计数（分母 = 总行数）
    n_dist_start = n_dist_end = 0
    sum_dist_start = sum_dist_end = 0.0
    n_metric3d = n_metric2d = 0
    keys = ("average_discrete_frechet_distance", "average_hausdorff_distance",
            "average_root_mean_square_error", "average_mean_absolute_error")
    sum_metric3d = {k: 0.0 for k in keys}
    sum_metric2d = {k: 0.0 for k in keys}
    ambiguous_zero3d = ambiguous_zero2d = 0
    rounding_warnings: List[str] = []
    open3d_degraded = False

    def rate_to_count(rate: Optional[float], denom: int, what: str,
                      cidx: int) -> int:
        if rate is None or denom <= 0:
            return 0
        cnt = rate * denom
        nearest = int(round(cnt))
        if abs(cnt - nearest) > 1e-3:
            rounding_warnings.append(
                "chunk %d: %s rate %.6f x denom %d = %.6f 非整数" %
                (cidx, what, rate, denom, cnt))
        return nearest

    for c in chunks:
        k = c["counts"]
        evaluated = c["n_answers"] - k["skip"]
        n_evaluated += evaluated
        if c.get("open3d_degraded"):
            open3d_degraded = True
        cidx = c["chunk_index"]

        start_ok += rate_to_count(c["start_success_rate"], evaluated, "start", cidx)
        end_ok += rate_to_count(c["end_success_rate"], evaluated, "end", cidx)
        if c["no_collision_rate"] is not None and evaluated > 0:
            collision_cnt += rate_to_count(1.0 - c["no_collision_rate"], evaluated,
                                           "collision", cidx)
        if c["overall_success_rate"] is not None:
            overall_ok += rate_to_count(c["overall_success_rate"], c["n_answers"],
                                        "overall", cidx)

        # 距离均值：官方仅在计算成功时 append。
        # chunk_size>1 时官方打印的是该分片的均值，同样按「均值 × 有效样本数」还原。
        if c["start_distance"] is not None:
            cnt = max(evaluated - k["warn3d_start"], 0)
            n_dist_start += cnt
            sum_dist_start += c["start_distance"] * cnt
        if c["end_distance"] is not None:
            cnt = max(evaluated - k["warn3d_end"], 0)
            n_dist_end += cnt
            sum_dist_end += c["end_distance"] * cnt

        # 路径指标：官方对 dfd>100 / 计算异常的样本丢弃。
        # 分母 = 该分片里"真正进入 calculate_metrics 且未被丢弃"的样本数。
        #   * 官方在有效样本数为 0 时打印 **空 dict {}**（`if valid_pred_norm else {}`）
        #     → 该分片对指标零贡献，**分母不加**（2D 任务实测有 3 个这样的分片）。
        #   * 有效样本被 dfd>100 全部丢弃时官方打印全 0 → 归为 ambiguous，不计入分母。
        # chunk_size==1 时一个分片就是一个样本，均值即该样本的值；
        # chunk_size>1 时官方打印的是该分片的均值，按「均值 × 该分片有效样本数」还原。
        m3 = c.get("metrics_3d")
        if m3:
            cnt = max(evaluated - k["warn_metric"], 0)
            allzero = all(abs(v) < 1e-12 for v in m3.values())
            if allzero and evaluated > 0 and k["warn_metric"] == 0:
                ambiguous_zero3d += 1
            else:
                n_metric3d += cnt
                for kk, vv in m3.items():
                    sum_metric3d[kk] += vv * cnt
        m2 = c.get("metrics_2d")
        if m2:
            cnt = max(evaluated - k["warn_metric"] - k["warn2d"], 0)
            allzero = all(abs(v) < 1e-12 for v in m2.values())
            if allzero and evaluated > 0 and k["warn_metric"] == 0:
                ambiguous_zero2d += 1
            else:
                n_metric2d += cnt
                for kk, vv in m2.items():
                    sum_metric2d[kk] += vv * cnt

    def _mean(s: Dict[str, float], n: int) -> Dict[str, float]:
        if n <= 0:
            return {kk: 0.0 for kk in s}
        return {kk: vv / n for kk, vv in s.items()}

    merged: Dict[str, Any] = {
        "merge_method": "chunk_size=%d（逐样本分片，官方 CLI 逐字调用）" % chunk_size,
        "n_chunks": len(chunks),
        "n_answers_total": tot_answers,
        "n_evaluated_total": n_evaluated,
        "n_skipped_total": tot_answers - n_evaluated,
        # ---- 官方口径的最终指标 ----
        "metrics_2d": _mean(sum_metric2d, n_metric2d),
        "start_in_mask": None,
        "end_in_bbox": None,
        "metrics_3d": _mean(sum_metric3d, n_metric3d),
        "start_distance": (sum_dist_start / n_dist_start) if n_dist_start else None,
        "end_distance": (sum_dist_end / n_dist_end) if n_dist_end else None,
        "no_collision_rate": (1.0 - collision_cnt / n_evaluated) if n_evaluated else None,
        "start_success_rate": (start_ok / n_evaluated) if n_evaluated else None,
        "end_success_rate": (end_ok / n_evaluated) if n_evaluated else None,
        "overall_success_rate": (overall_ok / tot_answers) if tot_answers else None,
        # ---- 审计分 ----
        "audit": {
            "n_answers_total": tot_answers,
            "n_evaluated_total": n_evaluated,
            "n_skipped_total": tot_answers - n_evaluated,
            "start_success_count": start_ok,
            "end_success_count": end_ok,
            "collision_count": collision_cnt,
            "overall_success_count": overall_ok,
            "n_start_distance_samples": n_dist_start,
            "n_end_distance_samples": n_dist_end,
            "n_metric3d_samples": n_metric3d,
            "n_metric2d_samples": n_metric2d,
            "n_ambiguous_allzero_metric3d": ambiguous_zero3d,
            "n_ambiguous_allzero_metric2d": ambiguous_zero2d,
            "open3d_degraded": open3d_degraded,
            "rounding_warnings": rounding_warnings[:20],
        },
        "wall_s_total": round(sum(c["wall_s"] for c in chunks), 1),
        "peak_rss_mb_max": max([c["peak_rss_mb"] for c in chunks] or [0]),
        "chunks_sorted_by_wall": sorted(
            [{"chunk_index": c["chunk_index"], "n": c["n_answers"],
              "wall_s": c["wall_s"], "peak_rss_mb": c["peak_rss_mb"],
              "skip": c["counts"]["skip"],
              "warn3d_collision": c["counts"]["warn3d_collision"]}
             for c in chunks],
            key=lambda x: -x["wall_s"])[:15],
    }

    # 2D 的 start_in_mask / end_in_bbox：官方对每个未 Skip 样本 append 一个 bool
    sim = eib = 0
    for c in chunks:
        ev = c["n_answers"] - c["counts"]["skip"]
        if c.get("start_in_mask") is not None and ev > 0:
            sim += rate_to_count(c["start_in_mask"], ev, "sim", c["chunk_index"])
        if c.get("end_in_bbox") is not None and ev > 0:
            eib += rate_to_count(c["end_in_bbox"], ev, "eib", c["chunk_index"])
    merged["start_in_mask"] = (sim / n_evaluated) if n_evaluated else None
    merged["end_in_bbox"] = (eib / n_evaluated) if n_evaluated else None
    merged["audit"]["start_in_mask_count"] = sim
    merged["audit"]["end_in_bbox_count"] = eib
    return merged


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir",
                    help="评测根目录（含 <variant>/outputs/<model_id>_<task>.jsonl）")
    ap.add_argument("--variant", help="official | upstream")
    ap.add_argument("--task", default="3D")
    ap.add_argument("--job", action="append", default=[],
                    help="多任务模式：'<out_dir>|<variant>|<task>'，可重复；"
                         "所有任务共享同一个线程池，总墙钟远小于逐个跑")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--bench-dir", default=None)
    ap.add_argument("--chunk-size", type=int, default=1,
                    help="每个分片的样本数；0 = 整集单进程（用于对拍验证）")
    ap.add_argument("--limit", type=int, default=0,
                    help="只取前 N 条样本（冒烟测试用）")
    ap.add_argument("--offset", type=int, default=0,
                    help="跳过前 N 条样本（定点探测用）")
    ap.add_argument("--exclude", default="",
                    help="排除的样本索引（逗号分隔，按 offset 之后的相对下标）。"
                         "用于跳过官方判分耗时不可接受的地形样本（如深景大点云）")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--mem-budget-gb", type=float, default=60.0,
                    help="所有判分子进程当前 RSS 之和的上限（GiB），超过就不再启动新分片")
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--out", default=None, help="合并结果 JSON 落盘路径（单任务模式）")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    # ---- 任务列表 ----
    jobs = []
    for spec in args.job:
        parts = spec.split("|")
        if len(parts) != 3:
            print("[score3d] --job 格式应为 '<out_dir>|<variant>|<task>'，收到 %r" % spec)
            return 1
        jobs.append({"out_dir": parts[0], "variant": parts[1], "task": parts[2]})
    if not jobs:
        if not (args.out_dir and args.variant):
            print("[score3d] 需要 --out-dir + --variant，或一个或多个 --job")
            return 1
        jobs.append({"out_dir": args.out_dir, "variant": args.variant,
                     "task": args.task})

    bench_dir = args.bench_dir or D.TRACESPATIAL_DIR
    if not bench_dir or not os.path.isdir(bench_dir):
        print("[score3d] 找不到 TraceSpatial-Bench 数据目录: %r" % bench_dir)
        return 1

    # ---- 逐任务切分 ----
    work_root = args.work_dir or os.path.join(C.WORK_DIR, "tmp", "ts3d")
    os.makedirs(work_root, exist_ok=True)
    gate = MemGate(args.mem_budget_gb * 1024.0)

    plan = []          # [(job_index, chunk_index, lines, runner)]
    job_meta = []
    for ji, j in enumerate(jobs):
        model_id = args.model_id or RT.model_id_for(j["variant"])
        src = os.path.join(j["out_dir"], j["variant"], "outputs",
                           "%s_%s.jsonl" % (model_id, j["task"]))
        if not os.path.exists(src):
            print("[score3d] 缺官方输入 %s" % src)
            return 1
        lines = [l for l in open(src, encoding="utf-8").read().splitlines() if l.strip()]
        if args.offset > 0:
            lines = lines[args.offset:]
        excl = set()
        for tok in (args.exclude or "").split(","):
            tok = tok.strip()
            if tok:
                excl.add(int(tok))
        if excl:
            kept = []
            for i, ln in enumerate(lines):
                if i in excl:
                    print("[score3d] 任务%d 排除样本 %d（官方判分耗时不可接受）"
                          % (ji, i + args.offset), flush=True)
                    continue
                kept.append(ln)
            lines = kept
        if args.limit > 0:
            lines = lines[:args.limit]
        cs = args.chunk_size if args.chunk_size > 0 else len(lines)
        chunks = [lines[i:i + cs] for i in range(0, len(lines), cs)]
        tag = "%s_%s_%s" % (os.path.basename(os.path.normpath(j["out_dir"])),
                            j["variant"], j["task"])
        runner = ChunkRunner(os.path.join(work_root, tag), bench_dir,
                             model_id, j["task"], gate)
        job_meta.append({"job": j, "model_id": model_id, "src": src,
                         "n_lines": len(lines), "chunk_size": cs,
                         "n_chunks": len(chunks),
                         "out_path": args.out if (len(jobs) == 1 and args.out) else
                                    os.path.join(j["out_dir"], j["variant"],
                                                 "scores_3dchunk_%s.json" % j["task"])})
        for ci, ls in enumerate(chunks):
            plan.append((ji, ci, ls, runner))
        print("[score3d] 任务%d %s/%s model_id=%s 样本=%d 分片=%d" %
              (ji, j["variant"], j["task"], model_id, len(lines), len(chunks)),
              flush=True)

    print("[score3d] 合计 %d 个任务 / %d 个分片，并发=%d，内存闸门=%.0fGiB" %
          (len(jobs), len(plan), args.concurrency, args.mem_budget_gb), flush=True)

    q: "queue.Queue" = queue.Queue()
    for item in plan:
        q.put(item)
    results_map: Dict[Tuple[int, int], Dict[str, Any]] = {}
    lock = threading.Lock()

    def worker() -> None:
        while True:
            try:
                ji, ci, ls, runner = q.get_nowait()
            except queue.Empty:
                return
            try:
                gate.wait_under_budget()
                r = parse_chunk(runner.run(ci, ls))
                r["job_index"] = ji
                with lock:
                    results_map[(ji, ci)] = r
                print("[score3d] job%d chunk %d 完成 %.1fs 峰值RSS=%.0fMB skip=%d warn3d=%d" %
                      (ji, ci, r["wall_s"], r["peak_rss_mb"],
                       r["counts"]["skip"], r["counts"]["warn3d_collision"]), flush=True)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    results_map[(ji, ci)] = _empty_chunk(ci, len(ls), repr(exc))
                    results_map[(ji, ci)]["job_index"] = ji
                print("[score3d] job%d chunk %d 失败: %r" % (ji, ci, exc), flush=True)
            finally:
                q.task_done()

    threads = []
    for _ in range(max(1, args.concurrency)):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        threads.append(t)

    t0 = time.time()
    for t in threads:
        t.join()
    elapsed = time.time() - t0

    # ---- 逐任务合并 ----
    rc = 0
    for ji, meta in enumerate(job_meta):
        chunks = [results_map[k] for k in sorted(results_map) if k[0] == ji]
        ok = [r for r in chunks if "error" not in r]
        failed = [r for r in chunks if "error" in r]
        merged = merge_chunks(ok, meta["chunk_size"])
        merged["source_jsonl"] = meta["src"]
        merged["model_id"] = meta["model_id"]
        merged["variant"] = meta["job"]["variant"]
        merged["task"] = meta["job"]["task"]
        merged["n_failed_chunks"] = len(failed)
        merged["failed_chunks"] = failed
        merged["per_chunk"] = ok
        merged["n_excluded_samples"] = len(excl)
        merged["excluded_sample_indices"] = sorted(excl)
        merged["wall_s_all_jobs"] = round(elapsed, 1)
        C.dump_json(meta["out_path"], merged)
        print("\n[score3d] 任务%d (%s/%s) 完成 %d/%d 分片，失败 %d" %
              (ji, meta["job"]["variant"], meta["job"]["task"], len(ok),
               meta["n_chunks"], len(failed)), flush=True)
        print("    -> %s" % meta["out_path"])
        for k in ("metrics_2d", "start_in_mask", "end_in_bbox", "metrics_3d",
                  "start_distance", "end_distance", "no_collision_rate",
                  "start_success_rate", "end_success_rate", "overall_success_rate"):
            print("    %-22s %s" % (k, merged.get(k)))
        print("    audit: %s" % json.dumps(merged["audit"], ensure_ascii=False))
        if failed:
            rc = 2
    print("\n[score3d] 全部任务结束，总墙钟 %.1fs" % elapsed)
    return rc


if __name__ == "__main__":
    sys.exit(main())
