#!/usr/bin/env python3
"""压测数据的「会话（同一张卡）」解析与跨卡隔离。

背景
----
SCNet 容器被平台回收重建后**物理卡会更换**（2026-11-01 实测）。同一配置在
不同卡上绝对值可差约 30%，因此：

  * 收益结论**只用同一张卡上的 A/B**；
  * 旧卡的数据降级为「跨卡存档」，不得与新卡相减；
  * 每次压测把卡指纹写进 ``<run-dir>/run_config.txt``（``card_id=``），
    没有 ``card_id`` 的老目录不能参与「当前卡」判定。

会话模型
--------
一个**会话** = 一张卡 + 一段测量窗口。同一个会话内部三套配置（eager /
baseline / candidate）之间可以相减；不同会话之间不行。

  * 会话 1：**当前卡**（由 ``outputs/CARD_ID.txt`` 标识）——优先级最高；
  * 会话 2：历史同节点数据（2026-10-31 节点，卡指纹当时未记录）——只有在
    当前卡凑不齐某一对配置时才启用，并在报告里显式标注为「历史同节点」。

本模块被下面三个脚本共用，保证三份报告对「哪份数据属于哪张卡」的判断完全一致：

  * scripts/build_tuning_comparison.py
  * scripts/fill_tuning_tables.py
  * scripts/patch_report25_conclusion.py

用法（作为命令行查当前节点的同卡目录）::

    python3 scripts/bench_set.py
"""
import json
import os
from pathlib import Path

WS = Path(os.environ.get("ROBO_WS", "/root/private_data/workspace/robo"))
OUT = WS / "outputs"
CARD_ID_FILE = OUT / "CARD_ID.txt"

# 配置名 → 运行目录名前缀（一个配置可能有多个历史前缀）
CONFIG_PREFIXES = {
    "eager": ("fullbench_eager_*",),
    "baseline": ("fullbench_baseline_*", "fullbench_graph_*"),
    "candidate": ("fullbench_candidate_*",),
}

CONFIG_LABEL = {
    "eager": "首提基线 eager",
    "baseline": "graph 基线",
    "candidate": "最优版",
}

# 官方六档（输入长度, 并发）
ORDER = [(4096, 1), (4096, 8), (4096, 64), (16384, 1), (16384, 8), (16384, 64)]
TIER_NAME = {(4096, 1): "4096/并发1", (4096, 8): "4096/并发8", (4096, 64): "4096/并发64",
             (16384, 1): "16384/并发1", (16384, 8): "16384/并发8", (16384, 64): "16384/并发64"}

# 历史同节点会话（卡指纹当时未记录）：只在当前卡凑不齐时兜底，且必须标注为历史。
LEGACY_LABEL = "2026-10-31 节点（历史同节点，卡指纹未记录）"
LEGACY_PATHS = {
    "candidate": ["outputs/fullbench_candidate_20261031_231152"],
    "baseline": ["outputs/phase5_ab_noneager/noneager_confirm_20261031_222226",
                 "outputs/fullbench_baseline_20261101_043223"],
    "eager": ["logs/clean_repro_full/clean_repro_full_20261031/bench"],
}


# --------------------------------------------------------------------------- 指纹
def current_card_id():
    """当前节点的卡指纹；读不到返回 None（此时不得做同卡判定）。"""
    if not CARD_ID_FILE.is_file():
        return None
    try:
        for line in CARD_ID_FILE.read_text(encoding="utf-8").splitlines():
            if line.startswith("card_id="):
                return line.split("=", 1)[1].strip() or None
    except OSError:
        return None
    return None


def current_card_info():
    """当前节点的完整卡指纹信息（dict）。"""
    info = {}
    if CARD_ID_FILE.is_file():
        try:
            for line in CARD_ID_FILE.read_text(encoding="utf-8").splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    info[k.strip()] = v.strip()
        except OSError:
            pass
    return info


def run_card_id(run_dir):
    """某个运行目录记录的卡指纹；老目录没记录时返回 None。"""
    f = Path(run_dir) / "run_config.txt"
    if not f.is_file():
        return None
    try:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.startswith("card_id="):
                return line.split("=", 1)[1].strip() or None
    except OSError:
        return None
    return None


# --------------------------------------------------------------------------- 目录
def list_runs(config):
    """某配置的全部运行目录，按目录名（时间戳）升序。"""
    dirs = []
    for pat in CONFIG_PREFIXES.get(config, ()):
        for d in OUT.glob(pat):
            if d.is_dir():
                dirs.append(d)
    return sorted(set(dirs), key=lambda d: d.name)


def resolve(config, card_id=None):
    """返回 (同卡运行目录 or None, [非同卡运行目录...])。

    同卡判定 = 该目录 run_config.txt 里记录的 card_id 与当前节点的 card_id 相同。
    没有记录 card_id 的老目录一律算「非同卡」（只作存档）。
    """
    if card_id is None:
        card_id = current_card_id()
    same, other = None, []
    for d in list_runs(config):
        rid = run_card_id(d)
        if card_id and rid and rid == card_id:
            same = d  # 目录名按时间戳升序，最后一个即最新
        else:
            other.append(d)
    return same, other


def resolve_all(config, card_id=None):
    """某配置在指定卡上的**全部**运行目录（按目录名时间戳升序）。

    为什么需要「全部」：单档耗时可能很长（例如 16384/并发64 单轮约 3.8 h），
    实务上会把便宜档与重档**分两次服务实例**跑。此时同一张卡上同一配置有多个目录，
    报告必须把它们合并（按档位聚合），而不是只取最新那一个。
    """
    if card_id is None:
        card_id = current_card_id()
    if not card_id:
        return []
    return [d for d in list_runs(config) if run_card_id(d) == card_id]


def sessions():
    """返回 [(会话标签, {配置: [目录...]})]，第一个是当前卡（若有）。"""
    out = []
    card = current_card_id()
    if card:
        cur = {}
        for cfg in ("eager", "baseline", "candidate"):
            dirs = resolve_all(cfg, card)
            if dirs:
                cur[cfg] = dirs
        if cur:
            out.append(("当前节点（同一张卡）", cur))
    legacy = {}
    for cfg, paths in LEGACY_PATHS.items():
        dirs = [WS / p for p in paths if (WS / p).is_dir()]
        if dirs:
            legacy[cfg] = dirs
    if legacy:
        out.append((LEGACY_LABEL, legacy))
    return out


def pick_pair(cfg_a, cfg_b):
    """在**同一个会话**内同时拿到 cfg_a / cfg_b 的数据。

    返回 (会话标签, cfg_a 目录列表, cfg_b 目录列表)；凑不齐返回 (None, [], [])。
    绝不会跨会话配对——那正是换卡后最容易产生的错误结论。
    """
    for label, cfgmap in sessions():
        if cfg_a in cfgmap and cfg_b in cfgmap:
            return label, cfgmap[cfg_a], cfgmap[cfg_b]
    return None, [], []


def pick_single(cfg):
    """取某个配置在「当前卡」上的**全部**运行目录；没有则回退到历史同节点会话。

    返回多个目录是因为同一配置可能被拆成多次服务实例跑（见 resolve_all），
    调用方需用 load_dirs 合并，而不是只取 [0]。
    """
    card = current_card_id()
    if card:
        dirs = resolve_all(cfg, card)
        if dirs:
            return "当前节点（同一张卡）", dirs
    for label, cfgmap in sessions():
        if cfg in cfgmap:
            return label, cfgmap[cfg]
    return None, []


# --------------------------------------------------------------------------- 数据
def load_rows(run_dir):
    """读取一个目录下的 *.jsonl，按 (输入长度, 并发) 聚合；同一文件内同档位只取第一次。"""
    rows = {}
    if run_dir is None:
        return rows
    d = Path(run_dir)
    if not d.is_dir():
        return rows
    files = sorted(d.glob("round*.jsonl"))
    if not files:
        files = sorted(d.glob("*.jsonl"))
    for f in files:
        seen = set()
        try:
            text = f.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (r.get("random_input_len"), r.get("max_concurrency"))
            if key in seen:
                continue
            seen.add(key)
            rows.setdefault(key, []).append(r)
    return rows


def load_dirs(dirs):
    """聚合多个目录的数据（同一档位多轮会累积）。"""
    rows = {}
    for d in dirs or []:
        for k, v in load_rows(d).items():
            rows.setdefault(k, []).extend(v)
    return rows


def mean(rs, field):
    return sum(float(r.get(field, 0) or 0) for r in rs) / len(rs) if rs else 0.0


def describe():
    """当前节点的同卡数据概览（供脚本打印与排查）。"""
    card = current_card_id()
    info = current_card_info()
    print(f"card_id      : {card or '（未知：缺少 outputs/CARD_ID.txt）'}")
    print(f"card_source  : {info.get('card_id_source', '—')}")
    print(f"card_name    : {info.get('card_name', '—')}")
    print(f"card_gfx     : {info.get('card_gfx', '—')}")
    print(f"node_host    : {info.get('node_host', '—')}")
    for cfg in ("eager", "baseline", "candidate"):
        same, other = resolve(cfg, card)
        print(f"[{cfg}] 同卡: {same if same else '（无）'}")
        for o in other:
            print(f"        存档: {o}  (card_id={run_card_id(o) or '未记录'})")
    print("--- 会话 ---")
    for label, cfgmap in sessions():
        print(f"  [{label}] " + ", ".join(f"{k}={len(v)} 目录" for k, v in cfgmap.items()))


if __name__ == "__main__":
    describe()
