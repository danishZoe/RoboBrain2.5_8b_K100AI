#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""加载 FlagEvalMM 官方判分模块（绕开框架的重 ``__init__``）。

为什么需要这个包装
------------------
``FlagEvalMM/flagevalmm/evaluator/__init__.py`` 会连带 import 一大串与本次
判分无关的评估器（CLIP score / FVD / inception ...），其依赖链里有
``accelerate``；本机 ``accelerate`` 的版本与系统 numpy 1.25 不兼容
（``np._core`` 属性不存在），导致 **只要 import 官方
``tasks/cv_bench/evaluate.py`` 就会在框架的 ``__init__`` 里炸掉**，
与判分逻辑本身无关。

这里的做法是：在 ``sys.modules`` 里放入 **轻量 stub 包**，只声明
``flagevalmm`` / ``flagevalmm.evaluator`` 的 ``__path__``，让 Python 直接
加载我们要的子模块文件，而**不执行**框架那个重的 ``__init__.py``。

因此：

* 被判分的官方代码（``tasks/cv_bench/evaluate.py``、
  ``flagevalmm/evaluator/common_types.py``、
  ``flagevalmm/evaluator/point_utils.py``、``pre_process.py``）
  **一字未改**，按原路径加载；
* 只跳过与判分无关的框架装配代码（属于"导入路径适配"，已在协议文档披露）；
* 若将来框架版本变化导致 ``__init__`` 不再有问题，这个包装会自动退回
  正常 import 路径。
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from typing import Any

FE_ROOT = os.path.join(os.environ.get(
    "WORK_DIR", "/root/private_data/workspace/robo"),
    "third_party/upstream_eval/FlagEvalMM")


def _ensure_stub_packages() -> None:
    """为 flagevalmm / flagevalmm.evaluator 建立轻量 stub（若尚未正常导入）。"""
    if "flagevalmm.evaluator" in sys.modules:
        return
    try:
        importlib.import_module("flagevalmm.evaluator")
        return
    except Exception:
        pass
    pkg = types.ModuleType("flagevalmm")
    pkg.__path__ = [os.path.join(FE_ROOT, "flagevalmm")]
    sys.modules["flagevalmm"] = pkg
    sub = types.ModuleType("flagevalmm.evaluator")
    sub.__path__ = [os.path.join(FE_ROOT, "flagevalmm", "evaluator")]
    sys.modules["flagevalmm.evaluator"] = sub


def load_module(rel_path: str, name: str) -> Any:
    """按文件路径加载官方模块（不经框架 ``__init__``）。"""
    _ensure_stub_packages()
    path = os.path.join(FE_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("无法加载 %s" % path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_cv_bench_evaluate():
    """官方 ``FlagEvalMM/tasks/cv_bench/evaluate.py``。"""
    return load_module(os.path.join("tasks", "cv_bench", "evaluate.py"),
                       "cv_bench_evaluate_official")


def load_point_utils():
    """官方 ``flagevalmm/evaluator/point_utils.py``。"""
    return load_module(os.path.join("flagevalmm", "evaluator", "point_utils.py"),
                       "flagevalmm_point_utils_official")
