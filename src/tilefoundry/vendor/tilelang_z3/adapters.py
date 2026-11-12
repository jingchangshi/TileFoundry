"""Adapters over the one shared native solver boundary.

No adapter carries a second stage/version interpretation. The single native
mainline solver is wired through solve_region in MAIN-A.
"""
from __future__ import annotations

from .core import SolverNotImplemented, core_source_sha256

__all__ = ["tf_adapter", "tl_adapter", "tf_loop_adapter", "tl_loop_adapter",
           "shared_core_sha256", "shared_loop_core_sha256"]


def _refuse(name):
    raise SolverNotImplemented("%s: adapter has no second solver" % name)


def tf_adapter(*args, **kwargs):
    return _refuse("tf_adapter")


def tl_adapter(*args, **kwargs):
    return _refuse("tl_adapter")


def tf_loop_adapter(*args, **kwargs):
    return _refuse("tf_loop_adapter")


def tl_loop_adapter(*args, **kwargs):
    return _refuse("tl_loop_adapter")


def shared_core_sha256() -> str:
    return core_source_sha256()


def shared_loop_core_sha256() -> str:
    return core_source_sha256()
