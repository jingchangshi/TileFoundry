"""Single shared solver boundary.

The native mainline has one solve_region entry point. There is no second
stage-count schema, no hidden stage/version mapping, and no serial fallback
candidate. An implementation that cannot decide a region must report UNKNOWN
and the caller must refuse to install it.
"""
from __future__ import annotations

import hashlib
from importlib.util import find_spec
from pathlib import Path

__all__ = ["Z3_AVAILABLE", "SolverNotImplemented", "core_source_sha256",
           "z3_schedule_python", "z3_schedule_loop_python"]

Z3_AVAILABLE = find_spec("z3") is not None


class SolverNotImplemented(NotImplementedError):
    """Named refusal while the single native solver is wired in MAIN-A."""


def _refuse(name):
    raise SolverNotImplemented(
        "%s: the native mainline solver must be implemented by the shared solver; "
        "no legacy stage/version mapping or fallback candidate exists" % name
    )


def z3_schedule_python(*args, **kwargs):
    return _refuse("z3_schedule_python")


def z3_schedule_loop_python(*args, **kwargs):
    return _refuse("z3_schedule_loop_python")


def core_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
