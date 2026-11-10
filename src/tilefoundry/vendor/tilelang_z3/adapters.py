"""TF and TileLang adapters over ONE physical pure-Z3 core (WBS 04-06/04-07).

Both adapters import the SAME module object (tilefoundry.vendor.tilelang_z3.core); there
is no second solver implementation to drift. This is what "shared core" means here: not
two similar files, and not merely a shared certificate format.

04-07 adds the LOOP adapters: both TF and TL call the identical
core.z3_schedule_loop_python function with the identical normalized FULL constraint
set. The optional 04-08 return_status flag does not create a second solver; it only
makes the core's own sat/fallback decision visible instead of hiding it in a tuple.
"""
from __future__ import annotations

from . import core


def _straight_call(latencies, iis, resource_flags, data_deps, resource_deps,
                   owner_exclusion_deps, pipe_order_deps, solve_timeout_ms):
    """ONE physical call site for both straight-line adapters.

    Keeping the two adapters on one call site is what makes "shared core" checkable: if
    this were duplicated, TF and TL could drift into different argument handling while
    still claiming to share the file.
    """
    return core.z3_schedule_python(
        list(latencies), list(iis), list(resource_flags),
        [tuple(x) for x in data_deps], [tuple(x) for x in resource_deps],
        owner_exclusion_deps=[tuple(x) for x in (owner_exclusion_deps or [])] or None,
        pipe_order_deps=[tuple(x) for x in (pipe_order_deps or [])] or None,
        solve_timeout_ms=solve_timeout_ms,
    )


def tf_adapter(latencies, iis, resource_flags, data_deps, resource_deps,
               owner_exclusion_deps=None, pipe_order_deps=None,
               solve_timeout_ms=None):
    """TF-side adapter: accepts plain Python lists, calls the shared core."""
    return _straight_call(latencies, iis, resource_flags, data_deps, resource_deps,
                          owner_exclusion_deps, pipe_order_deps, solve_timeout_ms)


def tl_adapter(latencies, iis, resource_flags, data_deps, resource_deps,
               owner_exclusion_deps=None, pipe_order_deps=None,
               solve_timeout_ms=None):
    """TileLang-side adapter: same physical core, same argument surface."""
    return _straight_call(latencies, iis, resource_flags, data_deps, resource_deps,
                          owner_exclusion_deps, pipe_order_deps, solve_timeout_ms)


def _loop_call(core_args: dict, *, return_status: bool = True):
    """ONE physical call site for both loop adapters."""
    args = dict(core_args)
    args["return_status"] = return_status
    return core.z3_schedule_loop_python(**args)


def tf_loop_adapter(core_args: dict, *, return_status: bool = True):
    """TF-side full-loop adapter: normalized inputs -> the SAME physical core."""
    return _loop_call(core_args, return_status=return_status)


def tl_loop_adapter(core_args: dict, *, return_status: bool = True):
    """TileLang-side full-loop adapter: normalized inputs -> the SAME physical core."""
    return _loop_call(core_args, return_status=return_status)


def shared_core_sha256() -> str:
    """Both straight-line adapters must report this SAME physical file hash."""
    assert tf_adapter.__module__ == tl_adapter.__module__ == __name__
    return core.core_source_sha256()


def shared_loop_core_sha256() -> str:
    """Both full-loop adapters must report this SAME physical file hash."""
    assert tf_loop_adapter.__module__ == tl_loop_adapter.__module__ == __name__
    assert tf_loop_adapter.__globals__["core"] is tl_loop_adapter.__globals__["core"]
    return core.core_source_sha256()


__all__ = ["tf_adapter", "tl_adapter", "tf_loop_adapter", "tl_loop_adapter",
           "shared_core_sha256", "shared_loop_core_sha256"]
