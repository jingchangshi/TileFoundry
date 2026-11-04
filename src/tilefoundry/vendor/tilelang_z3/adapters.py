"""TF and TileLang adapters over ONE physical pure-Z3 core (WBS 04-06).

Both adapters import the SAME module object (tilefoundry.vendor.tilelang_z3.core); there
is no second solver implementation to drift. This is what "shared core" means here: not
two similar files, and not merely a shared certificate format.
"""
from __future__ import annotations

from . import core


def tf_adapter(latencies, iis, resource_flags, data_deps, resource_deps,
               owner_exclusion_deps=None, pipe_order_deps=None):
    """TF-side adapter: accepts plain Python lists, calls the shared core."""
    return core.z3_schedule_python(
        list(latencies), list(iis), list(resource_flags),
        [tuple(x) for x in data_deps], [tuple(x) for x in resource_deps],
        owner_exclusion_deps=[tuple(x) for x in (owner_exclusion_deps or [])] or None,
        pipe_order_deps=[tuple(x) for x in (pipe_order_deps or [])] or None,
    )


def tl_adapter(latencies, iis, resource_flags, data_deps, resource_deps,
               owner_exclusion_deps=None, pipe_order_deps=None):
    """TileLang-side adapter: same physical core, same argument surface."""
    return core.z3_schedule_python(
        list(latencies), list(iis), list(resource_flags),
        [tuple(x) for x in data_deps], [tuple(x) for x in resource_deps],
        owner_exclusion_deps=[tuple(x) for x in (owner_exclusion_deps or [])] or None,
        pipe_order_deps=[tuple(x) for x in (pipe_order_deps or [])] or None,
    )


def shared_core_sha256() -> str:
    """Both adapters must report this SAME physical file hash."""
    assert tf_adapter.__module__ == tl_adapter.__module__ == __name__
    return core.core_source_sha256()
