"""Read the one typed pipeline schema off a HIR module."""
from __future__ import annotations

from typing import Any

__all__ = ["collect_buffer_versions", "collect_pipeline_declarations"]


def _loops(module) -> list:
    from tilefoundry.ir.hir.loop_region import LoopRegion  # noqa: PLC0415
    from tilefoundry.ir.visitor import collect_exprs  # noqa: PLC0415

    found: list = []
    for item in getattr(module, "functions", ()) or ():
        body = getattr(item, "body", None)
        if body is None:
            continue
        found.extend(expr for expr in collect_exprs(body) if isinstance(expr, LoopRegion))
    return found


def _loop_key(loop, index: int) -> str:
    return "loop[%d]:%s" % (index, getattr(getattr(loop, "induction_var", None), "name", "?"))


def collect_pipeline_declarations(module) -> dict[str, Any]:
    """{loop_key: PipelineSpec} for loops that declare a native pipeline."""
    out: dict[str, Any] = {}
    for index, loop in enumerate(_loops(module)):
        spec = getattr(loop, "pipeline", None)
        if spec is not None:
            out[_loop_key(loop, index)] = spec
    return out


def collect_buffer_versions(module) -> dict[str, int]:
    """{loop_key:buffer_root: declared_version_upper_bound} from typed schedules.

    The key is loop-qualified so two loops with a same-named buffer cannot
    overwrite each other. Fixed(n) reports n; Auto(min,max) reports max. No
    Empty.buffers side channel and no fallback from max_stage.
    """
    out: dict[str, int] = {}
    for index, loop in enumerate(_loops(module)):
        spec = getattr(loop, "pipeline", None)
        if spec is None:
            continue
        prefix = _loop_key(loop, index)
        for schedule in getattr(spec, "buffer_schedules", ()):
            versions = schedule.versions
            out["%s:%s" % (prefix, schedule.root)] = (
                versions[1] if versions[0] == "fixed" else versions[2]
            )
    return out
