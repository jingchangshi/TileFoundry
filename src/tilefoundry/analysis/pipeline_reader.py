"""Read typed pipeline and buffer-version declarations off a HIR module.

Kept separate from the checks so the reader can be reused by the finalize passes
without dragging the refusal policy along.
"""

from __future__ import annotations

from typing import Any

__all__ = ["collect_buffer_versions", "collect_pipeline_declarations"]


def _loops(module) -> list:
    """Every LoopRegion in the module, in a stable document order."""
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
    """A stable identity for a loop within one program image.

    Position in document order plus the induction variable name. Deliberately NOT
    id(): identity would differ between two parsed copies of identical source, so
    a comparison of two views of the same program could never match.
    """
    return "loop[%d]:%s" % (index, getattr(loop.induction_var, "name", "?"))


def collect_pipeline_declarations(module) -> dict[str, Any]:
    """{loop_key: PipelineSpec} for loops that actually declare a pipeline."""
    out: dict[str, Any] = {}
    for index, loop in enumerate(_loops(module)):
        spec = getattr(loop, "pipeline", None)
        if spec is not None:
            out[_loop_key(loop, index)] = spec
    return out


def collect_buffer_versions(module) -> dict[str, int]:
    """{buffer_key: versions} read from the authored Empty.buffers declarations.

    The declaration lives on the allocation, not in the pipeline spec: the budget
    is a loop-level statement, the version count is a buffer-level one.
    """
    from tilefoundry.ir.core import Call  # noqa: PLC0415
    from tilefoundry.ir.visitor import collect_exprs  # noqa: PLC0415

    out: dict[str, int] = {}
    for item in getattr(module, "functions", ()) or ():
        body = getattr(item, "body", None)
        if body is None:
            continue
        for index, expr in enumerate(collect_exprs(body)):
            if not isinstance(expr, Call):
                continue
            # Identify the op by its CLASS, not by a name attribute: Empty carries
            # no .name, and matching prose would silently find nothing.
            op = getattr(expr, "target", None)
            if type(op).__name__ != "Empty":
                continue
            buffers = getattr(op, "buffers", 1)
            name = getattr(expr, "name", None) or "empty"
            out["empty[%d]:%s" % (index, name)] = buffers
    return out
