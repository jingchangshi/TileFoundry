"""Adapters over the one fixed-interval scheduling core.

The Tensor-Foundry and the TileLang side both project into `tilelang_z3.core`; neither
carries a second interpretation of stage, phase or version. A projection is a complete
numeric view: a partially described problem is refused by name instead of being read as
an empty problem, because an empty problem is vacuously satisfiable.
"""
from __future__ import annotations

from . import core
from .core import (SAT, UNSAT, UNKNOWN, Assignment, Buffer, Capacity, Edge, Problem,
                   Resource, Reuse, SolverError, Task, lower_bound_ii, solve,
                   verify_assignment)

__all__ = [
    "tf_adapter", "tl_adapter", "project", "solve", "lower_bound_ii",
    "verify_assignment", "Assignment", "Buffer", "Capacity", "Edge", "Problem",
    "Resource", "Reuse", "SAT", "Task", "UNKNOWN", "UNSAT",
]

_REQUIRED = ("tasks", "edges", "buffers", "reuses", "resources", "capacities",
             "max_stage", "manual", "issue_after")


def _field(source, name):
    if isinstance(source, dict):
        return (name in source), source.get(name)
    return hasattr(source, name), getattr(source, name, None)


def project(source):
    """Project a complete native scheduling view onto the core data view.

    Every field must be present: a missing field means the projection is incomplete,
    not that the field is empty. The native plan stays authoritative; this view only
    carries the numbers the solver decides on.
    """
    if isinstance(source, core.Problem):
        return source
    missing = [name for name in _REQUIRED if not _field(source, name)[0]]
    if missing:
        raise SolverError(
            "PROJECTION_INCOMPLETE",
            "the numeric view is missing %s; an incomplete problem is refused rather "
            "than read as empty" % (", ".join(missing),),
        )
    values = {name: _field(source, name)[1] for name in _REQUIRED}
    return Problem(
        tasks=tuple(values["tasks"]), edges=tuple(values["edges"]),
        buffers=tuple(values["buffers"]), reuses=tuple(values["reuses"]),
        resources=tuple(values["resources"]), capacities=tuple(values["capacities"]),
        max_stage=values["max_stage"], manual=values["manual"],
        issue_after=tuple(values["issue_after"]),
    )


def tf_adapter(source):
    """The Tensor-Foundry projection: same core, no second schedule model."""
    return project(source)


def tl_adapter(source):
    """The TileLang projection: same core, no second schedule model."""
    return project(source)
