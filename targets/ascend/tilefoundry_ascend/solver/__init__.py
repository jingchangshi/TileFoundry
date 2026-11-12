"""Single native solver surface, with no fallback candidate protocol."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..plan.mainline import MainlineNotImplemented

__all__ = ["SolverError", "Assignment", "solve_region"]


class SolverError(RuntimeError):
    """A named solver refusal."""


@dataclass(frozen=True)
class Assignment:
    status: str = "UNKNOWN"
    starts: tuple = ()
    stages: tuple = ()
    phases: tuple = ()
    versions: tuple = ()
    issue_order: tuple = ()
    optimality: str = "UNKNOWN"


def solve_region(problem, deadline=None):
    raise MainlineNotImplemented("SOLVER_SHARED_REGION_NOT_IMPLEMENTED")
