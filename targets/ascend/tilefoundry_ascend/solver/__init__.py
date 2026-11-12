"""The one solver entry: a finite region problem in, an assignment out.

There is no fallback candidate protocol and no second stage/version interpretation:
the constraints live in `tilefoundry.vendor.tilelang_z3.core`, this module only
chooses the II candidates, enforces the shared deadline, and records what it proved.

UNKNOWN is not UNSAT and never becomes installable: a timed-out search returns the
candidates it did decide, and the caller must refuse to build a plan from anything
that is not SAT.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from tilefoundry.vendor.tilelang_z3 import adapters as _adapters
from tilefoundry.vendor.tilelang_z3 import core as _core

__all__ = ["Assignment", "Optimality", "SolverError", "ScheduleProblem", "solve_region"]


class SolverError(RuntimeError):
    """A named solver refusal."""


@dataclass(frozen=True)
class Optimality:
    """What the search actually proved, kept apart from what it returned."""

    ii_checks: tuple = ()
    ii_optimal_within_bounds: bool = False
    memory_optimal_for_ii: bool = False


@dataclass(frozen=True)
class Assignment:
    status: str = _core.UNKNOWN
    ii: int | None = None
    starts: dict = field(default_factory=dict)
    stages: dict = field(default_factory=dict)
    phases: dict = field(default_factory=dict)
    versions: dict = field(default_factory=dict)
    issue_order: tuple = ()
    optimality: Optimality = field(default_factory=Optimality)
    detail: str = ""


@dataclass(frozen=True)
class ScheduleProblem:
    """The numeric projection plus the finite II domain the author declared."""

    core: _core.Problem
    fixed_ii: int | None = None
    ii_min: int = 1
    ii_max: int = 1

    def candidates(self):
        """The declared finite domain, narrowed by a necessary lower bound.

        The domain is never widened: when the bound lies above `ii_max` the
        region is UNSAT within what the author declared, not silently searched
        outside it.
        """
        if self.fixed_ii is not None:
            return (self.fixed_ii,)
        low = max(1, self.ii_min, _core.lower_bound_ii(self.core))
        if low > self.ii_max:
            return ()
        return tuple(range(low, self.ii_max + 1))


def _remaining_ms(deadline):
    if deadline is None:
        return None
    remaining = (float(deadline) - time.monotonic()) * 1000.0
    return max(0, int(remaining))


def solve_region(problem, deadline=None):
    """Decide a finite region within one shared absolute deadline.

    `deadline` is a process-monotonic absolute time in seconds; None means
    explicitly unbounded. The budget is read once per candidate and never reset.
    """
    if not isinstance(problem, ScheduleProblem):
        raise SolverError(
            "SOLVER_II_DOMAIN: solve_region needs a declared finite II domain; a bare "
            "problem carries none, and inventing [1, lower_bound] would call an "
            "unsearched range declared"
        )
    checks = []
    for ii in problem.candidates():
        timeout = _remaining_ms(deadline)
        if timeout is not None and timeout <= 0:
            return Assignment(status=_core.UNKNOWN, optimality=Optimality(ii_checks=tuple(checks)),
                              detail="SOLVER_DEADLINE_EXPIRED")
        status, assignment, detail = _core.solve(problem.core, ii, timeout_ms=timeout)
        checks.append((ii, status))
        if status == _core.SAT:
            violations = _core.verify_assignment(problem.core, assignment)
            if violations:
                raise SolverError("SOLVER_POSTCHECK_FAILED: %s" % (", ".join(violations),))
            proved_smaller = all(previous_status == _core.UNSAT for _ii, previous_status in checks[:-1])
            optimality = Optimality(
                ii_checks=tuple(checks),
                ii_optimal_within_bounds=bool(proved_smaller),
                memory_optimal_for_ii=False,
            )
            return Assignment(
                status=_core.SAT, ii=ii, starts=dict(assignment.starts),
                stages=dict(assignment.stages), phases=dict(assignment.phases),
                versions=dict(assignment.versions), issue_order=tuple(assignment.issue_order),
                optimality=optimality, detail=detail,
            )
        if status == _core.UNKNOWN:
            return Assignment(status=_core.UNKNOWN, optimality=Optimality(ii_checks=tuple(checks)),
                              detail=detail)
    return Assignment(status=_core.UNSAT, optimality=Optimality(ii_checks=tuple(checks)),
                      detail="UNSAT_WITHIN_DECLARED_II_BOUNDS")
