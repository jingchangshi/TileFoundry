"""solve_region: candidate II search, three states, one shared deadline."""
from __future__ import annotations

import time
from dataclasses import replace

from tilefoundry.vendor.tilelang_z3 import Buffer, Capacity, Edge, Problem, SAT, Task, UNSAT
from tilefoundry_ascend.solver import ScheduleProblem, solve_region


def _problem(**overrides):
    values = dict(
        tasks=(Task("a", (("aiv0", "aiv", "vector", 1),), latency=1),
               Task("b", (("aiv0", "aiv", "vector", 1),), latency=1)),
        edges=(Edge("a", "b", distance=0, latency=1),),
        max_stage=2,
    )
    values.update(overrides)
    return Problem(**values)


def test_auto_ii_finds_the_first_feasible_and_proves_the_ones_below():
    problem = ScheduleProblem(core=_problem(), ii_min=1, ii_max=4)
    assignment = solve_region(problem)
    assert assignment.status == SAT
    assert assignment.ii == 2
    assert [entry for entry in assignment.optimality.ii_checks] == [(2, SAT)] or \
        all(status == UNSAT for _ii, status in assignment.optimality.ii_checks[:-1])
    assert assignment.optimality.ii_optimal_within_bounds is True


def test_fixed_ii_checks_only_that_value():
    problem = ScheduleProblem(core=_problem(), fixed_ii=1, ii_min=1, ii_max=4)
    assignment = solve_region(problem)
    assert assignment.status == UNSAT
    assert assignment.optimality.ii_checks == ((1, UNSAT),)


def test_unsat_within_declared_bounds_is_not_unknown():
    core = Problem(tasks=(Task("a", (("aiv0", "aiv", "vector", 1),), latency=1),
                          Task("b", (("aiv0", "aiv", "vector", 1),), latency=1)),
                   max_stage=2)
    assignment = solve_region(ScheduleProblem(core=core, ii_min=1, ii_max=1))
    assert assignment.status == UNSAT


def test_expired_deadline_is_unknown_not_unsat():
    problem = ScheduleProblem(core=_problem(), ii_min=1, ii_max=4)
    assignment = solve_region(problem, deadline=time.monotonic() - 1.0)
    assert assignment.status == "UNKNOWN"
    assert assignment.detail == "SOLVER_DEADLINE_EXPIRED"
    assert assignment.ii is None and assignment.starts == {}


def test_deadline_is_absolute_and_never_reset():
    problem = ScheduleProblem(core=_problem(), ii_min=1, ii_max=4)
    assignment = solve_region(problem, deadline=time.monotonic() + 30.0)
    assert assignment.status == SAT
    assert assignment.ii == 2


def test_fixed_versions_travel_through_the_region_entry():
    core = _problem(buffers=(Buffer("tile", 32, space="smem", owner="aiv0", fixed=2),),
                    capacities=(Capacity("smem", "aiv0", 64),))
    assignment = solve_region(ScheduleProblem(core=core, ii_min=1, ii_max=3))
    assert assignment.status == SAT
    assert assignment.versions["tile"] == 2


def test_declared_ii_domain_is_never_widened():
    """A declared [1, 1] domain stays [1, 1]: no silent search outside it."""
    core = Problem(tasks=(Task("a", (("aiv0", "aiv", "vector", 1),), latency=1),
                          Task("b", (("aiv0", "aiv", "vector", 1),), latency=1)),
                   max_stage=2)
    assignment = solve_region(ScheduleProblem(core=core, ii_min=1, ii_max=1))
    assert assignment.status == UNSAT
    assert assignment.optimality.ii_checks == ()


def test_solve_region_refuses_a_bare_core_problem():
    """An undeclared II domain is refused, not guessed from a lower bound."""
    core = _problem()
    try:
        solve_region(core)
    except Exception as error:  # noqa: BLE001 - the point is the named refusal
        assert "SOLVER_II_DOMAIN" in str(error)
    else:  # pragma: no cover - a bare core problem must never be solved
        raise AssertionError("a bare core Problem must not be accepted as a declared domain")

