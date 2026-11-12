"""Fixed-interval scheduling core: the constraints it must decide.

Every case here is small enough to check by hand, and case 2 compares the solver
with an independent enumeration of the same finite domain. The point is not that
Z3 can be called: it is that an authored stage survives, a Fixed(V) is never
rewritten after SAT, and two unit tasks on one capacity-1 pipe cannot share II=1.
"""
from __future__ import annotations

import pytest

from tilefoundry.vendor.tilelang_z3 import (
    Buffer, Capacity, Edge, Problem, Reuse, SAT, Task, UNKNOWN, UNSAT, lower_bound_ii,
    solve, verify_assignment,
)

V_USE = ("aiv0", "aiv", "vector", 1)


def _pipeline(ii, **overrides):
    values = dict(
        tasks=(
            Task("load", (("aiv0", "aiv", "mte2", 1),), latency=2, stage=None),
            Task("vf", (V_USE,), latency=2),
        ),
        edges=(Edge("load", "vf", distance=0, latency=1),),
        max_stage=1,
    )
    values.update(overrides)
    return Problem(**values)


def _campaign(ii):
    return dict(tasks=(
        Task("a", (V_USE,), latency=1),
        Task("b", (V_USE,), latency=1),
    ), max_stage=1)


def test_two_task_pipeline_sat_and_start_formula():
    problem = _pipeline(2)
    status, assignment, detail = solve(problem, 2)
    assert status == SAT, detail
    for name, start in assignment.starts.items():
        assert start == 2 * assignment.stages[name] + assignment.phases[name]
    assert all(0 <= phase < 2 for phase in assignment.phases.values())
    assert verify_assignment(problem, assignment) == []


def test_small_domain_enumeration_agrees_with_solver():
    """Enumerate stage/phase by hand over a finite domain and compare statuses."""
    problem = Problem(tasks=(Task("a", (V_USE,), latency=1),
                             Task("b", (V_USE,), latency=1)),
                      edges=(Edge("a", "b", distance=0, latency=1),),
                      max_stage=1)
    for ii in (1, 2, 3):
        feasible = []
        for stage_a in range(problem.max_stage + 1):
            for stage_b in range(problem.max_stage + 1):
                for phase_a in range(ii):
                    for phase_b in range(ii):
                        start_a = ii * stage_a + phase_a
                        start_b = ii * stage_b + phase_b
                        if start_b < start_a + 1:
                            continue
                        delta = (start_b - start_a) % ii
                        if not (1 <= delta and 1 <= ii - delta):
                            continue
                        feasible.append((stage_a, stage_b, phase_a, phase_b))
        status, assignment, detail = solve(problem, ii)
        assert status == (SAT if feasible else UNSAT), (ii, detail, feasible)


def test_fixed_versions_survive_sat():
    problem = _pipeline(2, buffers=(Buffer("tile", 64, space="smem", owner="aiv0", fixed=3),),
                        reuses=(Reuse("tile", last_reader="vf", next_writer="load", delta=0, read_latency=1),),
                        capacities=(Capacity("smem", "aiv0", 4096),))
    status, assignment, detail = solve(problem, 2)
    assert status == SAT, detail
    assert assignment.versions["tile"] == 3
    assert verify_assignment(problem, assignment) == []


def test_authored_stages_are_kept_absolute():
    problem = Problem(tasks=(Task("a", (V_USE,), latency=1, stage=2),
                             Task("b", (V_USE,), latency=1, stage=3)),
                      edges=(Edge("a", "b", distance=0, latency=1),),
                      max_stage=3, manual=True)
    status, assignment, detail = solve(problem, 4)
    assert status == SAT, detail
    assert assignment.stages == {"a": 2, "b": 3}
    assert assignment.starts["a"] == 4 * 2 + assignment.phases["a"]


def test_no_offset_refuses_authored_stage():
    problem = Problem(tasks=(Task("a", (V_USE,), latency=1, stage=2),),
                      max_stage=3, manual=False)
    status, assignment, detail = solve(problem, 2)
    assert status == UNSAT and assignment is None
    assert detail.startswith("NO_OFFSET_AUTHORED_STAGE")


def test_one_pipe_two_tasks_cannot_share_ii_one():
    problem = Problem(**_campaign(1))
    status, _assignment, detail = solve(problem, 1)
    assert status == UNSAT, detail
    status, assignment, detail = solve(problem, 2)
    assert status == SAT, detail
    assert verify_assignment(problem, assignment) == []
    assert lower_bound_ii(problem) == 2


def test_capacity_is_a_refusal_not_a_backfill():
    problem = _pipeline(2, buffers=(Buffer("tile", 64, space="smem", owner="aiv0", fixed=2),),
                        capacities=(Capacity("smem", "aiv0", 127),))
    status, _assignment, detail = solve(problem, 2)
    assert status == UNSAT, detail
    problem = _pipeline(2, buffers=(Buffer("tile", 64, space="smem", owner="aiv0", fixed=2),),
                        capacities=(Capacity("smem", "aiv0", 128),))
    status, assignment, detail = solve(problem, 2)
    assert status == SAT, detail
    assert verify_assignment(problem, assignment) == []


def test_postcheck_catches_a_corrupted_assignment():
    problem = _pipeline(2)
    status, assignment, detail = solve(problem, 2)
    assert status == SAT, detail
    from dataclasses import replace

    corrupted = replace(assignment, stages={**assignment.stages, "load": assignment.stages["load"] + 1})
    problems = verify_assignment(problem, corrupted)
    assert any(item.startswith("AUTHORED_STAGE_CHANGED") or item.startswith("START_FORMULA") for item in problems)
    bad_versions = replace(assignment, versions={**assignment.versions})
    problem = _pipeline(2, buffers=(Buffer("tile", 64, space="smem", owner="aiv0", fixed=3),),
                        reuses=(Reuse("tile", last_reader="vf", next_writer="load", delta=0, read_latency=1),),
                        capacities=(Capacity("smem", "aiv0", 4096),))
    status, assignment, detail = solve(problem, 2)
    assert status == SAT, detail
    corrupted = replace(assignment, versions={"tile": 2})
    assert any(item.startswith("FIXED_VERSIONS_CHANGED") for item in verify_assignment(problem, corrupted))

def test_single_task_quantity_above_interval_is_unsat():
    """One task reserving more slots than an interval cannot fit in it."""
    problem = Problem(tasks=(Task("wide", (("aiv0", "aiv", "vector", 3),), latency=1),), max_stage=0)
    status, _assignment, detail = solve(problem, 1)
    assert status == UNSAT and detail.startswith("TASK_EXCEEDS_INTERVAL")
    status, assignment, detail = solve(problem, 3)
    assert status == SAT, detail
    assert verify_assignment(problem, assignment) == []


def test_manual_unannotated_sibling_defaults_to_zero():
    """A manual list pins the unannotated sibling at stage 0; it is not free."""
    problem = Problem(tasks=(Task("a", (("aiv0", "aiv", "vector", 1),), latency=1, stage=1),
                             Task("b", (("aiv0", "aiv", "mte2", 1),), latency=1)),
                      edges=(Edge("b", "a", distance=0, latency=1),),
                      max_stage=1, manual=True)
    status, assignment, detail = solve(problem, 1)
    assert status == SAT, detail
    assert assignment.stages == {"a": 1, "b": 0}
    assert verify_assignment(problem, assignment) == []


def test_issue_order_bounds_the_emission_point():
    """An author issue edge may not be satisfied by putting the later task earlier."""
    from dataclasses import replace as _replace

    problem = Problem(tasks=(Task("a", (("aiv0", "aiv", "vector", 1),), latency=1),
                             Task("b", (("aiv0", "aiv", "mte2", 1),), latency=1)),
                      edges=(Edge("b", "a", distance=0, latency=1),),
                      max_stage=1)
    status, assignment, detail = solve(problem, 2)
    assert status == SAT, detail
    assert verify_assignment(problem, assignment) == []
    ordered = _replace(problem, issue_after=(("a", "b"),))
    status, _assignment, detail = solve(ordered, 2)
    assert status == UNSAT, detail


def test_a_space_without_a_capacity_fact_is_unconstrained():
    """Capacity constraints exist only for a space that publishes a usable size."""
    problem = _pipeline(2, buffers=(Buffer("tile", 64, space="smem", owner="aiv0", fixed=1),))
    status, assignment, detail = solve(problem, 2)
    assert status == SAT, detail
    assert verify_assignment(problem, assignment) == []


def test_postcheck_rejects_foreign_and_extra_mappings():
    problem = _pipeline(2)
    status, assignment, detail = solve(problem, 2)
    assert status == SAT, detail
    from dataclasses import replace

    polluted = replace(assignment, stages={**assignment.stages, "ghost": 0})
    assert any(item.startswith("FOREIGN_TASK_STAGES") for item in verify_assignment(problem, polluted))
    foreign_versions = replace(assignment, versions={**assignment.versions, "ghostbuf": 1})
    assert any(item.startswith("FOREIGN_BUFFER_VERSION") for item in verify_assignment(problem, foreign_versions))



def test_issue_order_follows_phase_before_name():
    """Later phases issue later: the printed order is physical, not alphabetical."""
    problem = Problem(tasks=(Task("a", (("aiv0", "aiv", "vector", 1),), latency=1),
                             Task("b", (("aiv0", "aiv", "mte2", 1),), latency=1)),
                      edges=(Edge("b", "a", distance=0, latency=1),),
                      max_stage=0)
    status, assignment, detail = solve(problem, 2)
    assert status == SAT, detail
    assert assignment.phases["a"] > assignment.phases["b"]
    assert assignment.issue_order == ("b", "a")
    assert verify_assignment(problem, assignment) == []


def test_offset_matches_independent_ring_enumeration():
    from itertools import product
    from dataclasses import replace

    # Same-task reservations are included: independently mark occupied cycle slots.
    cases = [
        ((('aiv', 'aiv', 'v', 1, 1),), (('aiv', 'aiv', 'v', 1, 0),)),
        ((('aiv', 'aiv', 'v', 2, 3),), (('aiv', 'aiv', 'v', 1, 0),)),
        ((('aiv', 'aiv', 'v', 1, 0), ('aiv', 'aiv', 'v', 1, 2)), ()),
    ]
    for uses_a, uses_b in cases:
        problem = Problem(tasks=(Task('a', uses_a), Task('b', uses_b)), max_stage=0)
        for ii in range(1, 6):
            feasible = []
            for a, b in product(range(ii), repeat=2):
                occupied = set()
                good = True
                for phase, uses in ((a, uses_a), (b, uses_b)):
                    for owner, core, pipe, duration, offset in uses:
                        slots = {(phase + offset + j) % ii for j in range(duration)}
                        if len(slots) != duration or occupied.intersection(slots):
                            good = False
                        occupied.update(slots)
                if good:
                    feasible.append((a, b))
            status, assignment, detail = solve(problem, ii)
            assert status == (SAT if feasible else UNSAT), (ii, uses_a, uses_b, detail)
            if assignment is not None:
                assert (assignment.phases['a'], assignment.phases['b']) in feasible
                assert verify_assignment(problem, assignment) == []

    # Starts differ but reservations coincide after applying offset.
    problem = Problem(tasks=(Task('a', (('x', 'aiv', 'v', 1, 1),)),
                             Task('b', (('x', 'aiv', 'v', 1, 0),))), max_stage=0)
    status, assignment, _ = solve(problem, 2)
    assert status == SAT
    bad = replace(assignment, starts={'a': 0, 'b': 1}, phases={'a': 0, 'b': 1},
                  issue_order=('a', 'b'))
    assert any(x.startswith('RESOURCE_OVERLAP') for x in verify_assignment(problem, bad))


def test_real_region_resource_offsets_are_projected_and_postchecked():
    from dataclasses import replace
    from tilefoundry.analysis import analyze
    from tilefoundry.cli.source import checked_program
    from tilefoundry.schedule import finalize
    from tilefoundry_ascend.plan.numeric import to_schedule_problem
    from tilefoundry_ascend.plan.problem import Dependence, analyze_native_problem
    from tilefoundry_ascend.plan.resolved import ResolveError, check_numeric_region
    from tilefoundry_ascend.plan.units import ResourceUse, build_native_unit_tree
    from tilefoundry_ascend.solver import Assignment, solve_region

    source = 'targets/ascend/tests_native_mainline/fixtures/authors/minimal_aiv.py:MinimalAivPipeline'
    program = checked_program(source, None, command='test resource offset')
    analyzed = analyze(program.module, program.entry, analysis=('compute-cost', 'memory'))
    module = finalize(analyzed, authored=program.authored)
    tree = build_native_unit_tree(module, authored=program.authored)
    problem = analyze_native_problem(tree, None, None, module.entry_function().target)
    region = next(r for r in problem.regions if r.region_ref.induction_var_ref is not None)
    a, b = region.tasks
    resource = a.uses[0].resource
    a = replace(a, uses=(ResourceUse(resource, 1, 1),), latency=2)
    b = replace(b, uses=(ResourceUse(resource, 1, 0),), latency=1)
    # Force a at phase 0 and b at phase 1 modulo 2. The reservations coincide.
    edges = (Dependence(a.task_id, b.task_id, 0, 1, 'RAW', ()),
             Dependence(b.task_id, a.task_id, 1, 1, 'RAW', ()))
    region = replace(region, tasks=(a, b), edges=edges, reuse=(),
                     fixed_ii=2, ii_min=2, ii_max=2, issue_after=(), manual_pipe_order=())
    numeric = to_schedule_problem(region)
    assert numeric.core.tasks[0].resources[0][4] == 1
    result = solve_region(numeric)
    assert result.status == UNSAT
    forged = Assignment(status='SAT', ii=2, starts={a.task_id: 0, b.task_id: 1},
                        stages={a.task_id: 0, b.task_id: 0},
                        phases={a.task_id: 0, b.task_id: 1},
                        versions={buf.buffer_id: getattr(buf.domain, 'value', 1)
                                  for buf in region.buffers},
                        issue_order=(a.task_id, b.task_id))
    with pytest.raises(ResolveError) as caught:
        check_numeric_region(region, forged)
    assert caught.value.code == 'PLAN_RESOURCE_OVERLAP'
