"""The numeric projection: a real region problem solved and resolved end to end."""
from __future__ import annotations

from dataclasses import replace

import pytest

from tilefoundry.analysis import analyze
from tilefoundry.cli.source import checked_program
from tilefoundry.schedule import finalize
from tilefoundry_ascend.plan.numeric import to_core, to_schedule_problem
from tilefoundry_ascend.plan.problem import analyze_native_problem
from tilefoundry_ascend.plan.resolved import ResolveError, check_numeric_region, resolve_schedule
from tilefoundry_ascend.plan.units import build_native_unit_tree
from tilefoundry_ascend.solver import solve_region

AUTHOR = "targets/ascend/tests_native_mainline/fixtures/authors/minimal_aiv.py:MinimalAivPipeline"


def _real_problem():
    program = checked_program(AUTHOR, None, command="test numeric")
    analyzed = analyze(program.module, program.entry, analysis=("compute-cost", "memory"))
    module = finalize(analyzed, authored=program.authored)
    tree = build_native_unit_tree(module, authored=program.authored)
    return analyze_native_problem(tree, None, None, module.entry_function().target)


def test_the_projection_carries_the_real_numbers():
    problem = _real_problem()
    region = problem.regions[0]
    core = to_core(region)
    assert len(core.tasks) == len(region.tasks)
    assert {task.name for task in core.tasks} == {task.task_id for task in region.tasks}
    assert len(core.edges) == len(region.edges)
    assert {buffer.name for buffer in core.buffers} == {buffer.buffer_id for buffer in region.buffers}
    assert core.max_stage == region.max_stage


def test_a_real_region_solves_and_resolves():
    problem = _real_problem()
    region = next(item for item in problem.regions if item.tasks)
    assignment = solve_region(to_schedule_problem(region))
    # The S0 author program is schedulable; a regression here must fail, not skip.
    assert assignment.status == "SAT", assignment.detail
    single = replace(problem, regions=(region,))
    assert check_numeric_region(region, assignment) == assignment.ii
    for task in region.tasks:
        assert assignment.starts[task.task_id] == assignment.ii * assignment.stages[task.task_id] + assignment.phases[task.task_id]
        assert 0 <= assignment.phases[task.task_id] < assignment.ii
    for buffer in region.buffers:
        assert assignment.versions[buffer.buffer_id] >= 1
    with pytest.raises(ResolveError, match="PLAN_DEPENDENCE_PROOF_UNAVAILABLE"):
        resolve_schedule(single, assignment)


def test_a_declared_domain_is_reported_not_widened():
    problem = _real_problem()
    region = next(item for item in problem.regions if item.tasks)
    narrowed = replace(region, fixed_ii=1, ii_min=1, ii_max=1)
    assignment = solve_region(to_schedule_problem(narrowed))
    assert assignment.status in ("SAT", "UNSAT")
    assert assignment.optimality.ii_checks in ((), ((1, "SAT"),), ((1, "UNSAT"),))


def test_the_whole_program_resolves_one_assignment_per_region():
    """A region owns its II: the whole problem resolves from a per-region answer."""
    from tilefoundry_ascend.plan.resolved import ResolveError

    problem = _real_problem()
    answers = {}
    for region in problem.regions:
        assignment = solve_region(to_schedule_problem(region))
        assert assignment.status == "SAT", assignment.detail
        answers[region.region_ref.region_id] = assignment
    for region in problem.regions:
        answer = answers[region.region_ref.region_id]
        assert check_numeric_region(region, answer) == answer.ii
    with pytest.raises(ResolveError, match="PLAN_DEPENDENCE_PROOF_UNAVAILABLE"):
        resolve_schedule(problem, answers)
    # One answer for two regions with different intervals is refused, not merged.
    try:
        resolve_schedule(problem, answers[problem.regions[0].region_ref.region_id])
    except ResolveError as error:
        assert error.code == "PLAN_ASSIGNMENT_PER_REGION"
    else:  # pragma: no cover - merging two intervals must never be accepted
        raise AssertionError("one assignment for two regions must be refused")

def test_reuse_projects_real_accesses_and_enforces_every_edge():
    from tilefoundry_ascend.plan.problem import Reuse

    problem = _real_problem()
    region = next(r for r in problem.regions if r.region_ref.induction_var_ref is not None)
    pair = next((writer, reader, buffer)
                for buffer in region.buffers
                for writer in region.accesses
                for reader in region.accesses
                if writer.root_ref is buffer.root_ref
                and reader.root_ref is buffer.root_ref
                and writer.mode in ('WRITE', 'READ_WRITE')
                and reader.mode in ('READ', 'READ_WRITE')
                and writer.task_id != reader.task_id)
    writer, reader, buffer = pair
    # These two obligations STATE their own numbers (delay_fact=None means the
    # record's own delay is the declared value): they probe the projection and
    # the constraint, and are not the facts the problem derives for a program.
    reuse1 = Reuse(reader.access_id, writer.access_id, buffer.buffer_id, 0, 0)
    reuse2 = Reuse(reader.access_id, writer.access_id, buffer.buffer_id, 1000, 1)

    # What the region's own edge now carries: a sourced model lower bound, never
    # a literal 0. Asserting the state here keeps a silent change visible.
    edge = region.edges[0]
    assert edge.delay == 64, (edge.delay, edge.delay_fact)
    assert edge.delay_fact.state == "DECLARED"
    assert edge.delay_fact.basis == "OCCUPANCY_FALLBACK"
    assert edge.delay_fact.unit == "MODEL_TICK"
    # The locator is STRUCTURED (plan/delays.py ModelSource): a prose substring
    # test here would break on a wording change and would let a reworded sentence
    # masquerade as a changed fact. Assert the locator fields instead.
    source = edge.delay_fact.source
    assert (source.module, source.symbol, source.field) == ("ir/isa.py", "gm_to_ub",
                                                            "latency_model"), source
    assert int(source.value) == edge.delay, (source.value, edge.delay)

    # The region's own model answer under that sourced delay. Before the delay
    # carried a source this region solved at ii=64; the 64-tick lower bound plus
    # a same-window consumer needs one more tick, so the natural answer moved to
    # 65. That is the delay entering the constraint, not a regression.
    natural = solve_region(to_schedule_problem(region))
    assert natural.status == "SAT", natural.detail
    ii = int(natural.ii)
    # Both tasks sit in stage 0 (enable_offset is False), so their starts differ
    # by less than II; a 64-tick lower bound therefore forces II past 64.
    assert ii == edge.delay + 1
    tightened = replace(region, fixed_ii=64, ii_min=64, ii_max=64)
    assert solve_region(to_schedule_problem(tightened)).status == "UNSAT"

    changed = replace(region, reuse=(reuse1, reuse2), fixed_ii=ii, ii_min=ii, ii_max=ii)
    projected = to_core(changed)
    assert len(projected.reuses) == 2
    assert [(r.last_reader, r.next_writer, r.delta, r.read_latency)
            for r in projected.reuses] == [
                (reader.task_id, writer.task_id, 0, 0),
                (reader.task_id, writer.task_id, 1, 1000)]
    # The high-delay second obligation cannot hold even with maximal V and phase.
    assert solve_region(to_schedule_problem(changed)).status == "UNSAT"
    # Same II, only the delay of that obligation changed: SAT. The delay value
    # demonstrably drives the constraint rather than being ignored.
    feasible = replace(changed, reuse=(reuse1,))
    assert solve_region(to_schedule_problem(feasible)).status == "SAT"

    # An obligation that states no number is refused at the projection; it is
    # never defaulted to 0 (or to the core's default of 1).
    unsourced = replace(region, reuse=(Reuse(reader.access_id, writer.access_id,
                                             buffer.buffer_id, None, 0),))
    with pytest.raises(Exception) as caught:
        to_core(unsourced)
    assert caught.value.code == "DELAY_UNSOURCED"


def test_reuse_projection_refuses_missing_ambiguous_and_foreign_endpoints():
    from tilefoundry_ascend.plan.problem import Reuse
    from tilefoundry.vendor.tilelang_z3.core import SolverError

    region = next(r for r in _real_problem().regions if r.region_ref.induction_var_ref is not None)
    access = region.accesses[0]
    buffer = region.buffers[0]
    reuse = Reuse(access.access_id, access.access_id, buffer.buffer_id, 0, 0)
    for changed, code in (
        (replace(region, reuse=(replace(reuse, last_reader='missing'),)),
         'PROJECTION_REUSE_ACCESS_UNKNOWN'),
        (replace(region, accesses=region.accesses + (access,), reuse=(reuse,)),
         'PROJECTION_ACCESS_DUPLICATE'),
        (replace(region, accesses=(replace(access, task_id='missing'),) + region.accesses[1:],
                 reuse=(reuse,)), 'PROJECTION_REUSE_TASK_UNKNOWN'),
    ):
        with pytest.raises(SolverError) as caught:
            to_core(changed)
        assert caught.value.code == code


# ------------------------------------- R7 item 4: reuse pairing must be derived
def _loop_problem_region():
    from tilefoundry.analysis import analyze
    from tilefoundry.cli.source import checked_program
    from tilefoundry.schedule import finalize
    from tilefoundry_ascend.plan.problem import analyze_native_problem
    from tilefoundry_ascend.plan.units import build_native_unit_tree

    program = checked_program(AUTHOR, None, command="test reuse pairing")
    analyzed = analyze(program.module, program.entry,
                       analysis=("compute-cost", "memory"))
    module = finalize(analyzed, authored=program.authored)
    problem = analyze_native_problem(
        build_native_unit_tree(module, authored=program.authored), None, None,
        module.entry_function().target)
    return next(r for r in problem.regions
                if r.region_ref.induction_var_ref is not None)


def test_a_forged_reuse_pairing_is_refused_at_the_projection():
    """R7 item 4: a reuse endpoint must be a read/write OF THAT BUFFER."""
    from tilefoundry_ascend.plan.numeric import to_core
    from tilefoundry_ascend.plan.problem import Reuse

    region = _loop_problem_region()
    # a buffer the region actually writes once and reads once
    tile = next(b for b in region.buffers
                if any(a.root_ref is b.root_ref and a.mode in ("WRITE", "READ_WRITE")
                       for a in region.accesses)
                and any(a.root_ref is b.root_ref and a.mode in ("READ", "READ_WRITE")
                        for a in region.accesses))
    reader = next(a for a in region.accesses
                  if a.mode in ("READ", "READ_WRITE") and a.root_ref is tile.root_ref)
    writer = next(a for a in region.accesses
                  if a.mode in ("WRITE", "READ_WRITE")
                  and a.root_ref is not tile.root_ref)
    forged = Reuse(last_reader=reader.access_id, next_writer=writer.access_id,
                   buffer=tile.buffer_id, delay=0, distance_offset=0)
    with pytest.raises(Exception) as caught:
        to_core(replace(region, reuse=(forged,)))
    assert caught.value.code == "PROJECTION_REUSE_UNPAIRED"


def test_a_reuse_pair_inside_one_task_is_refused():
    from tilefoundry_ascend.plan.numeric import to_core
    from tilefoundry_ascend.plan.problem import Reuse

    region = _loop_problem_region()
    acc = next(b for b in region.buffers
               if any(a.mode == "READ_WRITE" and a.root_ref is b.root_ref
                      for a in region.accesses))
    write = next(a for a in region.accesses if a.mode == "READ_WRITE"
                 and a.root_ref is acc.root_ref)
    read = next(a for a in region.accesses if a.mode == "READ"
                and a.root_ref is acc.root_ref and a is not write)
    assert read.task_id == write.task_id
    same_task = Reuse(last_reader=read.access_id, next_writer=write.access_id,
                      buffer=acc.buffer_id, delay=0, distance_offset=0)
    with pytest.raises(Exception) as caught:
        to_core(replace(region, reuse=(same_task,)))
    assert caught.value.code == "PROJECTION_REUSE_SAME_TASK"


def test_a_reuse_with_no_stated_number_never_defaults_to_zero_or_one():
    from tilefoundry_ascend.plan.numeric import to_core
    from tilefoundry_ascend.plan.problem import Reuse

    region = _loop_problem_region()
    tile = next(b for b in region.buffers
                if any(a.root_ref is b.root_ref and a.mode in ("WRITE", "READ_WRITE")
                       for a in region.accesses)
                and any(a.root_ref is b.root_ref and a.mode in ("READ", "READ_WRITE")
                        for a in region.accesses))
    write = next(a for a in region.accesses if a.mode in ("WRITE", "READ_WRITE")
                 and a.root_ref is tile.root_ref)
    read = next(a for a in region.accesses if a.mode in ("READ", "READ_WRITE")
                and a.root_ref is tile.root_ref)
    unsourced = Reuse(last_reader=read.access_id, next_writer=write.access_id,
                      buffer=tile.buffer_id, delay=None, distance_offset=0)
    with pytest.raises(Exception) as caught:
        to_core(replace(region, reuse=(unsourced,)))
    assert caught.value.code == "DELAY_UNSOURCED"


def test_distance_is_derived_and_an_unprovable_generation_refuses():
    """R7 item 4: no literal distance 0 is written into an edge."""
    import dataclasses

    from tilefoundry.ir.core import Constant, Var
    from tilefoundry.ir.types import DType, TensorType
    from tilefoundry_ascend.plan.problem import (ProblemError, _generation_distance)

    region = _loop_problem_region()
    left = next(a for a in region.accesses if a.mode in ("WRITE", "READ_WRITE"))
    right = next(a for a in region.accesses if a.mode in ("READ", "READ_WRITE")
                 and a.root_ref is left.root_ref and a is not left)
    assert _generation_distance(left, right) == 0

    iv = region.region_ref.induction_var_ref
    assert isinstance(iv, Var), iv
    later = dataclasses.replace(right, generation=iv)
    earlier = dataclasses.replace(
        right, generation=Constant(type=TensorType.scalar(DType.i64), value=7))
    with pytest.raises(ProblemError) as caught:
        _generation_distance(earlier, later)
    assert caught.value.code == "PROBLEM_DEPENDENCE_DISTANCE_UNSUPPORTED"
