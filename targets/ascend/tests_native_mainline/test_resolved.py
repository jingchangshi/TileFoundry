"""resolve_schedule: the one plan, and the refusals that keep it single."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tilefoundry_ascend.plan.resolved import (ResolveError, check_numeric_region,
                                               resolve_schedule,
                                               verify_plan_bindings)


def _problem(*, complete=True, tasks=None, buffers=None, max_stage=1):
    region_ref = SimpleNamespace(region_id="r0", lower_expr="lo", stop_expr="stop", step_expr="step")
    tasks = tasks if tasks is not None else (
        SimpleNamespace(task_id="load", unit_ref="unit-load", authored_stage=None),
        SimpleNamespace(task_id="vf", unit_ref="unit-vf", authored_stage=1),
    )
    buffers = buffers if buffers is not None else (
        SimpleNamespace(buffer_id="tile", root_ref="root-tile", domain="Auto(1,2)", slot_bytes=8),
    )
    coverage = SimpleNamespace(complete=complete)
    return SimpleNamespace(
        source_snapshot=SimpleNamespace(identity="sha256:source"),
        target_identity="sha256:target",
        model_version="sha256:model",
        coverage=coverage,
        regions=(SimpleNamespace(region_ref=region_ref, tasks=tasks, buffers=buffers,
                                 edges=(), reuse=(), capacities=(), manual_pipe_order=(),
                                 enable_offset=True, max_stage=max_stage, fixed_ii=None,
                                 ii_min=1, ii_max=2, owners=(), pipes=(), resources=(),
                                 accesses=(), effects=(), issue_after=()),),
    )


def _assignment(**overrides):
    values = dict(
        status="SAT", ii=2, detail="SAT",
        starts={"load": 0, "vf": 3}, stages={"load": 0, "vf": 1},
        phases={"load": 0, "vf": 1}, versions={"tile": 2},
        issue_order=("load", "vf"), optimality=SimpleNamespace(ii_optimal_within_bounds=True),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_numeric_assignment_does_not_certify_a_plan():
    problem = _problem()
    assignment = _assignment()
    assert check_numeric_region(problem.regions[0], assignment) == 2
    with pytest.raises(ResolveError, match="PLAN_SOURCE_EVIDENCE_MISSING"):
        resolve_schedule(problem, assignment)


def test_repeated_numeric_check_is_deterministic():
    problem, assignment = _problem(), _assignment()
    assert check_numeric_region(problem.regions[0], assignment) == 2
    assert check_numeric_region(problem.regions[0], assignment) == 2


def test_alternative_consistent_phases_pass_numeric_check():
    changed = _assignment(phases={"load": 1, "vf": 0}, starts={"load": 1, "vf": 2},
                          issue_order=("vf", "load"))
    assert check_numeric_region(_problem().regions[0], changed) == 2


def test_an_ii_outside_the_declared_domain_is_refused():
    """II is an author fact: the plan may not widen it after the solver answered."""
    with pytest.raises(ResolveError) as caught:
        check_numeric_region(_problem().regions[0], _assignment(ii=256))
    assert caught.value.code == "PLAN_II_OUT_OF_DOMAIN"


def test_a_same_phase_issue_edge_binds_the_emission_order():
    """Equal phases are ordered by the author's hard issue edge, not by the order given."""
    region = _problem()
    region.regions[0].issue_after = (("load", "vf"),)
    same_phase = _assignment(phases={"load": 0, "vf": 0}, starts={"load": 0, "vf": 2},
                             issue_order=("vf", "load"))
    with pytest.raises(ResolveError) as caught:
        check_numeric_region(region.regions[0], same_phase)
    assert caught.value.code == "PLAN_ISSUE_ORDER_VIOLATED"


def test_no_plan_without_sat():
    for status in ("UNSAT", "UNKNOWN"):
        with pytest.raises(ResolveError) as caught:
            resolve_schedule(_problem(), _assignment(status=status))
        assert caught.value.code == "PLAN_NOT_SAT"


def test_no_plan_without_complete_coverage():
    with pytest.raises(ResolveError) as caught:
        resolve_schedule(_problem(complete=False), _assignment())
    assert caught.value.code == "PLAN_COVERAGE_INCOMPLETE"


def test_no_plan_when_a_task_or_buffer_is_unresolved():
    with pytest.raises(ResolveError) as caught:
        check_numeric_region(_problem().regions[0], _assignment(stages={"load": 0}))
    assert caught.value.code == "PLAN_TASK_UNRESOLVED"
    with pytest.raises(ResolveError) as caught:
        check_numeric_region(_problem().regions[0], _assignment(versions={}))
    assert caught.value.code == "PLAN_BUFFER_UNRESOLVED"


def test_an_authored_stage_cannot_move():
    with pytest.raises(ResolveError) as caught:
        check_numeric_region(_problem(max_stage=2).regions[0], _assignment(stages={"load": 0, "vf": 2}, starts={"load": 0, "vf": 5}))
    assert caught.value.code == "PLAN_AUTHORED_STAGE_CHANGED"


def test_a_forged_sat_assignment_is_refused():
    """A status byte is not proof: the plan re-derives every number."""
    forged_phase = _assignment(phases={"load": 0, "vf": 999}, starts={"load": 0, "vf": 999})
    with pytest.raises(ResolveError) as caught:
        check_numeric_region(_problem().regions[0], forged_phase)
    assert caught.value.code == "PLAN_PHASE_OUT_OF_RANGE"
    forged_versions = _assignment(versions={"tile": 0})
    with pytest.raises(ResolveError) as caught:
        check_numeric_region(_problem().regions[0], forged_versions)
    assert caught.value.code == "PLAN_VERSION_OUT_OF_DOMAIN"
    forged_order = _assignment(issue_order=())
    with pytest.raises(ResolveError) as caught:
        check_numeric_region(_problem().regions[0], forged_order)
    assert caught.value.code == "PLAN_ISSUE_ORDER_INVALID"
    forged_start = _assignment(starts={"load": 0, "vf": 5})
    with pytest.raises(ResolveError) as caught:
        check_numeric_region(_problem().regions[0], forged_start)
    assert caught.value.code == "PLAN_START_INCONSISTENT"

def test_reuse_checks_task_starts_via_access_ids():
    from tilefoundry_ascend.plan.problem import Reuse

    problem = _problem()
    region = problem.regions[0]
    region.accesses = (SimpleNamespace(access_id="reader", task_id="vf"),
                       SimpleNamespace(access_id="writer", task_id="load"))
    # versions=2, ii=2: the writer lands at 0+4=4 and the reader at 3+delay.
    region.reuse = (Reuse(last_reader="reader", next_writer="writer", buffer="tile",
                          delay=2, distance_offset=0),)
    with pytest.raises(ResolveError, match="PLAN_REUSE_VIOLATED"):
        check_numeric_region(problem.regions[0], _assignment())
    region.reuse = (Reuse(last_reader="missing", next_writer="writer", buffer="tile",
                          delay=0, distance_offset=0),)
    with pytest.raises(ResolveError, match="PLAN_REUSE_ACCESS_UNKNOWN"):
        check_numeric_region(problem.regions[0], _assignment())
    # A reuse that states no number is refused; it is never read as zero.
    region.reuse = (Reuse(last_reader="reader", next_writer="writer", buffer="tile",
                          delay=None, distance_offset=0),)
    with pytest.raises(ResolveError, match="DELAY_UNSOURCED"):
        check_numeric_region(problem.regions[0], _assignment())


# ------------------------------------------------- R7 conditional dependence gate
from dataclasses import replace as _replace

import pytest as _pytest  # noqa: F401  (module already imports pytest)

INDEPENDENT = ("targets/ascend/tests_native_mainline/fixtures/authors/"
               "independent_ub.py:IndependentUB")
MULTI_REGION = ("targets/ascend/tests_native_mainline/fixtures/authors/"
                "minimal_aiv.py:MinimalAivPipeline")


def _independent_problem():
    from tilefoundry.analysis import analyze
    from tilefoundry.cli.source import checked_program
    from tilefoundry.schedule import finalize
    from tilefoundry_ascend.plan.problem import analyze_native_problem
    from tilefoundry_ascend.plan.units import build_native_unit_tree

    program = checked_program(INDEPENDENT, None, command="test gate")
    analyzed = analyze(program.module, program.entry,
                       analysis=("compute-cost", "memory"))
    module = finalize(analyzed, authored=program.authored)
    tree = build_native_unit_tree(module, authored=program.authored)
    problem = analyze_native_problem(tree, None, None, module.entry_function().target)
    return module, problem


def _assignments(problem):
    from tilefoundry_ascend.plan.numeric import to_schedule_problem
    from tilefoundry_ascend.solver import solve_region

    answers = {region.region_ref.region_id: solve_region(to_schedule_problem(region))
               for region in problem.regions}
    assert all(answer.status == "SAT" for answer in answers.values()), answers
    return answers


def test_a_proven_program_now_resolves_a_real_plan():
    """R7: complete evidence BUILDS the plan; refusing it was a dead end."""
    module, problem = _independent_problem()
    plan = resolve_schedule(problem, _assignments(problem))
    assert plan.feasibility == "SAT"
    assert len(plan.regions) == 1
    region = plan.regions[0]
    answer = _assignments(problem)[region.region_ref.region_id]
    assert region.ii == int(answer.ii)
    for task in region.tasks:
        assert task.start == region.ii * task.resolved_stage + task.phase
        assert 0 <= task.phase < region.ii
        # one straight-line domain: nothing is offset
        assert task.resolved_stage == 0 and region.origin == 0
        assert region.max_shift == 0
    for buffer in region.buffers:
        assert buffer.resolved_versions == 1
    # the plan binds the ACTUAL problem, its content anchors and the proof
    assert plan.problem_ref is problem
    assert plan.problem_identity == problem.problem_identity
    assert plan.problem_canonical_bytes == problem.problem_canonical_bytes
    assert plan.target_ref is module.entry_function().target
    assert plan.dependence_proof.complete()
    assert len(plan.dependence_proof.edges) == 3
    assert plan.plan_canonical_bytes.startswith(b"{")


def test_the_gate_refuses_a_coordinated_edge_forgery_by_rereading_the_ir():
    """Rewriting the problem's edges (and their ids) is not enough: the proof
    re-reads the real TIR and disagrees."""
    from tilefoundry_ascend.plan.source_audit import SourceAuditError

    _, problem = _independent_problem()
    region = problem.regions[0]
    flipped = _replace(region.edges[0], producer=region.edges[0].consumer,
                       consumer=region.edges[0].producer)
    forged = _replace(problem,
                      regions=(_replace(region, edges=(flipped, *region.edges[1:])),),
                      problem_identity="sha256:self-reported",
                      problem_canonical_bytes=b'{"lie":true}')
    answers = _assignments(problem)
    with _pytest.raises(ResolveError) as caught:
        resolve_schedule(forged, answers)
    assert caught.value.code == "SOURCE_DEPENDENCE_MISMATCH"


def test_the_gate_refuses_a_swapped_root_or_a_foreign_access():
    from tilefoundry_ascend.plan.source_audit import SourceAuditError

    _, problem = _independent_problem()
    region = problem.regions[0]
    answers = _assignments(problem)

    other_root = region.buffers[-1].root_ref
    target = region.accesses[1]
    swapped = tuple(_replace(access, root_ref=other_root)
                    if access is target else access for access in region.accesses)
    altered = _replace(problem, regions=(_replace(region, accesses=swapped),))
    with _pytest.raises(ResolveError) as caught:
        resolve_schedule(altered, answers)
    assert caught.value.code == "SOURCE_ACCESS_ROOT_MISMATCH"

    extra = _replace(region.accesses[0], access_id="access:invented")
    altered = _replace(problem, regions=(_replace(region, accesses=region.accesses + (extra,)),))
    with _pytest.raises(ResolveError) as caught:
        resolve_schedule(altered, answers)
    assert caught.value.code == "SOURCE_ACCESS_COVERAGE"


def test_a_multi_region_program_is_refused_with_every_missing_item_named():
    from tilefoundry.analysis import analyze
    from tilefoundry.cli.source import checked_program
    from tilefoundry.schedule import finalize
    from tilefoundry_ascend.plan.numeric import to_schedule_problem
    from tilefoundry_ascend.plan.problem import analyze_native_problem
    from tilefoundry_ascend.plan.units import build_native_unit_tree
    from tilefoundry_ascend.solver import solve_region

    program = checked_program(MULTI_REGION, None, command="test multi region gate")
    analyzed = analyze(program.module, program.entry,
                       analysis=("compute-cost", "memory"))
    module = finalize(analyzed, authored=program.authored)
    problem = analyze_native_problem(
        build_native_unit_tree(module, authored=program.authored), None, None,
        module.entry_function().target)
    answers = {region.region_ref.region_id: solve_region(to_schedule_problem(region))
               for region in problem.regions}
    assert all(answer.status == "SAT" for answer in answers.values())
    with _pytest.raises(ResolveError) as caught:
        resolve_schedule(problem, answers)
    assert caught.value.code == "PLAN_DEPENDENCE_PROOF_UNAVAILABLE"
    detail = caught.value.detail
    # names the region, the loop, the carried value AND that nothing was re-read
    assert "region:1" in detail and "carried" in detail
    assert "no region's evidence was re-read here" in detail


def test_the_proof_content_is_plan_identity_not_a_status_string():
    from tilefoundry_ascend.plan.resolved import _plan_bytes
    from tilefoundry_ascend.plan.source_audit import DependenceProof

    _, problem = _independent_problem()
    plan = resolve_schedule(problem, _assignments(problem))
    complete = plan.dependence_proof
    empty = DependenceProof(region_id=complete.region_id, straight_line=True,
                            edges=(), reads=(), writes=(), buffers=(), reuse=(),
                            missing=())
    # A caller-typed "proof" with nothing missing is NOT complete: completeness
    # Completeness comes from the CONTENT being non-empty and self-consistent,
    # never from the derived flag; see
    # test_an_empty_proof_is_never_complete_even_with_the_derived_flag.
    assert not empty.derived
    assert not empty.complete()
    assert _plan_bytes(problem, plan.regions, empty) != plan.plan_canonical_bytes
    # and dropping one proven edge changes the bound content too
    one_edge = _replace(complete, edges=complete.edges[:1])
    assert _plan_bytes(problem, plan.regions, one_edge) != plan.plan_canonical_bytes


def test_the_gate_takes_no_injected_proof():
    """The gate reads the IR itself: resolve_schedule has no proof argument."""
    import inspect

    from tilefoundry_ascend.plan.resolved import resolve_schedule as resolve

    signature = inspect.signature(resolve)
    assert list(signature.parameters) == ["problem", "assignment"], signature
    _, problem = _independent_problem()
    plan = resolve(problem, _assignments(problem))
    assert plan.dependence_proof.derived is True
    # a hand-made record with an empty missing list reports itself incomplete,
    # so nothing that bypasses dependence_proof() can look like evidence
    from tilefoundry_ascend.plan.source_audit import DependenceProof

    fake = DependenceProof(region_id="region:0", straight_line=True, edges=(),
                           reads=(), writes=(), buffers=(), reuse=(), missing=())
    assert not fake.complete()


def test_editing_the_delay_table_moves_the_plan_identity():
    from tilefoundry_ascend.plan import delays
    from tilefoundry_ascend.plan.numeric import to_schedule_problem
    from tilefoundry_ascend.plan.problem import analyze_native_problem
    from tilefoundry_ascend.solver import solve_region

    _, problem = _independent_problem()
    before = resolve_schedule(problem, _assignments(problem)).plan_identity
    row = delays._OCCUPANCY["gm_to_ub"]
    delays._OCCUPANCY["gm_to_ub"] = _replace(row, ticks=row.ticks + 1)
    try:
        # A bigger bound also invalidates the OLD assignment: re-solving is the
        # honest way to get the plan that the edited table describes.
        edited = analyze_native_problem(problem.unit_tree, None, None,
                                       problem.target_ref)
        answers = {region.region_ref.region_id:
                   solve_region(to_schedule_problem(region))
                   for region in edited.regions}
        after = resolve_schedule(edited, answers).plan_identity
    finally:
        delays._OCCUPANCY["gm_to_ub"] = row
    assert before != after
    # and restoring the table restores the identity, so nothing cached leaked
    assert resolve_schedule(problem, _assignments(problem)).plan_identity == before


def test_a_stale_assignment_is_refused_when_the_table_grows():
    """The schedule itself is re-checked: an assignment solved under the old
    bound cannot be paired with a proof built from a larger one."""
    from tilefoundry_ascend.plan import delays
    from tilefoundry_ascend.plan.problem import analyze_native_problem

    _, problem = _independent_problem()
    answers = _assignments(problem)
    row = delays._OCCUPANCY["gm_to_ub"]
    delays._OCCUPANCY["gm_to_ub"] = _replace(row, ticks=row.ticks + 100)
    try:
        edited = analyze_native_problem(problem.unit_tree, None, None,
                                       problem.target_ref)
        with _pytest.raises(ResolveError) as caught:
            resolve_schedule(edited, answers)
    finally:
        delays._OCCUPANCY["gm_to_ub"] = row
    assert caught.value.code == "PLAN_EDGE_VIOLATED"

def test_plan_references_must_be_the_problems_own_objects():
    """Content equality and id strings are not binding: swap the objects."""
    from tilefoundry_ascend.plan.resolved import verify_plan_bindings

    _, problem = _independent_problem()
    plan = resolve_schedule(problem, _assignments(problem))
    verify_plan_bindings(plan, problem)  # the honest pairing passes

    region = problem.regions[0]
    other_root = region.buffers[-1].root_ref
    swapped = tuple(
        _replace(planned, buffers=tuple(
            _replace(buffer, buffer_ref=other_root)
            for buffer in planned.buffers))
        for planned in plan.regions)
    with _pytest.raises(ResolveError) as caught:
        verify_plan_bindings(_replace(plan, regions=swapped), problem)
    assert caught.value.code == "PLAN_BUFFER_BINDING_MISMATCH"

    # a foreign region object carrying the same id string is refused too
    foreign_ref = _replace(region.region_ref, region_id=region.region_ref.region_id)
    foreign = _replace(plan, regions=tuple(
        _replace(planned, region_ref=foreign_ref) for planned in plan.regions))
    with _pytest.raises(ResolveError) as caught:
        verify_plan_bindings(foreign, problem)
    assert caught.value.code == "PLAN_REGION_BINDING_MISMATCH"

    # dropping a planned task is a count failure, not a silent slice
    dropped = _replace(plan, regions=tuple(
        _replace(planned, tasks=planned.tasks[1:]) for planned in plan.regions))
    with _pytest.raises(ResolveError) as caught:
        verify_plan_bindings(dropped, problem)
    assert caught.value.code == "PLAN_TASK_BINDING_COUNT"

    # and the source map may not name a task the plan no longer carries
    stale_map = dict(plan.task_to_native_source)
    stale_map["unit:invented"] = region.tasks[0].unit_ref
    with _pytest.raises(ResolveError) as caught:
        verify_plan_bindings(_replace(plan, task_to_native_source=stale_map), problem)
    assert caught.value.code == "PLAN_SOURCE_MAP_STALE"


def test_plan_bytes_recompute_identically_and_ignore_object_addresses():
    """No id() may enter the bound bytes: a consumer in another process must be
    able to regenerate exactly the same value from the same content."""
    from tilefoundry_ascend.plan.resolved import _plan_bytes

    _, problem = _independent_problem()
    plan = resolve_schedule(problem, _assignments(problem))
    assert _plan_bytes(problem, plan.regions, plan.dependence_proof) \
        == plan.plan_canonical_bytes
    assert b"0x" not in plan.plan_canonical_bytes
    # re-running the whole normal entry over the same source gives the same bytes
    _, other = _independent_problem()
    other_plan = resolve_schedule(other, _assignments(other))
    assert _plan_bytes(other, other_plan.regions, other_plan.dependence_proof) \
        == plan.plan_canonical_bytes
    assert other_plan.plan_identity == plan.plan_identity


def test_an_empty_proof_is_never_complete_even_with_the_derived_flag():
    """The gap the lead measured: DP(derived=True, everything empty).

    Completeness is content, not a flag: nothing was read, nothing was written,
    no buffer was listed, so nothing is proven.
    """
    from tilefoundry_ascend.plan.source_audit import DependenceProof

    empty = DependenceProof(region_id="region:0", straight_line=True, edges=(),
                            reads=(), writes=(), buffers=(), reuse=(), missing=(),
                            derived=True)
    assert empty.derived is True and empty.missing == ()
    assert not empty.satisfied_by_content()
    assert not empty.complete()


def test_a_non_empty_forgery_is_rejected_by_its_own_content():
    """Fabricated rows must still describe a coherent program."""
    import dataclasses

    _, problem = _independent_problem()
    real = resolve_schedule(problem, _assignments(problem)).dependence_proof
    assert real.complete()

    # an edge that names a task no read or write ever saw
    lying_edge = dataclasses.replace(real.edges[0], consumer="unit:nope")
    forged = dataclasses.replace(real, edges=(lying_edge, *real.edges[1:]))
    assert not forged.satisfied_by_content() and not forged.complete()

    # an edge whose effect ids are not among the record's own effects
    fake_effects = dataclasses.replace(real.edges[0], effect_ids=("effect:999",))
    forged = dataclasses.replace(real, edges=(fake_effects, *real.edges[1:]))
    assert not forged.satisfied_by_content()

    # a delay that lost its state/unit/source is not a proven bound
    unsource = dataclasses.replace(real.edges[0], delay_basis="UNSOURCED",
                                   delay_unit="", delay_source="")
    forged = dataclasses.replace(real, edges=(unsource, *real.edges[1:]))
    assert not forged.satisfied_by_content()

    # reuse that no second covering write justifies
    forged = dataclasses.replace(real, reuse=("invented",))
    assert not forged.satisfied_by_content()

    # dropping the buffer list empties the roots every read points at
    forged = dataclasses.replace(real, buffers=())
    assert not forged.satisfied_by_content()


def test_a_forgery_cannot_reach_the_gate_because_resolve_recomputes_it():
    """No consumer takes a proof as input; resolve re-reads the IR every time."""
    import inspect

    from tilefoundry_ascend.plan.resolved import _plan_bytes
    from tilefoundry_ascend.plan.source_audit import DependenceProof

    assert "proof" not in inspect.signature(resolve_schedule).parameters
    _, problem = _independent_problem()
    plan = resolve_schedule(problem, _assignments(problem))
    hollow = DependenceProof(region_id="region:0", straight_line=True, edges=(),
                             reads=(), writes=(), buffers=(), reuse=(),
                             missing=(), derived=True)
    # even if such a record were handed to the bytes encoder, its content binds
    # differently, and the gate above never reads the flag
    assert _plan_bytes(problem, plan.regions, hollow) != plan.plan_canonical_bytes
    assert not hollow.complete()
    # and re-resolving the honest problem yields the same plan identity again
    assert resolve_schedule(problem, _assignments(problem)).plan_identity \
        == plan.plan_identity


def _certified_plan():
    from tilefoundry.analysis import analyze
    from tilefoundry.cli.source import checked_program
    from tilefoundry.schedule import finalize
    from tilefoundry_ascend.plan.numeric import to_schedule_problem
    from tilefoundry_ascend.plan.problem import analyze_native_problem
    from tilefoundry_ascend.plan.units import build_native_unit_tree
    from tilefoundry_ascend.solver import solve_region

    source = ("targets/ascend/tests_native_mainline/fixtures/authors/"
              "independent_ub.py:IndependentUB")
    program = checked_program(source, None, command="test proof recheck")
    analyzed = analyze(program.module, program.entry,
                       analysis=("compute-cost", "memory"))
    module = finalize(analyzed, authored=program.authored)
    problem = analyze_native_problem(
        build_native_unit_tree(module, authored=program.authored), None, None,
        module.entry_function().target)
    answers = {region.region_ref.region_id:
               solve_region(to_schedule_problem(region))
               for region in problem.regions}
    return problem, resolve_schedule(problem, answers)


def test_relabelling_the_stored_proof_is_refused_by_the_consumer_recheck():
    """A proof cannot certify itself by describing itself.

    region_id, straight_line and the edge set are the three fields a stored
    record can change while staying internally consistent - a record with no
    edges contradicts nothing. verify_plan_bindings therefore RECOMPUTES the
    proof from the live problem and compares, so each relabelling is refused.
    """
    from dataclasses import replace as _replace

    problem, plan = _certified_plan()
    verify_plan_bindings(plan, problem)  # the honest pairing passes

    for field, value, code in (
            ("region_id", "region:99", "PLAN_PROOF_REGION_MISMATCH"),
            ("straight_line", False, "PLAN_PROOF_DOMAIN_MISMATCH"),
            ("edges", (), "PLAN_PROOF_STALE")):
        hollow = _replace(plan.dependence_proof, **{field: value})
        with pytest.raises(ResolveError) as caught:
            verify_plan_bindings(_replace(plan, dependence_proof=hollow), problem)
        assert caught.value.code == code, (field, caught.value.code)

    # dropping the proof entirely is refused too
    with pytest.raises(ResolveError) as caught:
        verify_plan_bindings(_replace(plan, dependence_proof=None), problem)
    assert caught.value.code == "PLAN_PROOF_MISSING"


def test_a_proof_built_over_different_content_is_refused_even_if_complete():
    """Self-consistent is not the same as true about THIS program.

    A record assembled from another program's ids can pass its own
    consistency rule; only comparison against the problem the caller hands
    the consumer refuses it.
    """
    from dataclasses import replace as _replace
    from tilefoundry_ascend.plan.source_audit import EdgeProof, dependence_proof

    problem, plan = _certified_plan()
    proof = dependence_proof(problem)
    assert proof.complete()
    edge = proof.edges[0]
    # same shape, correct universe membership would fail: the endpoints and
    # effects are swapped onto ids the region does not declare
    swapped = _replace(proof, edges=(EdgeProof(producer="unit:9", consumer="unit:8",
                                               kind=edge.kind, distance=0,
                                               delay=edge.delay,
                                               delay_state=edge.delay_state,
                                               delay_basis=edge.delay_basis,
                                               delay_unit=edge.delay_unit,
                                               delay_source=edge.delay_source,
                                               effect_ids=edge.effect_ids),))
    assert not swapped.complete(), "the universe check refuses foreign endpoints"
    with pytest.raises(ResolveError):
        verify_plan_bindings(_replace(plan, dependence_proof=swapped), problem)
