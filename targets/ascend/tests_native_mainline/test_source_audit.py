"""Compare actual instruction contracts against normalized effects."""
from dataclasses import replace
import pytest
from tilefoundry_ascend.plan.source_audit import audit_instruction_effects, SourceAuditError
from test_unit_tree import _run_chain, SOURCE


def test_actual_s0_effects_match_instruction_schemas():
    _, _, _, problem = _run_chain(SOURCE)
    for region in problem.regions:
        audit_instruction_effects(region)


@pytest.mark.parametrize("change", [
    {"completion": "ISSUE"}, {"pipe": "wrong-pipe"},
    {"kind": "CONTROL"}, {"access_ids": ()},
])
def test_changed_instruction_effect_is_refused(change):
    _, _, _, problem = _run_chain(SOURCE)
    region = problem.regions[0]
    effect = region.effects[0]
    changed = replace(region, effects=(replace(effect, **change), *region.effects[1:]))
    with pytest.raises(SourceAuditError, match="SOURCE_EFFECT_MISMATCH"):
        audit_instruction_effects(changed)

def test_real_plan_requires_dependency_proof_before_buffer_audit():
    from tilefoundry_ascend.plan.source_audit import audit_plan_buffers
    from tilefoundry_ascend.plan.numeric import to_schedule_problem
    from tilefoundry_ascend.plan.resolved import resolve_schedule
    from tilefoundry_ascend.solver import solve_region

    _, _, _, problem = _run_chain(SOURCE)
    assignments = {r.region_ref.region_id: solve_region(to_schedule_problem(r), None)
                   for r in problem.regions}
    assert all(a.status == "SAT" for a in assignments.values())
    from tilefoundry_ascend.plan.resolved import ResolveError
    with pytest.raises(ResolveError, match="PLAN_DEPENDENCE_PROOF_UNAVAILABLE"):
        resolve_schedule(problem, assignments)

def test_tampered_real_effect_refused_at_resolver_boundary():
    from tilefoundry_ascend.plan.numeric import to_schedule_problem
    from tilefoundry_ascend.plan.resolved import ResolveError, resolve_schedule
    from tilefoundry_ascend.solver import solve_region

    _, _, _, problem = _run_chain(SOURCE)
    assignments = {r.region_ref.region_id: solve_region(to_schedule_problem(r), None)
                   for r in problem.regions}
    assert all(a.status == "SAT" for a in assignments.values())
    region = problem.regions[0]
    altered = replace(region.effects[0], kind="CONTROL", pipe="wrong-pipe",
                      completion="ISSUE", access_ids=())
    forged = replace(problem, regions=(replace(region, effects=(altered, *region.effects[1:])),
                                       *problem.regions[1:]))
    assert forged.coverage.complete
    with pytest.raises(ResolveError, match="SOURCE_EFFECT_MISMATCH"):
        resolve_schedule(forged, assignments)

def test_independent_ub_author_program_has_disjoint_actual_ranges():
    import numpy as np
    from tilefoundry.analysis import analyze
    from tilefoundry.cli.source import checked_program
    from tilefoundry.ir.core import Constant
    from tilefoundry.ir.tir.stmts import LetStmt
    from tilefoundry.ir.tir.memory.tensor_view import TensorView
    from tilefoundry.schedule import finalize
    from tilefoundry_ascend.verify.executor import run_entry
    from test_normal_entry import _walk

    source = "targets/ascend/tests_native_mainline/fixtures/authors/independent_ub.py:IndependentUB"
    program = checked_program(source, None, command="test independent UB")
    analyzed = analyze(program.module, program.entry, analysis=("compute-cost", "memory"))
    module = finalize(analyzed, authored=program.authored)
    entry = module.entry_function()
    assert entry.output_count == 2
    ranges = set()
    for node in _walk(entry.body):
        if not isinstance(node, LetStmt) or not isinstance(getattr(node.value, "target", None), TensorView):
            continue
        if node.var.type.storage.name != "SMEM":
            continue
        base = node.value.args[0]
        assert isinstance(base, Constant)
        assert tuple(node.var.type.shape) == (64,)
        assert node.var.type.dtype.name == "f32"
        ranges.add((int(base.value), int(base.value) + 64 * 4))
    assert ranges == {(0, 256), (256, 512)}
    for x in (np.arange(64, dtype="float32"), np.arange(64, dtype="float32") / 4 - 8):
        result = run_entry(module, {"x": x})
        assert np.array_equal(result[entry.params[-2].name], 2 * x)
        assert np.array_equal(result[entry.params[-1].name], x)

def test_untrusted_plan_view_missing_real_buffer_is_refused():
    from types import SimpleNamespace
    from tilefoundry_ascend.plan.source_audit import audit_plan_buffers
    _, _, _, problem = _run_chain(SOURCE)
    regions = []
    for region in problem.regions:
        buffers = tuple(SimpleNamespace(buffer_ref=b.root_ref, buffer_id=b.buffer_id,
                                       domain=b.domain, resolved_versions=1)
                        for b in region.buffers)
        regions.append(SimpleNamespace(region_ref=region.region_ref, buffers=buffers))
    # This is an untrusted projection for the checker, not an authenticated plan.
    index = next(i for i, region in enumerate(regions)
                 if any(b.buffer_ref.type.storage.name == "RMEM" for b in region.buffers))
    regions[index].buffers = tuple(b for b in regions[index].buffers
                                  if b.buffer_ref.type.storage.name != "RMEM")
    with pytest.raises(SourceAuditError, match="SOURCE_BUFFER_COVERAGE"):
        audit_plan_buffers(SimpleNamespace(regions=tuple(regions)), problem)

def test_static_ub_window_reader_uses_actual_byte_ranges():
    from tilefoundry.analysis import analyze
    from tilefoundry.cli.source import checked_program
    from tilefoundry.schedule import finalize
    from tilefoundry_ascend.plan.source_audit import read_static_ub_windows
    source = "targets/ascend/tests_native_mainline/fixtures/authors/independent_ub.py:IndependentUB"
    program = checked_program(source, None, command="test UB audit")
    analyzed = analyze(program.module, program.entry, analysis=("compute-cost", "memory"))
    module = finalize(analyzed, authored=program.authored)
    windows = read_static_ub_windows(module.entry_function())
    assert {(start, end) for _, start, end in windows} == {(0, 256), (256, 512)}
    assert len({id(var) for var, _, _ in windows}) == len(windows)

def test_independent_reaching_definitions_from_actual_tir():
    from tilefoundry.analysis import analyze
    from tilefoundry.cli.source import checked_program
    from tilefoundry.schedule import finalize
    from tilefoundry_ascend.plan.units import build_native_unit_tree
    from tilefoundry_ascend.plan.problem import analyze_native_problem
    from tilefoundry_ascend.plan.source_audit import audit_single_region_reaching_definitions
    source = "targets/ascend/tests_native_mainline/fixtures/authors/independent_ub.py:IndependentUB"
    program = checked_program(source, None, command="test reaching definitions")
    module = finalize(analyze(program.module, program.entry, analysis=("compute-cost", "memory")), authored=program.authored)
    problem = analyze_native_problem(build_native_unit_tree(module, authored=program.authored), None, None, module.target)
    found = audit_single_region_reaching_definitions(problem)
    wrong = replace(problem.regions[0].accesses[0], root_ref=entry_output(problem))
    region = problem.regions[0]
    changed = replace(problem, regions=(replace(region, accesses=(wrong, *region.accesses[1:])),))
    with pytest.raises(SourceAuditError, match="SOURCE_ACCESS_ROOT_MISMATCH"):
        audit_single_region_reaching_definitions(changed)
    assert found == frozenset({
        ("unit:0", "unit:1", "RAW", "effect:0", "effect:1"),
        ("unit:0", "unit:1", "RAW", "effect:0", "effect:2"),
        ("unit:0", "unit:3", "RAW", "effect:0", "effect:7"),
        ("unit:1", "unit:2", "RAW", "effect:5", "effect:6"),
    })

def entry_output(problem):
    return problem.source_snapshot.entry_ref.params[-1]

def test_independent_dependency_comparison_detects_missing_and_reversed_raw():
    from tilefoundry.analysis import analyze
    from tilefoundry.cli.source import checked_program
    from tilefoundry.schedule import finalize
    from tilefoundry_ascend.plan.units import build_native_unit_tree
    from tilefoundry_ascend.plan.problem import analyze_native_problem
    from tilefoundry_ascend.plan.source_audit import (
        audit_single_region_dependencies,
        dependence_proof,
    )

    source = "targets/ascend/tests_native_mainline/fixtures/authors/independent_ub.py:IndependentUB"
    program = checked_program(source, None, command="test dependency audit")
    module = finalize(analyze(program.module, program.entry, analysis=("compute-cost", "memory")), authored=program.authored)
    problem = analyze_native_problem(build_native_unit_tree(module, authored=program.authored), None, None, module.target)
    region = problem.regions[0]
    first = region.edges[0]
    assert first.kind == "RAW"
    assert set(first.effect_ids) == {"effect:0", "effect:1", "effect:2"}
    # R7: this program's evidence IS complete, so the reader says so instead of
    # refusing. That is a derivation, not a status: the proof carries every edge,
    # read and write it used, and an empty missing list.
    proof = audit_single_region_dependencies(problem)
    assert proof.complete() and proof.missing == ()
    assert len(proof.edges) == 3 and len(proof.reads) == 9
    assert {edge.delay_basis for edge in proof.edges} == {"OCCUPANCY_FALLBACK"}
    assert all(edge.delay_unit == "MODEL_TICK" for edge in proof.edges)
    assert proof.reuse == ()
    for edges in (region.edges[1:],
                  (replace(first, producer=first.consumer, consumer=first.producer), *region.edges[1:]),
                  (replace(first, effect_ids=("effect:0", "effect:1")), *region.edges[1:])):
        altered = replace(problem, regions=(replace(region, edges=edges),))
        with pytest.raises(SourceAuditError, match="SOURCE_DEPENDENCE_MISMATCH"):
            audit_single_region_dependencies(altered)
    altered = replace(problem, regions=(replace(region, edges=(replace(first, delay=-1), *region.edges[1:])),))
    with pytest.raises(SourceAuditError, match="SOURCE_DEPENDENCE_DELAY_INVALID"):
        audit_single_region_dependencies(altered)
    # An edge whose delay lost its source is a NAMED missing item, not a silence.
    unsourced = replace(first, delay=None, delay_fact=None)
    altered = replace(problem, regions=(replace(region, edges=(unsourced, *region.edges[1:])),))
    incomplete = dependence_proof(altered)
    assert not incomplete.complete()
    assert any("no sourced delay" in item for item in incomplete.missing)

def test_real_tir_reordering_exposes_uninitialized_ub_read():
    from tilefoundry.analysis import analyze
    from tilefoundry.cli.source import checked_program
    from tilefoundry.schedule import finalize
    from tilefoundry.ir.tir.stmts import Sequential
    from tilefoundry_ascend.plan.units import build_native_unit_tree
    from tilefoundry_ascend.plan.problem import analyze_native_problem
    from tilefoundry_ascend.plan.source_audit import audit_single_region_reaching_definitions
    source = "targets/ascend/tests_native_mainline/fixtures/authors/independent_ub.py:IndependentUB"
    p = checked_program(source, None, command="test uninitialized read")
    module = finalize(analyze(p.module, p.entry, analysis=("compute-cost", "memory")), authored=p.authored)
    problem = analyze_native_problem(build_native_unit_tree(module, authored=p.authored), None, None, module.target)
    entry = module.entry_function()
    mesh = entry.body.body[0]
    statements = mesh.body.body
    # Preserve the real statements and binders; move the GM load after its VF consumer.
    reordered = (statements[0], statements[2], statements[1], *statements[3:])
    changed_entry = replace(entry, body=Sequential((replace(mesh, body=Sequential(reordered)),)))
    changed = replace(problem, source_snapshot=replace(problem.source_snapshot, entry_ref=changed_entry))
    with pytest.raises(SourceAuditError, match="SOURCE_READ_UNINITIALIZED"):
        audit_single_region_reaching_definitions(changed)

def test_real_tir_partially_overlapping_ub_views_are_refused():
    from tilefoundry.analysis import analyze
    from tilefoundry.cli.source import checked_program
    from tilefoundry.schedule import finalize
    from tilefoundry.ir.tir.stmts import Sequential, ExecutionScope, LetStmt
    from tilefoundry.ir.tir.memory.tensor_view import TensorView
    from tilefoundry_ascend.plan.source_audit import read_static_ub_windows
    source = "targets/ascend/tests_native_mainline/fixtures/authors/independent_ub.py:IndependentUB"
    p = checked_program(source, None, command="test overlapping views")
    module = finalize(analyze(p.module, p.entry, analysis=("compute-cost", "memory")), authored=p.authored)
    entry = module.entry_function()
    mesh = entry.body.body[0]
    statements = list(mesh.body.body)
    position = next(i for i, item in enumerate(statements) if isinstance(item, ExecutionScope))
    vf = statements[position]
    body = list(vf.body.body)
    for i, stmt in enumerate(body):
        value = getattr(stmt, "value", None)
        if isinstance(stmt, LetStmt) and isinstance(getattr(value, "target", None), TensorView) and value.args[0].value == 0:
            body[i] = replace(stmt, value=replace(value, args=(replace(value.args[0], value=128),)))
            break
    else:
        raise AssertionError("expected actual UB output view")
    statements[position] = replace(vf, body=Sequential(tuple(body)))
    altered = replace(entry, body=Sequential((replace(mesh, body=Sequential(tuple(statements))),)))
    with pytest.raises(SourceAuditError, match="SOURCE_UB_ALIAS_UNSUPPORTED"):
        read_static_ub_windows(altered)

def test_real_tir_wrong_predicate_width_cannot_prove_old_dst_dead():
    from tilefoundry.analysis import analyze
    from tilefoundry.cli.source import checked_program
    from tilefoundry.schedule import finalize
    from tilefoundry_ascend.plan.units import build_native_unit_tree
    from tilefoundry_ascend.plan.problem import analyze_native_problem
    from tilefoundry_ascend.plan.source_audit import audit_single_region_reaching_definitions
    from tilefoundry_ascend.ir.ops import Pset
    source = "targets/ascend/tests_native_mainline/fixtures/authors/independent_ub.py:IndependentUB"
    p = checked_program(source, None, command="test mask proof")
    module = finalize(analyze(p.module, p.entry, analysis=("compute-cost", "memory")), authored=p.authored)
    problem = analyze_native_problem(build_native_unit_tree(module, authored=p.authored), None, None, module.target)
    effect = next(e for e in problem.regions[0].effects if isinstance(e.source_ref.callable, Pset))
    op = effect.source_ref.callable
    original_width = op.elem_width
    try:
        op.elem_width = 16
        with pytest.raises(SourceAuditError, match="SOURCE_MASK_UNPROVEN"):
            audit_single_region_reaching_definitions(problem)
    finally:
        op.elem_width = original_width


def _independent_problem():
    from tilefoundry.analysis import analyze
    from tilefoundry.cli.source import checked_program
    from tilefoundry.schedule import finalize
    from tilefoundry_ascend.plan.problem import analyze_native_problem
    from tilefoundry_ascend.plan.units import build_native_unit_tree

    source = ("targets/ascend/tests_native_mainline/fixtures/authors/"
              "independent_ub.py:IndependentUB")
    program = checked_program(source, None, command="test prose bound")
    analyzed = analyze(program.module, program.entry,
                       analysis=("compute-cost", "memory"))
    module = finalize(analyzed, authored=program.authored)
    return analyze_native_problem(
        build_native_unit_tree(module, authored=program.authored), None, None,
        module.entry_function().target)


def test_a_stored_proof_carries_no_prose_in_the_bound_content():
    """What the identity hashes is names and values, never an explanation.

    A complete proof has an empty missing list by definition, so the only
    prose the reader produces (which names what is NOT proven) exists in the
    refusal detail and never reaches plan bytes. Any wording change to those
    messages therefore cannot invalidate a stored plan.
    """
    _, _, _, problem = _run_chain(SOURCE)
    from tilefoundry_ascend.plan.numeric import to_schedule_problem
    from tilefoundry_ascend.plan.resolved import resolve_schedule
    from tilefoundry_ascend.solver import solve_region

    independent = _independent_problem()
    plan = resolve_schedule(independent, {region.region_ref.region_id:
                                          solve_region(to_schedule_problem(region), None)
                                          for region in independent.regions})
    proof = plan.dependence_proof
    assert proof.complete() and proof.missing == ()
    blob = plan.plan_canonical_bytes
    for sentence in (b"UNRESOLVED", b"never projected", b"lower bound from",
                     b"no role value", b"carried", b"is not derivable"):
        assert sentence not in blob, sentence
    # and the locators that ARE bound are records of names
    for edge in proof.canonical()["edges"]:
        locator = edge["delay_source"]
        assert locator["#"] == "model_source", locator
        assert all(locator.get(key) for key in ("module", "symbol", "field"))
        assert "value" in locator


def test_renaming_the_effect_instruction_field_is_refused_by_the_ir():
    """A tamper that used to be a no-op.

    Effect.instruction names the instruction an effect was declared by. It is
    bound by the audit against the REAL op's contract name, so renaming it in
    the problem alone cannot pass (it also used to leave every hashed digest
    unchanged, which made it an invisible field).
    """
    _, _, _, problem = _run_chain(SOURCE)
    region = problem.regions[0]
    first = region.effects[0]
    assert first.instruction, "the tree records the instruction per effect"
    forged = replace(region.effects[0], instruction="ghost_instruction")
    altered = replace(problem, regions=(replace(
        region, effects=(forged, *region.effects[1:])), *problem.regions[1:]))
    with pytest.raises(SourceAuditError, match="SOURCE_EFFECT_MISMATCH"):
        audit_instruction_effects(altered.regions[0])
    from tilefoundry_ascend.plan.numeric import to_schedule_problem
    from tilefoundry_ascend.plan.resolved import ResolveError, resolve_schedule
    from tilefoundry_ascend.solver import solve_region

    answers = {item.region_ref.region_id:
               solve_region(to_schedule_problem(item), None)
               for item in problem.regions}
    with pytest.raises(ResolveError, match="SOURCE_EFFECT_MISMATCH"):
        resolve_schedule(altered, answers)


def test_a_locator_pointing_at_no_registered_row_is_named_not_ignored():
    """A structurally valid but vacuous locator is refused with its name.

    Renaming only the symbol inside the locator used to produce a plan: the
    content check saw a well-formed record and the delay number was still
    correct, so the proof cited a source that does not exist. The proof now
    resolves every locator against the delay table and names it as a defect.
    """
    from tilefoundry_ascend.plan.numeric import to_schedule_problem
    from tilefoundry_ascend.plan.resolved import ResolveError, resolve_schedule
    from tilefoundry_ascend.plan.source_audit import dependence_proof
    from tilefoundry_ascend.solver import solve_region

    problem = _independent_problem()
    answers = {region.region_ref.region_id:
               solve_region(to_schedule_problem(region))
               for region in problem.regions}
    assert dependence_proof(problem).complete()

    region = problem.regions[0]
    edge = region.edges[0]
    ghost = replace(edge, delay_fact=replace(
        edge.delay_fact,
        source=replace(edge.delay_fact.source, symbol="gm_to_ubx")))
    altered = replace(problem, regions=(replace(
        region, edges=(ghost, *region.edges[1:])),))
    proof = dependence_proof(altered)
    assert not proof.complete()
    # the reason lives in the single rule, defects(), which the resolver prints
    assert any("gm_to_ubx" in item and "no registered delay row" in item
               for item in proof.defects()), proof.defects()
    assert proof.missing == (), "the locator defect is derived, not pre-stated"
    with pytest.raises(ResolveError) as caught:
        resolve_schedule(altered, answers)
    assert caught.value.code == "PLAN_DEPENDENCE_PROOF_UNAVAILABLE"
    assert "no registered delay row" in caught.value.detail, caught.value.detail


def test_a_typed_anchor_is_refused_not_believed_not_silently_ignored():
    """A self-reported anchor can never certify itself.

    The proof reads the LIVE content, so a typed digest does not move it. The
    resolver then compares the anchor fields against its own recompute and
    REFUSES BY NAME: ignoring a lie silently would leave the plan recording
    one anchor while the object claims another, which is exactly the kind of
    disagreement a consumer cannot see later.
    """
    from tilefoundry_ascend.plan.numeric import to_schedule_problem
    from tilefoundry_ascend.plan.problem import problem_identity_of
    from tilefoundry_ascend.plan.resolved import ResolveError, resolve_schedule
    from tilefoundry_ascend.plan.source_audit import dependence_proof
    from tilefoundry_ascend.solver import solve_region

    problem = _independent_problem()
    answers = {region.region_ref.region_id:
               solve_region(to_schedule_problem(region))
               for region in problem.regions}
    real = dependence_proof(problem)
    lied = replace(problem, problem_identity="sha256:" + "0" * 64,
                   problem_canonical_bytes=b'{"lie":1}')
    assert dependence_proof(lied).canonical() == real.canonical(), \
        "the proof must not read a caller-supplied anchor"
    with pytest.raises(ResolveError) as caught:
        resolve_schedule(lied, answers)
    assert caught.value.code == "PROBLEM_CONTENT_CHANGED"
    assert "recomputed" in caught.value.detail

    # a lie in the digest ALONE, with intact bytes, is caught as well
    digest_only = replace(problem, problem_identity="sha256:" + "f" * 64)
    with pytest.raises(ResolveError) as caught:
        resolve_schedule(digest_only, answers)
    assert caught.value.code == "PROBLEM_CONTENT_CHANGED"
    assert "sha256:" + "f" * 64 in caught.value.detail

    # and an honest problem records the recomputed anchors on the plan
    plan = resolve_schedule(problem, answers)
    recomputed, identity = problem_identity_of(problem)
    assert plan.problem_identity == identity == problem.problem_identity
    assert plan.problem_canonical_bytes == recomputed
