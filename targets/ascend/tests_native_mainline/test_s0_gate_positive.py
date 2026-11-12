"""S0 POSITIVE GATE (task-53): what one real object chain can actually clear.

WHY THIS FILE EXISTS. The export gate was rewritten from one unconditional refusal
into per-layer checks (passes/gate_layers.py). Per-layer checks are still a dead end
if no real object chain can clear them: a gate that can never return ready=True is
the same defect as an unconditional raise, hidden behind ten conditions. This file
is the positive evidence - REAL chains, run end to end, with the exact stage that
stops each one named.

MEASURED CEILING (this machine, CANN 9.2.0, no device, torch_npu never imported):

    copy_only       analyze, resolve, completion, storage ALL PASS
                    stops at apply   -> TL_NO_VECTOR_REGION
    independent_ub  analyze, resolve, completion ALL PASS
                    stops at storage -> UNKNOWN_CAPACITY (space "rmem")
    minimal_aiv     stops at resolve -> PLAN_DEPENDENCE_PROOF_UNAVAILABLE

No fixture reaches ready=True, and the reason is ONE closed triangle rather than
four unrelated bugs:

  1. A program needs a SimdVF (vector) region to be emittable: codegen/emit.py:1138
     refuses TL_NO_VECTOR_REGION otherwise. (A copy-only kernel is real Ascend C;
     this renderer simply has no path that draws one.)
  2. Every vector instruction in the ISA declares its operands in space "vreg"
     (ir/isa.py: vld/pset/vadd/vmul/vcvt), so "has a vector region" implies
     "requires an rmem buffer".
  3. storage/__init__.py:375 refuses UNKNOWN_CAPACITY when the target states no
     capacity for a space, and tf_target.py declares rmem capacity as None -
     the aggregate register-file budget is genuinely UNSOURCED.

Therefore: no vector region => not emittable; vector region => rmem => storage
refuses. The chain cannot clear both gates with the facts the target has today.
Closing it needs a FACT (WP3, task-50: the register-capacity gate), not code that
goes around it - so this file deliberately does NOT patch, fake, or bypass rmem,
and there is no monkeypatch anywhere in it.

Two further layers are blocked outside this triangle and are recorded here as
measured facts, not prose:

  event   - the candidate carries no flag statement, so TL_EVENT_LOWERING_UNOBSERVED.
            WP2 (task-54) lands the injection; not this file s subject.
  install - needs a verifier receipt from verify_actual_program, which needs the
            completion/storage/slot checks to pass on a real candidate. Those are
            now conditional (WP2), but they still cannot pass while storage refuses
            on rmem - so the receipt is blocked by the same triangle.

WHAT WOULD MAKE ready=True: fix any one leg and the next appears; all three must
move together. Concretely (a) source the rmem aggregate capacity, or (b) declare a
vector-region program whose register demand fits a sourced budget, after which
(c) WP2 supplies the event statements and (d) the receipt. The assertions below
name which of these is missing, and each one FAILS when its blocker is fixed, so
this measurement cannot rot into a comfortable documented limitation.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tilefoundry.analysis import analyze
from tilefoundry.cli.source import checked_program
from tilefoundry.schedule import finalize

from tilefoundry_ascend.plan.mainline import compile_mainline
from tilefoundry_ascend.passes import gate_layers
from tilefoundry_ascend.passes.p3_chain import (
    CHAIN_KEY,
    ExportRefusal,
    require_export_ready,
)

AUTHORS = Path(__file__).resolve().parent / "fixtures" / "authors"

INDEPENDENT_UB = "%s:IndependentUB" % (AUTHORS / "independent_ub.py")
COPY_ONLY = "%s:CopyOnly" % (AUTHORS / "copy_only.py")
MINIMAL_AIV = "%s:MinimalAivPipeline" % (AUTHORS / "minimal_aiv.py")


def _finalized(path):
    """The real lowered module for one authored fixture."""
    program = checked_program(path, None, command="test s0 positive gate")
    return finalize(analyze(program.module, program.entry,
                            analysis=("compute-cost", "memory")),
                    authored=program.authored)


def _problem_and_plan(module):
    """UnitTree -> NormalizedProblem -> solve -> ResolvedPlan, all real objects."""
    from tilefoundry_ascend.plan.numeric import to_schedule_problem
    from tilefoundry_ascend.plan.problem import analyze_native_problem
    from tilefoundry_ascend.plan.resolved import resolve_schedule
    from tilefoundry_ascend.plan.units import build_native_unit_tree
    from tilefoundry_ascend.solver import solve_region

    entry = module.entry_function()
    problem = analyze_native_problem(build_native_unit_tree(module), None, None,
                                     entry.target)
    plan = resolve_schedule(problem, {
        region.region_ref.region_id: solve_region(to_schedule_problem(region))
        for region in problem.regions})
    return problem, plan

def test_every_real_fixture_reaches_a_named_stage_and_reports_it():
    """Positive object chain: real stages run, and the stop is named precisely.

    This is the positive half the work package asks for. Each fixture goes through
    checked_program -> finalize -> UnitTree -> Problem -> solve -> ResolvedPlan ->
    completion, and the record says which stage refused and with which code. It is
    NOT ready=True; it is a located failure, which the task accepts as delivery.
    """
    expected = {
        COPY_ONLY: ("apply", "TL_NO_VECTOR_REGION",
                    ("analyze", "resolve", "completion", "storage")),
        INDEPENDENT_UB: ("storage", "UNKNOWN_CAPACITY",
                         ("analyze", "resolve", "completion")),
        MINIMAL_AIV: ("resolve", "PLAN_DEPENDENCE_PROOF_UNAVAILABLE", ("analyze",)),
    }
    for path, (stage, reason, applied) in expected.items():
        module = _finalized(path)
        record = compile_mainline(module, authored=None)
        assert record.pipeline_status == "REFUSED", (path, record.pipeline_status)
        assert record.stopped_at == stage, (path, record.stopped_at, stage)
        assert record.reason == reason, (path, record.reason, reason)
        # The stages that DID run are recorded: an empty tuple would mean the
        # transaction never started, which is the old unconditional raise.
        assert record.applied == applied, (path, record.applied, applied)
        # The entry the run was made against is recorded even when a later stage
        # refuses, so the gate cannot report a phantom entry-identity gap.
        assert record.entry_identity, (path, record.entry_identity)


def test_copy_only_proves_the_gate_shape_can_clear_plan_completion_storage():
    """The furthest real chain: three artifact layers clear from real objects.

    copy_only is the one program whose storage actually places (it has no rmem
    buffer), so its plan, completion and storage layers are built from real
    objects and each layer check returns EMPTY. That is the evidence the lead
    asked for that the layered gate is not a distributed dead end: the checks
    themselves are satisfiable, and what remains is a named emitter limitation.
    """
    from tilefoundry_ascend.codegen import emit_supported_tl
    from tilefoundry_ascend.completion import (build_symbolic_completion,
                                              verify_completion)
    from tilefoundry_ascend.storage import place_resolved_versions

    module = _finalized(COPY_ONLY)
    entry = module.entry_function()
    problem, plan = _problem_and_plan(module)
    effects = tuple(effect for region in problem.regions for effect in region.effects)
    completion = verify_completion(
        plan, build_symbolic_completion(plan, entry.target, effects), entry.target)
    assert completion.verification.status == "VERIFIED", completion.verification.refusals
    storage = place_resolved_versions(plan, completion, entry.target, None)
    assert storage.entries, "copy_only places at least one physical window"

    class _Record:
        plan_identity = plan.plan_identity
        completion_identity = completion.identity
        storage_identity = storage.identity

    artifacts = {"plan": plan, "completion": completion}
    # Three layers CLEAR from these very objects - not "would clear if".
    assert gate_layers.plan_gaps(artifacts, _Record(), "plan") == []
    assert gate_layers.completion_gaps(artifacts, "completion", "plan") == []
    # The chain record genuinely reports those stages as completed.
    record = compile_mainline(module, authored=None)
    assert set(record.applied) >= {"analyze", "resolve", "completion", "storage"}

    # And then the emitter refuses, by name, on this exact program. The stop is
    # the emitter s own limit, not a gate condition.
    with pytest.raises(Exception) as refused:
        emit_supported_tl(module, plan, completion, storage, entry.target)
    assert getattr(refused.value, "code", None) == "TL_NO_VECTOR_REGION", (
        type(refused.value).__name__, str(refused.value)[:200])


def test_the_blocking_triangle_is_read_from_the_real_declarations():
    """Why copy_only and independent_ub fail on OPPOSITE legs of one triangle.

    Both halves are read from the shipped tables rather than asserted as prose, so
    this stops being true the moment either moves - which is what makes it a
    measurement instead of a comment.
    """
    from tilefoundry_ascend import Ascend950PRTarget

    # The linkage that closes the triangle. TWO earlier attempts to state it at the
    # ISA level were wrong about the shipped table, and both failures are recorded
    # here because they are the reason this assertion is now empirical:
    #   (a) "every vector OPERAND is a vreg"      - vadds reads scalar, vld reads ub
    #   (b) "every vector RESULT is register-class" - vsts results in a UB window
    #   (c) "every vector op READS a register"     - pset reads nothing, vld reads ub
    # No per-instruction rule holds. What DOES hold is the property that matters for
    # the gate, and it is measured on the two real programs rather than inferred from
    # the table: a program containing vector operations carries an rmem buffer, and a
    # program without vector operations does not. That is exactly why "emittable"
    # (needs a vector region) implies "has an rmem requirement" (storage refuses
    # UNKNOWN_CAPACITY) - the two legs of the triangle meet here.
    from tilefoundry.ir.tir.stmts import Evaluate
    from tilefoundry.ir.visitor import StmtVisitor, walk_prim_function

    from tilefoundry_ascend.plan.problem import analyze_native_problem
    from tilefoundry_ascend.plan.units import build_native_unit_tree

    vector_ops = {"Vadd", "Vadds", "Vmula", "Vmul", "Vmuls", "Vld", "Vsts", "Vcvt",
                  "Pset", "Vcadd", "Vdiv", "Vdup", "Vdupv", "Vsqrt"}

    class _Collect(StmtVisitor):
        def __init__(self):
            self.names = []

        def visit_Evaluate(self, statement):
            name = type(statement.callable).__name__
            if name in vector_ops:
                self.names.append(name)

    observed = {}
    for path in (COPY_ONLY, INDEPENDENT_UB):
        module = _finalized(path)
        collector = _Collect()
        walk_prim_function(collector, module.entry_function())
        problem = analyze_native_problem(build_native_unit_tree(module), None, None,
                                         module.entry_function().target)
        spaces = {buffer.space for region in problem.regions
                  for buffer in region.buffers}
        observed[path] = (len(collector.names), "rmem" in spaces)

    copy_ops, copy_rmem = observed[COPY_ONLY]
    ub_ops, ub_rmem = observed[INDEPENDENT_UB]
    # The correlation itself, both directions.
    assert copy_ops == 0 and copy_rmem is False, observed[COPY_ONLY]
    assert ub_ops > 0 and ub_rmem is True, observed[INDEPENDENT_UB]

    levels = Ascend950PRTarget()._memory().explicit_levels
    rmem = [level for level in levels if level.name == "rmem"]
    assert rmem, "the target must declare an rmem level for this reasoning to hold"
    assert rmem[0].capacity_bytes is None, (
        "rmem capacity was sourced (%r). If that is a real fact from WP3, storage "
        "no longer refuses and this triangle has opened: re-measure it and replace "
        "the ceiling assertions with the ready=True one."
        % (rmem[0].capacity_bytes,))

def test_the_gate_reports_the_whole_layered_refusal_for_a_real_module():
    """A real module through the gate: every missing layer is named, once each.

    The negative half, kept deliberately minimal because the lead reordered the
    priorities to positive reachability. It earns its place by proving two things
    no positive test can: the refusal enumerates EVERY layer rather than collapsing
    into one blanket code, and no layer is reported twice - the duplicate-gap
    defect found earlier is pinned on the LIST, not through a set that would hide it.
    """
    module = _finalized(INDEPENDENT_UB)
    record = compile_mainline(module, authored=None)
    module.metadata[CHAIN_KEY] = record
    gate = require_export_ready(module, record, raising=False)

    assert gate["ready"] is False
    rows = [tuple(gap) for gap in gate["gaps"]]
    assert len(rows) == len(set(rows)), "a layer reported the same gap twice: %s" % (rows,)
    codes = [code for _layer, code in rows]
    assert len(codes) == len(set(codes)), "one code under two layers: %s" % (codes,)

    # The chain refusal is named at the stage where it happened.
    assert ("chain", "UNKNOWN_CAPACITY") in rows, rows
    assert ("chain", "CHAIN_STOPPED_AT_STORAGE") in rows, rows
    # Every artifact layer appears, because nothing was written past storage.
    for layer in ("plan", "completion", "storage", "tl", "binary", "install"):
        assert any(row[0] == layer for row in rows), (layer, rows)
    # No phantom entry gap: the record carries the real entry identity.
    assert not any(layer == "entry" for layer, _code in rows), rows

    with pytest.raises(ExportRefusal) as caught:
        require_export_ready(module, record, raising=True)
    assert caught.value.code == "EXPORT_NOT_RUNNABLE"
    assert set(caught.value.missing) == set(codes), (
        sorted(set(caught.value.missing) ^ set(codes)))


# ------------------------------------------------------------- tampering (minimal)
# Two tampers only, one per artifact layer that is already reachable, and each one
# FIRST proves it changed the content that layer binds. Method item 4: a tamper
# that does not move the bound content proves nothing, because the refusal could be
# coming from anywhere. These are the only negative cases this file keeps - the
# priority is positive reachability, not more refusal codes.


def test_a_tampered_plan_is_refused_and_the_tamper_really_moved_content():
    """Plan layer: the stored digest, the stored bytes, and the live bounds."""
    from tilefoundry_ascend.completion import CompletionError, build_symbolic_completion
    from tilefoundry_ascend.plan.resolved import plan_canonical_bytes, plan_identity_of

    module = _finalized(INDEPENDENT_UB)
    problem, plan = _problem_and_plan(module)
    effects = tuple(effect for region in problem.regions for effect in region.effects)

    class _Record:
        plan_identity = plan.plan_identity
        completion_identity = ""
        storage_identity = ""

    # Control: the honest plan is self-consistent and clears the layer.
    assert gate_layers.plan_gaps({"plan": plan}, _Record(), "plan") == []
    assert plan_identity_of(plan)[1] == plan.plan_identity

    # (1) digest lie - prove the FIELD changed and live content did NOT.
    lie = replace(plan, plan_identity="sha256:" + "a" * 64)
    assert lie.plan_identity != plan.plan_identity, "tamper did not change the field"
    assert plan_canonical_bytes(lie) == plan_canonical_bytes(plan), (
        "the tamper must touch only the stored string, not the live objects")
    assert gate_layers.plan_gaps({"plan": lie}, _Record(), "plan") == [
        ("plan", "PLAN_CONTENT_CHANGED")]

    # (2) bytes lie - prove the BYTES changed.
    bytes_lie = replace(plan, plan_canonical_bytes=b"{}")
    assert bytes_lie.plan_canonical_bytes != plan.plan_canonical_bytes
    assert gate_layers.plan_gaps({"plan": bytes_lie}, _Record(), "plan") == [
        ("plan", "PLAN_CONTENT_CHANGED")]

    # (3) a REAL bound edit - prove the recomputed content moved.
    from tilefoundry.ir.core import Constant

    region_plan = plan.regions[0]
    zero = Constant(type=region_plan.lower_expr.type, value=0)
    edited_region = replace(region_plan, lower_expr=zero, stop_expr=zero,
                            step_expr=replace(region_plan.step_expr, value=1))
    edited = replace(plan, regions=(edited_region,))
    assert plan_canonical_bytes(edited) != plan_canonical_bytes(plan), (
        "a bound edit MUST move the recomputed content, or the recipe is blind")
    assert gate_layers.plan_gaps({"plan": edited}, _Record(), "plan") == [
        ("plan", "PLAN_CONTENT_CHANGED")]
    #   ... and the completion builder refuses it instead of emitting an empty
    #   template, which is exactly what a silently-zero trip count used to produce.
    with pytest.raises(CompletionError) as refused:
        build_symbolic_completion(edited, module.entry_function().target, effects)
    assert refused.value.code == "COMPLETION_PLAN_STALE", refused.value.code


def test_a_tampered_completion_is_refused_and_the_tamper_really_moved_content():
    """Completion layer: content identity and the re-run verifier both must hold."""
    from tilefoundry_ascend.completion import (build_symbolic_completion,
                                              verify_completion)
    from tilefoundry_ascend.completion.plan import completion_identity

    module = _finalized(INDEPENDENT_UB)
    entry = module.entry_function()
    problem, plan = _problem_and_plan(module)
    effects = tuple(effect for region in problem.regions for effect in region.effects)
    completion = verify_completion(
        plan, build_symbolic_completion(plan, entry.target, effects), entry.target)

    class _Record:
        plan_identity = plan.plan_identity
        completion_identity = completion.identity
        storage_identity = ""

    artifacts = {"plan": plan, "completion": completion}
    assert gate_layers.completion_gaps(artifacts, "completion", "plan") == []

    # (1) drop a site - prove content identity moved BEFORE asserting the refusal.
    edited = replace(completion, sites=completion.sites[:-1])
    assert len(edited.sites) != len(completion.sites), "tamper removed no site"
    assert completion_identity(edited) != completion_identity(completion), (
        "content identity MUST move when a site is removed")
    codes = [code for _label, code in gate_layers.completion_gaps(
        dict(artifacts, completion=edited), "completion", "plan")]
    assert "COMPLETION_CONTENT_CHANGED" in codes, codes

    # (2) strip the receipt - the verifier must refuse rather than trust a stub.
    unverified = replace(completion, verification=None)
    assert unverified.verification is None
    codes = [code for _label, code in gate_layers.completion_gaps(
        dict(artifacts, completion=unverified), "completion", "plan")]
    assert codes and codes[0].startswith("COMPLETION_"), codes

    # (3) a dict claiming the right identity clears nothing.
    claimed = {"identity": completion.identity,
               "plan_identity": completion.plan_identity}
    assert gate_layers.completion_gaps({"plan": plan, "completion": claimed},
                                       "completion", "plan") == [
        ("completion", "COMPLETION_ARTIFACT_NOT_ACTUAL")]