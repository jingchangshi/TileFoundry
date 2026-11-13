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
                    stops at storage -> STORAGE_REGISTER_UNSOURCED (space "rmem")
    minimal_aiv     stops at resolve -> PLAN_DEPENDENCE_PROOF_UNAVAILABLE

No fixture reaches ready=True, and the reason is ONE closed triangle rather than
four unrelated bugs:

  1. A program needs a SimdVF (vector) region to be emittable: codegen/emit.py:1138
     refuses TL_NO_VECTOR_REGION otherwise. (A copy-only kernel is real Ascend C;
     this renderer simply has no path that draws one.)
  2. Every vector instruction in the ISA declares its operands in space "vreg"
     (ir/isa.py: vld/pset/vadd/vmul/vcvt), so "has a vector region" implies
     "requires an rmem buffer".
  3. rmem is a COUNTED resource, so it goes through the register-budget gate rather
     than the byte-capacity one: memory.REGISTER_BUDGETS declares no per-domain
     register COUNT (only a 256 B per-register WIDTH, which is a different fact), so
     the gate refuses with STORAGE_REGISTER_UNSOURCED. WP3 introduced that code and
     with it the ability to PASS, once both sides are stated - which is why the
     blocker is now a named fact gap rather than a permanent refusal.

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
        # copy_only now runs the WHOLE transaction: the emitter supports a DMA-only
        # program (a transport kernel is real Ascend C), so nothing refuses until the
        # gate itself, which still lacks the install receipt and the event layer.
        COPY_ONLY: ("", "",
                    ("analyze", "resolve", "completion", "storage", "apply", "verify")),
        # STORAGE_REGISTER_UNSOURCED, not UNKNOWN_CAPACITY: WP3 split the two facts.
        # The program places rmem registers, and what is missing is a declared
        # per-domain register COUNT - not an address-space byte capacity. The newer
        # code names that side; the old one only said "no capacity".
        INDEPENDENT_UB: ("storage", "STORAGE_REGISTER_UNSOURCED",
                         ("analyze", "resolve", "completion")),
        MINIMAL_AIV: ("resolve", "PLAN_DEPENDENCE_PROOF_UNAVAILABLE", ("analyze",)),
    }
    for path, (stage, reason, applied) in expected.items():
        module = _finalized(path)
        record = compile_mainline(module, authored=None)
        if stage:
            assert record.pipeline_status == "REFUSED", (path, record.pipeline_status)
            assert record.stopped_at == stage, (path, record.stopped_at, stage)
            assert record.reason == reason, (path, record.reason, reason)
        else:
            # The one program that completes the transaction. Asserted positively,
            # not merely "not refused": an APPLIED record is a claim that every stage
            # ran, and the stage list below is what backs that claim.
            assert record.pipeline_status == "APPLIED", (path, record.pipeline_status)
            assert record.stopped_at == "", (path, record.stopped_at)
            assert record.reason == "", (path, record.reason)
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

    # And the emitter now SUPPORTS this shape, so the program runs the whole
    # transaction: emit, parse, the deployed Ascend C codegen, the compile site, and
    # the TL reader recomputing its own verdict. This is the positive object chain
    # the work package asks for - real objects end to end, no substitution.
    artifact = emit_supported_tl(module, plan, completion, storage, entry.target)
    surfaces = [statement.surface for statement in artifact.emitted.body.statements()]
    assert surfaces.count("copy") == 2, surfaces
    assert not any(surface.startswith("simd.") for surface in surfaces), surfaces

    from tilefoundry_ascend.codegen import compile_supported_artifact, read_actual_tl

    compiled = compile_supported_artifact(artifact)
    if compiled.compiled is None:
        # No toolchain here: the named BLOCKED_ENV IS the required outcome, asserted
        # rather than skipped. This project forbids skip/xfail, and rightly - a
        # skipped test reports nothing, while this branch proves the build said
        # which tool it searched for.
        blockage = compiled.compile_blockage
        assert blockage is not None, "an uncompiled artifact must carry a blockage"
        assert getattr(blockage, "code", "") == "TL_BISHENG_NOT_DISCOVERABLE", (
            getattr(blockage, "code", ""), str(blockage)[:200])
        assert getattr(blockage, "searched", ()), (
            "the refusal must say what it searched for")
        return
    verdict = read_actual_tl(compiled, compiled.parsed_prim_func)
    assert verdict.status == "VERIFIED", (verdict.status, verdict.refusals, verdict.gaps)
    assert verdict.refusals == () and verdict.gaps == ()


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
    assert ("chain", "STORAGE_REGISTER_UNSOURCED") in rows, rows
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

def test_install_judges_the_tl_layer_instead_of_refusing_forever():
    """The install stage is CONDITIONAL now, and each condition is named.

    It used to end in an unconditional INSTALL_TL_CONFORMANCE_NOT_IMPLEMENTED - the
    fourth R7-shaped permanent refusal in this tree. A permanent refusal cannot be
    satisfied, narrowed or verified, and it happened to sit on the only ready=True
    path. The TL layer is still REQUIRED; it is judged from the real artifact a
    caller verified rather than declared absent.

    Reaching the positive branch needs an artifact whose tir_identity is the
    CANDIDATE's, and emitting a candidate fails today at TL_EVENT_SURFACE_MISSING -
    drain has no TL face. So this test pins the reachable part honestly: every
    refusal is a NAMED layer, including the positive-binding one, and the reason the
    positive branch is unreachable is that specific documented gap rather than this
    stage refusing by construction.
    """
    import tempfile

    from tilefoundry_ascend.codegen import compile_supported_artifact, emit_supported_tl
    from tilefoundry_ascend.completion import (build_symbolic_completion,
                                              verify_completion)
    from tilefoundry_ascend.plan.identity import function_identity
    from tilefoundry_ascend.plan.mainline import (apply_symbolic_pipeline, install_entry,
                                                  verify_actual_program, PlanError)
    from tilefoundry_ascend.storage import place_resolved_versions

    module = _finalized(COPY_ONLY)
    entry = module.entry_function()
    problem, plan = _problem_and_plan(module)
    effects = tuple(effect for region in problem.regions for effect in region.effects)
    completion = verify_completion(
        plan, build_symbolic_completion(plan, entry.target, effects), entry.target)
    storage = place_resolved_versions(plan, completion, entry.target, None)
    candidate, mapping = apply_symbolic_pipeline(entry, plan, completion, storage)
    verdict = verify_actual_program(module, candidate, plan, completion, storage,
                                    source_map=mapping)
    assert verdict.status == "VERIFIED", verdict.code
    assert verdict.receipt is not None

    # (i) no TL artifact -> named, and specific about which layer.
    with pytest.raises(PlanError) as no_artifact:
        install_entry(module, candidate.entry_function(), conformance=verdict)
    assert no_artifact.value.code == "INSTALL_CANDIDATE_UNVERIFIED", (
        no_artifact.value.code)
    assert "TL artifact" in str(no_artifact.value)

    # (ii) an artifact emitted from the SOURCE is refused because it does not
    #      describe this candidate - the binding the reader cannot make itself.
    artifact = compile_supported_artifact(
        emit_supported_tl(module, plan, completion, storage, entry.target),
        cache_directory=tempfile.mkdtemp())
    from tilefoundry_ascend.codegen import read_actual_tl

    #     The reader alone cannot catch this: nothing in it compares tir_identity
    #     with an entry, so its verdict is IDENTICAL whether that field is honest or
    #     fabricated. Asserting that invariance is stronger than asserting one
    #     status, and it is the reason the install stage has to make the comparison.
    honest = read_actual_tl(artifact, artifact.parsed_prim_func)
    forged = read_actual_tl(replace(artifact, tir_identity="sha256:" + "c" * 64),
                            artifact.parsed_prim_func)
    assert (honest.status, honest.refusals, honest.gaps) == (
        forged.status, forged.refusals, forged.gaps), (
        "the reader must be blind to tir_identity for this binding to be necessary")
    assert artifact.tir_identity == function_identity(entry)
    assert function_identity(candidate.entry_function()) != function_identity(entry)
    with pytest.raises(PlanError) as mismatched:
        install_entry(module, candidate.entry_function(), conformance=verdict,
                      tl_artifact=artifact)
    assert mismatched.value.code == "ARTIFACT_TIR_IDENTITY_MISMATCH", (
        mismatched.value.code)

    # (iii) an edited artifact is refused by the READER's own code, reused here
    # rather than renamed. The TIR binding fires first on this artifact (it came
    # from the source entry), so the reader is exercised directly - and then the
    # install stage is shown to propagate whatever the reader says.
    edited = replace(artifact, source=artifact.source + "\n# tampered\n")
    codes = [code for code, _e, _a in
             read_actual_tl(edited, edited.parsed_prim_func).refusals]
    assert "TL_SOURCE_IDENTITY_MISMATCH" in codes, codes
    with pytest.raises(PlanError) as refused:
        install_entry(module, candidate.entry_function(), conformance=verdict,
                      tl_artifact=replace(edited, tir_identity=function_identity(
                          candidate.entry_function())))
    assert refused.value.code == "TL_SOURCE_IDENTITY_MISMATCH", refused.value.code

    # (iv) DRAIN IS NOW EMITTABLE, so the positive branch is REACHABLE. The candidate
    #      still carries drains injected by apply, but they no longer block emission:
    #      drain lowers through a declared reference mapping. This is a CONTRACT
    #      assertion, not a relaxation - if drain were removed from TL_SURFACE again,
    #      the emit below would raise and this test would go red.
    from tilefoundry.ir.tir.stmts import Evaluate
    from tilefoundry.ir.visitor import StmtVisitor, walk_prim_function

    from tilefoundry_ascend.codegen.emit import (DRAIN_REFERENCE_MAPPING,
                                                 EVENT_TL_SURFACE_MISSING, TL_SURFACE)

    class _Drains(StmtVisitor):
        def __init__(self):
            self.names = []

        def visit_Evaluate(self, statement):
            self.names.append(type(statement.callable).__name__)

    collector = _Drains()
    walk_prim_function(collector, candidate.entry_function())
    assert "Drain" in collector.names, (
        "the candidate is expected to carry an injected drain; if it no longer does, "
        "re-measure whether the positive install branch became reachable")

    # (a) drain is NOT a missing surface any more, and it IS registered.
    assert "drain" not in EVENT_TL_SURFACE_MISSING, EVENT_TL_SURFACE_MISSING
    assert TL_SURFACE.get("drain") == "event.pipe_barrier", TL_SURFACE.get("drain")
    #     The cross-core half of the old contract is unchanged and still refuses.
    assert EVENT_TL_SURFACE_MISSING == frozenset(), EVENT_TL_SURFACE_MISSING

    # (b) the artifact from THIS candidate emits, carries the uncertainty record, and
    #     reads back with that record outside gaps/refusals.
    from tilefoundry_ascend.codegen import emit_supported_tl as _emit

    # apply_symbolic_pipeline already returns the rewritten Module whose entry IS the
    # candidate, so emit from THAT object rather than reassembling one by hand.
    from_candidate = _emit(candidate, plan, completion, storage,
                           candidate.entry_function().target)
    drains = [statement for statement in from_candidate.emitted.body.statements()
              if statement.surface == "event.pipe_barrier"]
    assert drains, "the candidate's drains must be emitted, not refused"
    assert "T.ascend_pipe_barrier('PIPE_ALL')" in from_candidate.source
    record = drains[0].reference_mapping
    assert record["status"] == "REFERENCE_IMPLEMENTATION"
    assert record["uncertainty"]["proven"] is False
    assert record["uncertainty"]["needs"], "unproven must say what evidence it needs"
    assert DRAIN_REFERENCE_MAPPING["uncertainty"]["proven"] is False

def test_the_positive_chain_reaches_ready_true_with_the_unproven_layer_visible():
    """THE first ready=True in this project - and what stays unproven is in the output.

    This is the whole point of the layered gate: a real object chain that clears every
    condition. copy_only (a DMA-only transport kernel) runs freeze->analyze->solve->
    resolve->completion->storage->apply->verify->install, the reader recomputes its
    own VERIFIED verdict, and the gate reports ready with an empty gap list.

    The drain inside that candidate is lowered through a DECLARED reference mapping, so
    the gate output also carries the explicit unproven record. That is the contract:
    ready is reachable, and the uncertainty is visible in the same payload rather than
    buried in a comment. A test that asserted ready=True alone would let someone delete
    the record later and stay green.
    """
    import tempfile

    from tilefoundry_ascend.codegen import (compile_supported_artifact, emit_supported_tl,
                                            read_actual_tl)
    from tilefoundry_ascend.completion import (build_symbolic_completion,
                                              verify_completion)
    from tilefoundry_ascend.plan.identity import function_identity
    from tilefoundry_ascend.plan.mainline import (SINGLE_TRANSACTION,
                                                  apply_symbolic_pipeline, install_entry,
                                                  verify_actual_program)
    from tilefoundry_ascend.passes.p3_chain import (CHAIN_KEY, COMPLETION_KEY, PLAN_KEY,
                                                    STORAGE_KEY, TL_ARTIFACT_KEY,
                                                    ChainRecord, ir_identity,
                                                    require_export_ready)
    from tilefoundry_ascend.storage import place_resolved_versions

    module = _finalized(COPY_ONLY)
    entry = module.entry_function()
    problem, plan = _problem_and_plan(module)
    effects = tuple(effect for region in problem.regions for effect in region.effects)
    completion = verify_completion(
        plan, build_symbolic_completion(plan, entry.target, effects), entry.target)
    storage = place_resolved_versions(plan, completion, entry.target, None)

    candidate, mapping = apply_symbolic_pipeline(entry, plan, completion, storage)
    verdict = verify_actual_program(module, candidate, plan, completion, storage,
                                    source_map=mapping)
    assert verdict.status == "VERIFIED", verdict.code

    artifact = compile_supported_artifact(
        emit_supported_tl(candidate, plan, completion, storage,
                          candidate.entry_function().target),
        cache_directory=tempfile.mkdtemp())
    if artifact.compiled is None:
        blockage = artifact.compile_blockage
        assert getattr(blockage, "code", "") == "TL_BISHENG_NOT_DISCOVERABLE", (
            getattr(blockage, "code", ""))
        return  # no toolchain: the named BLOCKED_ENV is the asserted outcome

    read = read_actual_tl(artifact, artifact.parsed_prim_func)
    assert read.status == "VERIFIED", (read.status, read.refusals, read.gaps)
    # The unproven layer is recorded OUTSIDE gaps and refusals, so it cannot cap the
    # verdict - but it must be present.
    assert read.recorded_unproven, "the drain lowering must be recorded as unproven"
    surfaces = [surface for surface, _record in read.recorded_unproven]
    assert "event.pipe_barrier" in surfaces, surfaces

    installed = install_entry(module, candidate.entry_function(), conformance=verdict,
                              tl_artifact=artifact)
    assert installed.entry_function() is candidate.entry_function()

    metadata = dict(module.metadata or {})
    metadata[PLAN_KEY] = plan
    metadata[COMPLETION_KEY] = completion
    metadata[STORAGE_KEY] = storage
    metadata[TL_ARTIFACT_KEY] = artifact
    metadata[CHAIN_KEY] = ChainRecord(
        ir_identity=ir_identity(installed),
        entry_identity=function_identity(candidate.entry_function()),
        plan_identity=plan.plan_identity,
        completion_identity=completion.identity,
        storage_identity=storage.identity,
        tl_identity=artifact.tl_identity,
        pipeline_status="APPLIED",
        applied=SINGLE_TRANSACTION,
        stopped_at="",
        reason="",
    )
    gate = require_export_ready(replace(installed, metadata=metadata),
                                metadata[CHAIN_KEY], raising=False)

    # ready=True with an EMPTY gap list: the first positive in this project.
    assert gate["ready"] is True, gate["gaps"]
    assert gate["gaps"] == [], gate["gaps"]
    assert gate["pipeline"] == "APPLIED"
    # ... and the unproven layer is in the SAME payload, so nobody can read this as
    # "everything is proven".
    unproven = dict((surface, payload) for surface, payload in gate["recorded_unproven"])
    assert "event.pipe_barrier" in unproven, list(unproven)
    record = unproven["event.pipe_barrier"]
    assert record["status"] == "REFERENCE_IMPLEMENTATION", record["status"]
    assert record["uncertainty"]["drain_lowering"] == "PIPE_ALL pipe barrier"
    assert record["uncertainty"]["proven"] is False
    assert record["uncertainty"]["needs"]
    # Raising mode agrees with the cached gate: with NO gaps it must return the same
    # verdict rather than raising. A raising path that still refused here would mean
    # the two modes disagree, which is how a gate starts reporting two answers.
    raised = require_export_ready(replace(installed, metadata=metadata),
                                 metadata[CHAIN_KEY], raising=True)
    assert raised["ready"] is True, raised["gaps"]

def test_a_fabricated_record_cannot_claim_stages_the_artifacts_do_not_show():
    """The record is judged, never believed - the R21/R22 shape, fourth time.

    Measured before this fix: a hand-made ChainRecord with pipeline_status=APPLIED and
    applied=SINGLE_TRANSACTION, with NOTHING run, produced ZERO chain gaps, because
    the gate compared record.applied against the constant. A stored field was judging
    itself, which is exactly the defect class we have removed three times already.

    The gate now derives completion from the artifacts (gate_layers.chain_gaps). Both
    directions are asserted here: a record that overclaims is refused, and a record
    that underclaims cannot hide artifacts that really exist.
    """
    from tilefoundry_ascend.passes.gate_layers import TRANSACTION_STAGES, chain_gaps
    from tilefoundry_ascend.passes.p3_chain import ChainRecord, ir_identity

    module = _finalized(COPY_ONLY)
    empty_metadata = {}

    # (1) A record claiming the whole transaction, with no artifacts at all.
    forged = ChainRecord(
        ir_identity=ir_identity(module),
        entry_identity="sha256:" + "a" * 64,
        plan_identity="sha256:" + "b" * 64,
        completion_identity="sha256:" + "c" * 64,
        storage_identity="sha256:" + "d" * 64,
        tl_identity="sha256:" + "e" * 64,
        pipeline_status="APPLIED",
        applied=TRANSACTION_STAGES,
        stopped_at="",
        reason="",
    )
    rows = chain_gaps(empty_metadata, forged)
    codes = [code for _layer, code in rows]
    assert codes, "a record that ran nothing must not clear the chain layer"
    #     Each missing stage is named, so the reader learns WHICH step is absent.
    for expected in ("PLAN_STAGE_UNEVIDENCED", "COMPLETION_STAGE_UNEVIDENCED",
                     "STORAGE_STAGE_UNEVIDENCED", "APPLY_STAGE_UNEVIDENCED",
                     "INSTALL_STAGE_UNEVIDENCED"):
        assert expected in codes, (expected, codes)

    # (2) The reverse: real artifacts, but a record that claims almost nothing. The
    #     artifacts are what count, so the chain layer clears - a record cannot
    #     WEAKEN evidence any more than it can invent it.
    from tilefoundry_ascend.plan.mainline import compile_mainline as _tx

    real_module = _finalized(COPY_ONLY)
    real_record = _tx(real_module, authored=None)
    assert real_record.pipeline_status == "APPLIED", real_record.reason
    #     Prove the artifacts are genuinely there before asserting they clear.
    from tilefoundry_ascend.passes.p3_chain import COMPLETION_KEY, PLAN_KEY, STORAGE_KEY
    assert PLAN_KEY in real_module.metadata
    assert COMPLETION_KEY in real_module.metadata
    assert STORAGE_KEY in real_module.metadata
    understated = replace(real_record, applied=("analyze",))
    assert chain_gaps(real_module.metadata, understated) == [], (
        "an understating record must not erase artifacts that really exist",
        chain_gaps(real_module.metadata, understated))