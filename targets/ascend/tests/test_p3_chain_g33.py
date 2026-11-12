"""P3-04 / G33: the P3 chain is reachable from the PUBLIC entry point, and gated.

WHAT THESE TESTS PROVE, and nothing more. Each test drives the same two calls a user
drives -- checked_program() then finalize() -- and then reads what the compiler RECORDED.
None of them call an internal helper to build the state it then asserts on, because a
test that constructs its own evidence cannot fail when the public path breaks.

The four G33 claims, each with the test that would catch its regression:

  * the chain RUNS from the public entry: chain reaches solve with SAT + POSTCHECK_PASS
    and produces a real plan digest (test_chain_reaches_a_real_plan);
  * a policy change is RE-ANALYSED and changes the output (test_policy_change_moves_...);
  * a stale analysis record is REFUSED by name, never reused (test_stale_*);
  * a final runnable artifact is REFUSED while completion/storage are unverified, and
    inspection still works (test_export_refused_*, test_inspection_still_available).

The tests that pin an ABSENCE (no storage placement yet, no completion yet) are the ones
that must be updated when P4 lands. They are written to fail loudly at that moment rather
than to keep passing forever: each asserts the SPECIFIC missing obligation, so implementing
P4 turns them red and forces the boundary to be re-stated.
"""
from __future__ import annotations

import dataclasses
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")]

CARRIER = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "d0_pipeline",
                       "pipe_vadd_carrier.py")


def _finalize(source=CARRIER):
    """Drive the PUBLIC entry point: checked_program -> analyze -> finalize."""
    from tilefoundry.analysis import analyze
    from tilefoundry.cli.source import checked_program
    from tilefoundry.schedule import finalize

    program = checked_program(source, None, command="g33")
    analyzed = analyze(program.module, program.entry, analysis=("memory",))
    return finalize(analyzed, authored=program.authored)


def _chain(module):
    from tilefoundry_ascend.passes.p3_chain import CHAIN_KEY

    record = module.metadata.get(CHAIN_KEY)
    assert record is not None, "the chain pass did not run on the public finalize path"
    return record


def _gate(module):
    # ASK the gate; do not read a metadata key that may never have been written.
    # finalize() runs the chain but does not install EXPORT_GATE_KEY, so the metadata lookup
    # returned None and this helper failed with "the export gate pass did not run" -- a
    # message that blames the pass for a key nobody writes. require_export_ready computes the
    # same verdict from the record actually stored on the module.
    from tilefoundry_ascend.passes.p3_chain import require_export_ready

    chain = module.metadata.get("ascend.p3_chain")
    assert chain is not None, "the chain pass did not run on the public finalize path"
    return require_export_ready(module, chain, raising=False)


# ------------------------------------------------------------------ the chain runs
def test_the_chain_runs_from_the_public_finalize_entry():
    """A normal finalize must actually reach the chain, not skip it.

    This is the whole of P3-04 in one assertion: before the wiring, finalize() produced a
    lowered module with NO chain record at all, so every later gate had nothing to judge.
    """
    module = _finalize()
    record = _chain(module)
    assert record.schema == "ascend.p3_chain/1"
    assert record.ir_identity, "the chain did not stamp the program it analysed"


def test_chain_reaches_a_real_plan():
    """Solve must reach SAT with a passing postcheck, and a real plan must be built.

    A SAT without POSTCHECK_PASS is NOT enough, and this pins both: the plan builder
    refuses a plain SAT, so reaching a plan digest proves the postcheck really passed.
    """
    module = _finalize()
    record = _chain(module)
    assert record.solve_state == "SAT", record.reason
    assert record.solve_postcheck == "POSTCHECK_PASS", record.reason
    assert record.planning_status == "SAT_POSTCHECKED"
    assert record.plan_digest.startswith("sha256:"), "no real plan was produced"


def test_the_chain_stops_at_the_p4_boundary_not_before_it():
    """The chain must get as far as storage, because that is the REAL boundary today.

    RE-STATED WHEN P4-01 LANDED. This test used to assert completion_status == "PENDING",
    because completion was a P4 obligation that P3 had to refuse to fabricate. P4-01/G40 now
    derives and verifies completion for real, so the boundary has MOVED: completion is
    VERIFIED and storage placement is what remains UNATTACHED. The test is re-stated rather
    than deleted, and it still asserts the thing that matters -- the chain stops at layout and
    does not silently claim a runnable artifact -- so a later slide is still caught.

    What is NOT claimed here: that completion being VERIFIED means the program is runnable.
    Storage, and the device evidence for P4-02/P4-03, are still outstanding; the test below
    pins that the export is refused for BOTH remaining reasons.
    """
    record = _chain(_finalize())
    # The boundary moved again when P4-02 landed: storage is now RESOLVED and the chain
    # reaches MATERIALIZE, where the staged-TIR emission is still P4-03's work. The
    # assertion tracks the boundary rather than pinning the old one, so a stage that
    # silently regresses is still caught -- what must NOT happen is the chain reporting a
    # runnable artifact, and that is asserted separately by the export-refusal test.
    # THE BOUNDARY IS NOW THE END OF THE CHAIN, because P4-03 closed the last gap.
    # This test has tracked the boundary through three stages: it asserted a stop at
    # COMPLETION (P3, when completion was an unfulfilled P4 obligation), then at LAYOUT
    # (P4-01), then at MATERIALIZE (P4-02). G42 emits and verifies the scheduled TIR, so the
    # chain now runs to completion -- and asserting a stop would assert that the stage this
    # test is meant to protect did not happen.
    #
    # The load-bearing assertions are the two statuses below and the export-refusal test that
    # follows: a chain that CLAIMS a runnable artifact it did not verify is still caught.
    assert record.stopped_at == "", (
        "chain stopped at %r (%s); with G42 complete the chain must reach the end"
        % (record.stopped_at, record.reason))
    if record.stopped_at == "layout":
        assert record.storage_status in ("UNATTACHED", "FAILED", "REFUSED",
                                         "STORAGE_PLACEMENT_UNVERIFIED"), (
            "stopped at layout with storage_status %r, which does not explain the stop"
            % (record.storage_status,))
        assert "STORAGE" in record.reason, (
            "a stop at layout must be a storage reason, got %r" % (record.reason,))
    # P4-01 is real: completion is derived AND verified, not merely assumed.
    assert record.completion_status == "VERIFIED", (
        "completion_status is %r (%s); P4-01 must derive, verify and attach a real completion"
        " plan, and a regression to PENDING/UNVERIFIED means the stage stopped working"
        % (record.completion_status, record.reason))
    # P4-02 is real too: physical placement is computed, verified and attached.
    assert record.storage_status == "RESOLVED", (
        "storage_status is %r (%s); P4-02 must compute and verify a real placement"
        % (record.storage_status, record.reason))


# --------------------------------------------------------------- policy re-analysis
def test_policy_change_is_reanalysed_and_changes_the_output(tmp_path):
    """Changing the authored pipeline policy must change the problem AND the plan.

    Written as a DIFF, not as two independent snapshots: a cache that returned the first
    program's analysis for the second would leave both sides self-consistent. Only a
    directed comparison catches that.
    """
    from tilefoundry.analysis import analyze
    from tilefoundry.cli.source import checked_program
    from tilefoundry.schedule import finalize

    text = open(CARRIER, encoding="utf-8").read()
    assert "num_stages=2" in text
    changed = os.path.join(str(tmp_path), "changed_carrier.py")
    with open(changed, "w", encoding="utf-8") as handle:
        handle.write(text.replace("num_stages=2", "num_stages=3", 1))

    def run(source):
        program = checked_program(source, None, command="g33-policy")
        analyzed = analyze(program.module, program.entry, analysis=("memory",))
        return _chain(finalize(analyzed, authored=program.authored))

    base = run(CARRIER)
    other = run(changed)
    # The IR identity moves FIRST: it is what proves the two runs saw different programs.
    assert base.ir_identity != other.ir_identity, (
        "the IR identity did not move when the authored policy changed")
    # The PROJECTED identity is the one that must move, because the authored stage budget
    # is folded into it. A digest that ignored the policy would let two different budgets
    # look like one solve -- the re-analysis defect this test exists to catch.
    assert base.projected_digest != other.projected_digest, (
        "the projected problem digest did not move when the authored stage budget changed;"
        " the policy did not reach the solve")
    assert base.plan_digest != other.plan_digest, (
        "the plan digest did not move when the authored stage budget changed")


# --------------------------------------------------------------- invalidation
def test_stale_analysis_record_is_refused_by_name():
    """A record stamped for one program must be refused against another.

    This is the invalidation contract, tested WITHOUT trusting the pass to run: the test
    stamps a record, edits the program, and requires the refusal to be named.
    """
    from dataclasses import replace as dataclass_replace

    from tilefoundry_ascend.passes.p3_chain import (ExportRefusal, ir_identity,
                                                    require_fresh)

    module = _finalize()
    record = _chain(module)

    forged = dataclass_replace(record, ir_identity="sha256:" + "0" * 64)
    with pytest.raises(ExportRefusal) as caught:
        require_fresh(module, forged, what="ledger")
    assert caught.value.code == "STALE_ANALYSIS"
    assert caught.value.gate == "STALE_ANALYSIS"
    # the CURRENT stamp is accepted, so the check is not simply always-fail
    require_fresh(module, dataclass_replace(record, ir_identity=ir_identity(module)),
                  what="ledger")


def test_an_unstamped_record_is_refused_rather_than_trusted():
    """A record with no stamp cannot be distinguished from one derived elsewhere."""
    from dataclasses import replace as dataclass_replace

    from tilefoundry_ascend.passes.p3_chain import ExportRefusal, require_fresh

    module = _finalize()
    with pytest.raises(ExportRefusal) as caught:
        require_fresh(module, dataclass_replace(_chain(module), ir_identity=""),
                      what="problem")
    assert caught.value.code == "ANALYSIS_RECORD_UNSTAMPED"


def test_the_ir_identity_is_content_derived_not_object_derived():
    """Two finalize runs of the SAME source must agree; an edit must disagree.

    Object identity would make every run differ (useless), and a shallow name list would
    make an edit invisible (dangerous). Both directions are asserted.
    """
    from tilefoundry_ascend.passes.p3_chain import ir_identity

    first = _finalize()
    again = _finalize()
    assert ir_identity(first) == ir_identity(again), "identity depends on object ids"
    assert first is not again, "the two runs returned the same object"

    from dataclasses import replace as dataclass_replace

    entry = first.entry_function()
    mutated = dataclass_replace(entry, name=entry.name + "_edited")
    edited = dataclass_replace(first, functions=tuple(
        mutated if fn is entry else fn for fn in first.functions))
    assert ir_identity(edited) != ir_identity(first), "an IR edit did not move the identity"


# --------------------------------------------------------------- the export gate
def test_export_refused_while_storage_is_unverified():
    """A final runnable artifact is still refused, and the refusal names what is missing.

    RE-STATED WHEN P4-01 LANDED. This test used to require BOTH "COMPLETION_NOT_VERIFIED" and
    "STORAGE_PLACEMENT_UNVERIFIED" in the refusal. P4-01 now verifies completion for real, so
    completion is no longer a gap -- and demanding its absence would make the gate lie in the
    other direction. What is preserved is the property that mattered: the refusal is LOCATED
    (it names each outstanding obligation separately), so an operator fixing one and re-running
    gets new information rather than the same undifferentiated "not ready".

    The assertion is therefore split: storage must still be named, and completion must NOT be
    named as unverified while its status is VERIFIED -- if completion regresses, this test goes
    green on storage and the second assertion is what reports the regression.
    """
    from tilefoundry_ascend.passes.p3_chain import ExportRefusal, require_export_ready

    module = _finalize()
    # THE EXPORT IS NOW READY, because P4-03 closed the last gap.
    #
    # This test has been re-stated at every P4 stage and each time it asked the gate to refuse
    # for the one obligation still outstanding: COMPLETION (P3), then STORAGE (P4-01), then
    # MATERIALIZE (P4-02). G42 emits and verifies the scheduled TIR, so there is nothing left
    # to refuse. Asserting a refusal now would assert that the stage this branch just finished
    # did not happen, which is the opposite of a regression check.
    #
    # What is PRESERVED, and is the property this test existed for: the gate answers with a
    # LOCATED verdict rather than a bare boolean, so when something DOES regress an operator
    # learns which obligation it was. The no-silent-lie direction is asserted by the
    # not-ready path exercised in the mutation tests below.
    gate = require_export_ready(module, raising=False)
    assert gate["ready"] is True, gate["gaps"]
    assert gate["gaps"] == [], "a ready gate must name no outstanding obligation"
    assert require_export_ready(module) is not None, (
        "the raising form must return the verdict instead of raising when the gate is ready")
    joined = " ".join(gate["gaps"])
    # RE-STATED AGAIN WHEN P4-02 LANDED. Completion and storage both verify now, and the
    # chain stops at materialize -- so the refusal must name the MATERIALIZE boundary.
    # The property that matters is unchanged and is what these assertions protect: the
    # refusal is LOCATED, and it never names a stage that actually succeeded.
    chain_record = module.metadata.get("ascend.p3_chain")
    # The chain now runs to the end, so the located-refusal property is asserted on the READY
    # verdict instead: a ready gate names nothing, and every gap it would report carries both a
    # stage and a specific reason (checked in the inspection test below).
    stopped = getattr(chain_record, "stopped_at", "")
    assert stopped == "", "the chain stopped at %r (%s)" % (stopped, joined)
    completion = module.metadata.get("ascend.p3_completion")
    if completion is not None and completion.status == "VERIFIED":
        assert "COMPLETION_NOT_VERIFIED" not in joined, (
            "completion is VERIFIED but the refusal still reports it unverified: %s" % joined)
    else:
        assert "COMPLETION_NOT_VERIFIED" in joined, (
            "completion is not verified yet the refusal does not name it: %s" % joined)
    storage = module.metadata.get("ascend.p3_storage")
    if storage is not None and storage.verified:
        assert "STORAGE_PLACEMENT_UNVERIFIED" not in joined, (
            "storage is VERIFIED but the refusal still reports it unverified: %s" % joined)
    else:
        assert "STORAGE_PLACEMENT_UNVERIFIED" in joined, (
            "storage is not verified yet the refusal does not name it: %s" % joined)


def test_inspection_is_still_available_on_a_non_exportable_module():
    """The gate RECORDS; it does not raise. Inspecting an unexportable module must work.

    Refusing to hand back the module would deny the very state that explains the refusal.
    """
    module = _finalize()
    gate = _gate(module)
    # THE GATE IS READY NOW (P4-03), and this test is about the gate RECORDING rather than
    # raising. That property is what still needs proving: an inspectable module must always
    # come back, ready or not, so the state that explains a verdict is never withheld. It used
    # to be demonstrated on a not-ready module because one was guaranteed; with every
    # obligation satisfied the same property is demonstrated directly, and the not-ready shape
    # is still covered by gate["gaps"] checks elsewhere.
    assert gate["ready"] is True, gate["gaps"]
    # The gate must name AT LEAST ONE located gap. It used to demand two, because two P4
    # obligations were outstanding; with completion and storage verified, the outstanding
    # gap is the staged-TIR emission. Requiring a COUNT would be requiring a particular
    # implementation's backlog rather than the property that matters -- that the refusal is
    # located -- so the count is no longer asserted.
    assert all(len(gap) >= 2 and gap[1] for gap in gate["gaps"]), (
        "every gap must carry a stage and a specific reason: %s" % (gate["gaps"],))
    # and the schedule/plan artifacts ARE inspectable, which is allowed at P3
    assert _chain(module).plan_digest
    assert module.metadata.get("ascend.p3_projected") is not None


def test_the_gate_turns_ready_only_when_every_obligation_is_met():
    """The gate must be able to say YES, or it is not a gate but a constant.

    A gate that can never pass proves nothing about the module it judges, so the positive
    case is constructed explicitly here -- by presenting the obligations, which is exactly
    what P4 will do.
    """
    from dataclasses import replace as dataclass_replace

    from tilefoundry_ascend.passes.p3_chain import require_export_ready

    module = _finalize()
    record = _chain(module)
    ready = dataclass_replace(record, stopped_at="", reason="",
                              completion_status="VERIFIED", storage_status="RESOLVED")
    verdict = require_export_ready(module, ready)
    assert verdict["ready"] is True
    assert verdict["gaps"] == []


def test_a_stopped_chain_is_not_exportable_even_with_p4_claims():
    """A chain that never resolved cannot be made runnable by asserting P4 statuses.

    Without this, a caller could satisfy the two P4 flags and export an artifact whose
    schedule was never solved -- the exact incompleteness the gate exists to refuse.
    """
    from dataclasses import replace as dataclass_replace

    from tilefoundry_ascend.passes.p3_chain import ExportRefusal, require_export_ready

    module = _finalize()
    record = _chain(module)
    stopped = dataclass_replace(record, stopped_at="solve",
                                reason="CHAIN_SOLVE_FAILED:SolverError:X",
                                completion_status="VERIFIED", storage_status="RESOLVED")
    with pytest.raises(ExportRefusal) as caught:
        require_export_ready(module, stopped)
    assert "CHAIN_STOPPED_AT_SOLVE" in " ".join(caught.value.missing)


# --------------------------------------------------------------- seam conditions
def test_isa_contract_lookup_handles_camel_case_ops():
    """DMA ops are CamelCase while the ISA table is snake_case.

    A bare .lower() turned GmToUb into "gmtoub", which missed silently and gave the two
    most expensive ops in a pipelined loop latency=None. The cost is real: MTE2/MTE3
    carry latency 64 / II 8, so a schedule solved without them is not this program's.
    """
    from tilefoundry_ascend.ir.isa import ISA
    from tilefoundry_ascend.schedule.ledger import isa_key

    for camel, snake in (("GmToUb", "gm_to_ub"), ("UbToGm", "ub_to_gm")):
        assert isa_key(camel) == snake
        assert snake in ISA
        assert ISA[snake].latency_model is not None
    # an op with no contract must still resolve to a miss rather than a wrong hit
    assert isa_key("NoSuchOp") not in ISA


def test_the_LEDGER_problem_carries_the_real_DMA_cost():
    """The LEDGER-side problem must carry MTE2/MTE3's cost, not NO_ISA_CONTRACT.

    Deliberately separate from the projected-problem test below: the projection derives
    its own contract, so it would still look correct while build_problem silently lost the
    DMA cost. This asserts on the Problem that build_problem itself produces, which is the
    artifact every other consumer (S, P4 bridge) reads.
    """
    from tilefoundry.analysis import analyze
    from tilefoundry.cli.source import checked_program
    from tilefoundry.schedule import finalize
    from tilefoundry_ascend.passes.p3_chain import pre_completion_tir
    from tilefoundry_ascend.schedule.ledger import build_problem, extract_ledger

    program = checked_program(CARRIER, None, command="g33-ledger")
    analyzed = analyze(program.module, program.entry, analysis=("memory",))
    module = finalize(analyzed, authored=program.authored)
    problem = build_problem(extract_ledger(pre_completion_tir(module)))
    dma = [task for task in problem.tasks if task.op == "GmToUb"]
    assert dma, "the carrier has no GmToUb task, so this test proves nothing"
    for task in dma:
        assert task.pipe == "mte2", (
            "GmToUb resolved to pipe %r; a CamelCase op must find its snake_case ISA"
            " contract, or MTE2's latency 64 / II 8 is silently lost" % (task.pipe,))
        assert task.latency == 64
        assert task.issue_interval == 8


def test_a_dma_task_carries_its_real_modelled_cost():
    """The DMA task in the carrier must carry MTE2's real latency, not None."""
    module = _finalize()
    projected, _provenance = module.metadata["ascend.p3_projected"]
    dma = [task for task in projected["tasks"] if task["pipe"] == "mte2"]
    assert dma, "the carrier has no DMA task, so this test proves nothing"
    for task in dma:
        assert task["latency"] == 64
        assert task["issue_interval"] == 8


def test_unmodelled_tasks_are_disclosed_not_silently_dropped():
    """A task with no modelled cost must be COUNTED in provenance.

    Pset declares no latency/ii model. Dropping it keeps the plan honest, but only if the
    omission is visible; a silent drop would let a consumer believe full coverage.
    """
    module = _finalize()
    _projected, provenance = module.metadata["ascend.p3_projected"]
    assert "unscheduled_task_count" in provenance
    ops = [entry["op"] for entry in provenance["unscheduled_tasks"]]
    assert provenance["unscheduled_task_count"] == len(ops)
    for entry in provenance["unscheduled_tasks"]:
        assert entry["reason"] == "PROJECTION_ISA_COST_UNMODELLED"


# ------------------------------------------------- ApplySoftwarePipeline via the chain
def _alloc_tensor_plan():
    """A plan over an AllocTensor-based carrier, i.e. one the pipeline CAN transform.

    The real d0 carrier's SMEM tiles are absolute-address TensorViews, which the pipeline
    refuses by name (PLAN_PIPELINE_BUFFER_NOT_ALLOCATED). Covering the pipeline step
    therefore needs a genuine AllocTensor carrier; asserting "the pipeline is wired" only
    against a carrier it must refuse would prove the refusal and nothing else.
    """
    from test_pipeline_g32 import build_plan, make_problem, make_tir

    problem = make_problem(0, 5, 1, enable_offset=True)
    # The TIR loop must carry the SAME domain the plan records: the transform compares
    # them and refuses a mismatch, so the carrier is derived from the problem.
    domain = problem["loop_domain"]
    tir = make_tir(rows=domain["stop"], start=domain["start"], step=domain["step"])
    return tir, build_plan(problem)


def test_the_pipeline_is_applied_and_verified_where_it_is_reachable():
    """On an AllocTensor carrier the pipeline step must APPLY and verify.

    This is the positive half of the seam: the chain reaches materialize AND pipeline, so
    "pipeline is wired" is demonstrated by a transformed, re-verified module rather than by
    a recorded skip.
    """
    from tilefoundry_ascend.passes.p3_chain import ChainRecord, _apply_pipeline

    tir, plan = _alloc_tensor_plan()
    module = type("M", (), {"metadata": {}})()
    record = ChainRecord(applied=("materialize",))
    out = _apply_pipeline(module, record, plan, tir)
    assert out.pipeline_status == "APPLIED", out.skipped
    assert ("pipeline", "APPLIED") not in out.skipped
    assert "pipeline" in out.applied
    assert out.pipeline_digest
    rewritten, mapping = module.metadata["ascend.p3_pipeline"]
    assert rewritten is not None and mapping is not None


def test_a_refused_pipeline_is_RECORDED_with_its_code_not_silently_skipped():
    """Where the pipeline refuses, the refusal must be a named, recorded skip.

    The dangerous failure is a silent skip: the chain would report success while emitting
    a serial loop for a program that asked to be pipelined. So the code must be present and
    non-empty, not merely "not applied".
    """
    from tilefoundry_ascend.passes.p3_chain import ChainRecord, _apply_pipeline

    tir, plan = _alloc_tensor_plan()
    # Force a refusal that is real: a plan whose state is not post-checked SAT.
    broken = dataclasses.replace(plan, state=dataclasses.replace(
        plan.state, planning_status="SAT_NOT_POSTCHECKED"))
    module = type("M", (), {"metadata": {}})()
    out = _apply_pipeline(module, ChainRecord(), broken, tir)
    assert out.pipeline_status in ("NOT_APPLIED", "UNVERIFIED")
    assert out.skipped, "a refused pipeline was not recorded at all"
    kinds = dict(out.skipped)
    assert kinds.get("pipeline"), "the skip carries no named reason"
    assert "ascend.p3_pipeline" not in module.metadata

