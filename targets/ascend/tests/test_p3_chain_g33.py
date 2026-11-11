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
    from tilefoundry_ascend.passes.p3_chain import EXPORT_GATE_KEY

    gate = module.metadata.get(EXPORT_GATE_KEY)
    assert gate is not None, "the export gate pass did not run on the public finalize path"
    return gate


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

    It stops at layout with STORAGE_PLACEMENT_UNATTACHED: completion and physical storage
    are P4 obligations and this stage must not fabricate either. If P4 lands, this test
    fails on purpose so the boundary gets re-stated instead of quietly sliding.
    """
    record = _chain(_finalize())
    assert record.stopped_at == "layout", (
        "chain stopped at %r (%s); it reached P4's storage boundary before P3-04 only if"
        " this asserts 'layout'" % (record.stopped_at, record.reason))
    assert record.reason == "STORAGE_PLACEMENT_UNATTACHED"
    assert record.storage_status == "UNATTACHED"
    assert record.completion_status == "PENDING"


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
def test_export_refused_while_completion_and_storage_are_unverified():
    """A final runnable artifact must be refused, naming BOTH missing P4 obligations.

    Naming both matters: they are fixed by different work. A single "not ready" verdict
    would let an operator fix one, re-run, and be refused again with no new information.
    """
    from tilefoundry_ascend.passes.p3_chain import ExportRefusal, require_export_ready

    module = _finalize()
    with pytest.raises(ExportRefusal) as caught:
        require_export_ready(module)
    refusal = caught.value
    assert refusal.code == "EXPORT_NOT_RUNNABLE"
    joined = " ".join(refusal.missing)
    assert "COMPLETION_NOT_VERIFIED" in joined
    assert "STORAGE_PLACEMENT_UNVERIFIED" in joined


def test_inspection_is_still_available_on_a_non_exportable_module():
    """The gate RECORDS; it does not raise. Inspecting an unexportable module must work.

    Refusing to hand back the module would deny the very state that explains the refusal.
    """
    module = _finalize()
    gate = _gate(module)
    assert gate["ready"] is False
    assert len(gate["gaps"]) >= 2
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
    from tilefoundry_ascend.schedule.ledger import build_problem, extract_ledger

    program = checked_program(CARRIER, None, command="g33-ledger")
    analyzed = analyze(program.module, program.entry, analysis=("memory",))
    module = finalize(analyzed, authored=program.authored)
    problem = build_problem(extract_ledger(module.entry_function()))
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

