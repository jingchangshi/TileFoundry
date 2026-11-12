"""S0 entry: a real authored Ascend program through the ONE normal entry.

The fixture under fixtures/authors is authored source: a GM tensor accumulation
whose loop declares tf.pipelined and whose body is one whole T.SimdVF execution
region doing a real GM load, predicated vector add and UB store back into the
loop-carried slice. This test drives the normal entry as written --
checked_program -> analyze -> finalize -- and reads the fail-closed records the
target's own pass leaves on the lowered module. Nothing here builds TIR by hand,
skips a stage, or relaxes a refusal to make the chain look complete.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tilefoundry.analysis import analyze
from tilefoundry.cli.source import checked_program, load_authored_ir
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Var
from tilefoundry.ir.core.op import Op
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.pipeline import PipelineSpec
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, ExecutionScope, For, Sequential
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.utils import make_tensor_type, static_dim_value
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.schedule import finalize
from tilefoundry_ascend.codegen.abi import abi_from_module
from tilefoundry_ascend.ir.execution import SIMD_VF_KIND
from tilefoundry_ascend.verify.executor import ExecutionRefusal, run_entry
from tilefoundry_ascend.passes.p3_chain import (
    CHAIN_KEY,
    EXPORT_GATE_KEY,
    ChainRecord,
    ExportRefusal,
    ir_identity,
    require_export_ready,
)

AUTHORS = Path(__file__).resolve().parent / "fixtures" / "authors"
SOURCE = "%s:MinimalAivPipeline" % (AUTHORS / "minimal_aiv.py",)
ENTRY = "vsum"
LANES = 64
STEPS = 4


def _walk(node):
    yield node
    if isinstance(node, (tuple, list)):
        for item in node:
            yield from _walk(item)
        return
    for field in getattr(node, "__dataclass_fields__", ()):
        if field in ("type", "metadata", "origin"):
            continue
        yield from _walk(getattr(node, field))


def _loop_of(function):
    loops = [expr for expr in collect_exprs(function.body) if isinstance(expr, LoopRegion)]
    assert len(loops) == 1, "the authored source must declare exactly one loop"
    return loops[0]


def _op_names(node):
    return [type(item.callable).__name__ for item in _walk(node) if isinstance(item, Evaluate)]


def _run_normal_entry():
    program = checked_program(SOURCE, None, command="test normal entry")
    analyzed = analyze(program.module, program.entry, analysis=("compute-cost", "memory"))
    module = finalize(analyzed, authored=program.authored)
    return program, analyzed, module


def test_authored_pipeline_declaration_survives_parse():
    """The authored tf.pipelined is one typed spec on the real loop, not a string."""
    program = checked_program(SOURCE, None, command="test normal entry")
    loop = _loop_of(program.authored)
    assert isinstance(loop.pipeline, PipelineSpec)
    assert (loop.pipeline.max_stage, loop.pipeline.max_buffer_versions) == (2, 2)
    assert loop.pipeline.fixed_ii is None and loop.pipeline.enable_offset is False
    assert (static_dim_value(loop.start), static_dim_value(loop.extent),
            static_dim_value(loop.step)) == (0, STEPS, 1)
    assert len(loop.yield_values) == 1, "the accumulation must be loop-carried, not a dead loop"
    assert type(loop.body).__name__ == "ExecutionRegion"


def test_normal_entry_lowers_the_pipelined_aiv_kernel():
    """finalize runs, the entry is a real PrimFunction, and the compute is still there."""
    program, analyzed, module = _run_normal_entry()
    assert analyzed.analyses == ("compute-cost", "memory")
    assert set(analyzed.executed) >= {"compute-cost", "memory"}

    entry = module.entry_function()
    assert isinstance(entry, PrimFunction)
    assert entry.name == ENTRY
    assert entry.output_count == 1
    params = tuple(entry.params)
    assert [parameter.name for parameter in params] == ["x", "out"]

    abi = abi_from_module(module)
    assert abi.entry == ENTRY
    assert abi.output_count == entry.output_count
    assert abi.output_split == len(params) - entry.output_count
    assert [spec.name for spec in abi.inputs] == ["x"]
    assert [spec.name for spec in abi.outputs] == ["out"]
    assert abi.outputs[0].role == "tensor" and abi.outputs[0].storage == "gmem"
    assert tuple(abi.outputs[0].shape) == (LANES,)

    loops = [node for node in _walk(entry.body) if isinstance(node, For)]
    assert len(loops) == 1, "one pipelined loop must lower to one TIR For"
    lowered = loops[0]
    assert lowered.pipeline == _loop_of(program.authored).pipeline
    assert (static_dim_value(lowered.start), static_dim_value(lowered.stop),
            static_dim_value(lowered.step)) == (0, STEPS, 1)

    scopes = [node for node in _walk(entry.body) if isinstance(node, ExecutionScope)]
    assert [scope.kind for scope in scopes] == [SIMD_VF_KIND]
    # The region computes five real inside-only instructions; the per-iteration
    # GM load is emitted before the scope and the initial load/final store outside it.
    assert _op_names(scopes[0].body) == ["Vld", "Vld", "Pset", "Vadd", "Vsts"]
    assert _op_names(lowered.body) == ["GmToUb", "Vld", "Vld", "Pset", "Vadd", "Vsts"]
    assert _op_names(entry.body) == [
        "GmToUb", "GmToUb", "Vld", "Vld", "Pset", "Vadd", "Vsts", "UbToGm",
    ]
    # The last store must write the ENTRY's trailing output parameter itself, so a
    # store into some other GM tensor cannot keep this test green.
    stores = [item for item in _walk(entry.body)
              if isinstance(item, Evaluate) and type(item.callable).__name__ == "UbToGm"]
    assert len(stores) == 1
    assert stores[0].args[-1] is entry.params[-1]


def test_printer_restores_the_authored_pipeline_surface(tmp_path):
    """The printer emits the pipelined/SimdVF surface and re-parses to the same program."""
    _, _, module = _run_normal_entry()
    printed = as_script(module)
    assert "tf.pipelined(4, max_stage=2, max_buffer_versions=2" in printed
    assert "with T.SimdVF():" in printed
    assert "T.ascend_gm_to_ub(" in printed and "T.ascend_vadd(" in printed

    path = tmp_path / "roundtrip.py"
    path.write_text(printed, encoding="utf-8")
    again = load_authored_ir("%s:MinimalAivPipeline" % (path,))
    assert as_script(again) == printed


def test_chain_stops_at_a_named_stage_with_the_stages_it_completed():
    """The mainline is refused at a NAMED stage, not by an escaping exception.

    compile_mainline used to raise MAINLINE_CHAIN_NOT_IMPLEMENTED before doing
    anything, so the record said stopped_at="freeze" - the name of "nothing
    started". It now walks the real stages and stops at the first refusal, which
    for this fixture is resolve: analyze completes, then the plan is refused with
    PLAN_DEPENDENCE_PROOF_UNAVAILABLE because minimal_aiv carries a loop region
    whose distance needs a generation delta this slice does not derive (task-46).

    The assertion is therefore STRICTER than before, not looser: it names the
    stage AND checks that the stages before it really ran, so a regression back to
    "nothing started" fails here instead of passing.
    """
    _, _, module = _run_normal_entry()
    record = module.metadata.get(CHAIN_KEY)
    assert isinstance(record, ChainRecord)
    assert record.stopped_at == "resolve", record.stopped_at
    assert record.reason == "PLAN_DEPENDENCE_PROOF_UNAVAILABLE", record.reason
    assert record.pipeline_status == "REFUSED"
    # Real work happened: an empty `applied` would mean we are back to freeze.
    assert record.applied == ("analyze",), record.applied
    assert record.ir_identity == ir_identity(module)


# The gate's named gaps, as the layered checks actually report them today.
# ("tl", "NATIVE_EXPORT_VERIFICATION_NOT_IMPLEMENTED") used to be here: it was
# one unconditional byte appended after careful checks, so every refusal looked
# the same and no layer could be narrowed as it landed. passes/gate_layers.py
# replaced it with per-layer verdicts drawn from real objects, which is why the
# set below is longer and each entry is separately attributable.
# Measured on THIS fixture after compile_mainline started really running:
# the chain stops in resolve, so the record carries that stage and its code.
EXPECTED_GAPS = {
    ("chain", "PLAN_DEPENDENCE_PROOF_UNAVAILABLE"),
    ("chain", "TRANSACTION_INCOMPLETE"),
    ("chain", "CHAIN_STOPPED_AT_RESOLVE"),

    ("plan", "PLAN_ARTIFACT_MISSING"),
    ("storage", "STORAGE_ARTIFACT_MISSING"),
    ("completion", "COMPLETION_ARTIFACT_MISSING"),
    ("install", "INSTALL_CANDIDATE_UNVERIFIED"),
    ("binary", "COMPILE_SITE_NOT_RUN"),
    ("tl", "TL_ARTIFACT_NOT_ACTUAL"),
}
# Every layer that has not landed must still be named: the set is closed under
# the layers the gate owns, so removing a check cannot silently shrink the
# refusal into "almost ready".
# The layers this fixture is EXPECTED to report today. "entry" is absent on
# purpose: compile_mainline now records the real entry identity, so there is no
# mismatch to report. The layer is still implemented and still exercised - a
# tampered entry produces ("entry", "ENTRY_IDENTITY_MISMATCH") in
# test_s0_gate_positive.py - so the closed-set check below stays honest without
# demanding a gap that no longer exists.
GAP_LAYERS = {"chain", "plan", "storage", "completion", "install", "binary", "tl"}


def test_export_gate_names_every_missing_layer():
    """Still refused, with each missing layer NAMED - not one blanket byte.

    Split from the top-level test on purpose (lead ruling): "every missing layer
    is named" and "the top-level refusal shape is unchanged" are two different
    properties, and merging them into one lenient assertion is how a layer gets
    silently dropped. This half keeps the >= semantics: the measured gaps must be
    a SUPERSET of the table below, and the set of reporting layers must be exactly
    GAP_LAYERS, so neither a lost check nor a new unlisted layer passes.
    """
    _, _, module = _run_normal_entry()
    record = module.metadata.get(CHAIN_KEY)
    gate = module.metadata.get(EXPORT_GATE_KEY)
    assert gate["ready"] is False
    assert gate["pipeline"] == "REFUSED"
    gaps = {tuple(gap) for gap in gate["gaps"]}
    assert gaps >= EXPECTED_GAPS, sorted(EXPECTED_GAPS - gaps)
    assert {layer for layer, _ in gaps} == GAP_LAYERS, sorted(gaps)


def test_export_gate_top_level_refusal_shape_is_unchanged():
    """The raising path still reports EXPORT_NOT_RUNNABLE plus every named code.

    The cached gate, a fresh recomputation and the raising refusal must agree -
    that is the property this half keeps. The gate can only ever be narrowed by
    satisfying a layer, never by weakening this envelope.
    """
    _, _, module = _run_normal_entry()
    record = module.metadata.get(CHAIN_KEY)
    gate = module.metadata.get(EXPORT_GATE_KEY)
    assert require_export_ready(module, record, raising=False) == gate
    with pytest.raises(ExportRefusal) as caught:
        require_export_ready(module, record, raising=True)
    refusal = caught.value
    assert refusal.code == "EXPORT_NOT_RUNNABLE"
    # `gate` names the layer of the FIRST reported gap. Which layer that is depends
    # on the order the checks run, not on any contract - asserting a specific name
    # would pin an accident. What matters is that it names a REAL layer from the
    # same set the gaps report.
    assert refusal.gate in GAP_LAYERS, refusal.gate
    assert set(refusal.missing) == {code for _layer, code in EXPECTED_GAPS}, (
        sorted(set(refusal.missing) ^ {code for _layer, code in EXPECTED_GAPS}))


def test_actual_tir_runs_two_distinct_inputs_on_cpu():
    """The FINALIZED actual PrimFunction executes on the CPU with two real inputs."""
    _, _, module = _run_normal_entry()
    entry = module.entry_function()
    first = np.arange(LANES, dtype="float32")
    second = (np.arange(LANES, dtype="float32") - np.float32(32)) * np.float32(0.25)
    first_out = run_entry(entry, {"x": first})
    second_out = run_entry(module, {"x": second})
    assert set(first_out) == set(second_out) == {"out"}
    assert first_out["out"].shape == (LANES,)
    assert first_out["out"].dtype == np.dtype("float32")
    # Semantics of the authored program: load x once, then each whole-VF iteration
    # computes carried = carried + x, so out = x + 4*x = 5*x elementwise.
    assert np.array_equal(first_out["out"], np.float32(5) * first)
    assert np.array_equal(second_out["out"], np.float32(5) * second)
    assert not np.array_equal(first_out["out"], second_out["out"])
    assert np.array_equal(first, np.arange(LANES, dtype="float32"))


class _UnregisteredInstruction(Op):
    """A real Op subclass with no registered target evaluator."""


def test_executor_refuses_an_unregistered_instruction_by_name():
    lane = make_tensor_type((LANES,), DType.f32, "gmem")
    x = Var("x", type=lane)
    out = Var("out", type=lane)
    function = PrimFunction(
        "noop", (x, out),
        Sequential((Evaluate(callable=_UnregisteredInstruction(), args=()),)),
        output_count=1,
    )
    with pytest.raises(ExecutionRefusal) as caught:
        run_entry(function, {"x": np.zeros(LANES, dtype="float32")})
    refusal = caught.value
    assert refusal.code == "EXEC_INSTRUCTION_UNSUPPORTED"
    assert refusal.subject == "_UnregisteredInstruction"
    assert "GmToUb" in str(refusal)
