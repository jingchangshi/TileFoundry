"""Effect-aware DCE: retained vs deleted loops, stated and pinned."""

from __future__ import annotations

import pathlib
import tempfile

from tilefoundry.analysis.effects import (
    classify_dropped_loop,
    instruction_operand_writes,
    storage_of,
    value_is_external,
)
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.core.param_def import MemoryEffect
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.pipeline import PipelineError, PipelineSpec
from tilefoundry.ir.types import DType, make_tensor_type
from tilefoundry.ir.visitor import collect_exprs

HEADER = (
    "from __future__ import annotations\n"
    "from tilefoundry.dsl import *\n"
    "from tilefoundry.target import CpuTarget\n"
)
SMEM = 'Tensor[(64,), "f32", (64,), "smem"]'


def _load(body_lines):
    rows = [
        HEADER,
        '@module(entry="k", target=CpuTarget())',
        "class M:",
        "    @func",
        '    def k(x: Tensor[(64,), "f32"]) -> Tensor[(64,), "f32"]:',
    ]
    rows.extend("        " + line for line in body_lines)
    directory = pathlib.Path(tempfile.mkdtemp(prefix="tf_dce_"))
    path = directory / "prog.py"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return load_authored_ir(f"{path}:M")


def _loops(module):
    function = module.entry_function()
    return [e for e in collect_exprs(function.body) if isinstance(e, LoopRegion)]


def test_private_smem_only_loop_is_deleted():
    module = _load(
        [
            f"acc = tf.empty({SMEM})",
            "for i in range(4):",
            f"    back = tf.empty({SMEM})",
            "    acc = tf.insert_slice(acc, back, (0,))",
            "return x",
        ]
    )
    assert len(_loops(module)) == 0


def test_smem_result_consumed_into_gm_is_retained():
    module = _load(
        [
            f"acc = tf.empty({SMEM})",
            "for i in range(4):",
            f"    back = tf.empty({SMEM})",
            "    acc = tf.insert_slice(acc, back, (0,))",
            "return tf.insert_slice(x, acc, (0,))",
        ]
    )
    assert len(_loops(module)) == 1


def test_loop_carried_gmem_value_is_retained():
    module = _load(
        [
            "acc = x",
            "for i in range(4):",
            "    acc = acc + 1.0",
            "return acc",
        ]
    )
    assert len(_loops(module)) == 1


def test_unconsumed_computed_value_is_deleted():
    module = _load(
        [
            "acc = x",
            "for i in range(4):",
            "    acc = acc + 1.0",
            "return x",
        ]
    )
    assert len(_loops(module)) == 0


def test_pipeline_loop_deleted_still_refused_by_intent_check():
    module_path_body = [
        f"acc = tf.empty({SMEM})",
        "for i in tf.pipelined(4, max_stage=2, max_buffer_versions=2):",
        f"    back = tf.empty({SMEM})",
        "    acc = tf.insert_slice(acc, back, (0,))",
        "return x",
    ]
    try:
        _load(module_path_body)
    except PipelineError as error:
        assert error.code == "PIPELINE_DECLARATION_DROPPED_BY_DCE", error
    else:
        raise AssertionError("a deleted pipelined loop must be refused, not silently dropped")


def test_pipeline_loop_consumed_is_retained():
    module = _load(
        [
            f"acc = tf.empty({SMEM})",
            "for i in tf.pipelined(4, max_stage=2, max_buffer_versions=2):",
            f"    back = tf.empty({SMEM})",
            "    acc = tf.insert_slice(acc, back, (0,))",
            "return tf.insert_slice(x, acc, (0,))",
        ]
    )
    loops = _loops(module)
    assert len(loops) == 1 and loops[0].pipeline is not None


class _WriteParam:
    kind = "input"
    effect = MemoryEffect.WRITE


class _WriteOp:
    _op_schema = type("S", (), {"signature": (_WriteParam(),)})()


def test_instruction_operand_write_is_reported_but_is_not_a_root():
    gmem = make_tensor_type((64,), dtype=DType.f32, storage="gmem")
    smem = make_tensor_type((64,), dtype=DType.f32, storage="smem")
    call = Call(target=_WriteOp(), args=(Var(name="dst", type=gmem),), type=smem)
    writes = instruction_operand_writes(call)
    assert writes == ((0, None),) or writes[0][0] == 0, writes
    assert storage_of(call) is smem.storage
    assert value_is_external(call) is False


def test_dropped_loop_classification():
    index = make_tensor_type((), dtype=DType.i64)
    smem = make_tensor_type((64,), dtype=DType.f32, storage="smem")
    plain = LoopRegion(
        type=smem,
        induction_var=Var(name="i", type=index),
        params=(),
        args=(),
        body=Constant(0.0, type=smem),
        yield_values=(),
        extent=Constant(4, type=index),
        step=Constant(1, type=index),
    )
    assert classify_dropped_loop(plain) == "private-dead"
    pipelined = LoopRegion(
        type=smem,
        induction_var=Var(name="i", type=index),
        params=(),
        args=(),
        body=Constant(0.0, type=smem),
        yield_values=(),
        extent=Constant(4, type=index),
        step=Constant(1, type=index),
        pipeline=PipelineSpec(max_stage=2, max_buffer_versions=2),
    )
    assert classify_dropped_loop(pipelined) == "pipeline"
