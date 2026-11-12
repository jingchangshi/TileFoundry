"""Effect-aware DCE: retained vs deleted loops, stated and pinned.

D0 HIR is value-form SSA, so "externally visible write" and "reachable from the
returned value" are the same fact. These controls pin the three rules:

  * a loop whose produced value no one consumes (private SMEM dead) is DELETED;
  * the same loop consumed by an insert_slice back to GM / returned is RETAINED;
  * an instruction's WRITE operand is NOT an HIR mutation of the old Var, so it is
    not a DCE root on its own;
  * F0's intent check still refuses a deleted loop that declared a pipeline.
"""

from __future__ import annotations

import pathlib
import tempfile

from tilefoundry.analysis.effects import (
    classify_dropped_loop,
    instruction_operand_writes,
    reachable_loop_ids,
    reachable_loops,
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
    rows = [HEADER, '@module(entry="k", target=CpuTarget())', "class M:", "    @func",
            '    def k(x: Tensor[(64,), "f32"]) -> Tensor[(64,), "f32"]:']
    rows.extend("        " + line for line in body_lines)
    directory = pathlib.Path(tempfile.mkdtemp(prefix="tf_dce_"))
    path = directory / "prog.py"
    path.write_text("\n".join(rows) + "\n")
    return load_authored_ir("%s:M" % path)


def _loops(module):
    function = module.entry_function()
    return [e for e in collect_exprs(function.body) if isinstance(e, LoopRegion)]


def test_private_smem_only_loop_is_deleted():
    """No consumer of the produced SMEM value -> the loop is dead and drops out."""
    module = _load(
        [
            "acc = tf.empty(%s)" % SMEM,
            "for i in range(4):",
            "    back = tf.empty(%s)" % SMEM,
            "    acc = tf.insert_slice(acc, back, (0,))",
            "return x",
        ]
    )
    assert len(_loops(module)) == 0


def test_smem_result_consumed_into_gm_is_retained():
    """The same loop, whose SMEM result feeds a returned GM value, is retained."""
    module = _load(
        [
            "acc = tf.empty(%s)" % SMEM,
            "for i in range(4):",
            "    back = tf.empty(%s)" % SMEM,
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
        "acc = tf.empty(%s)" % SMEM,
        "for i in tf.pipelined(4, num_stages=2):",
        "    back = tf.empty(%s)" % SMEM,
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
            "acc = tf.empty(%s)" % SMEM,
            "for i in tf.pipelined(4, num_stages=2):",
            "    back = tf.empty(%s)" % SMEM,
            "    acc = tf.insert_slice(acc, back, (0,))",
            "return tf.insert_slice(x, acc, (0,))",
        ]
    )
    loops = _loops(module)
    assert len(loops) == 1 and loops[0].pipeline is not None


# --- rule 3: instruction operand WRITE is not a DCE root ----------------------


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
    # Report it, but do NOT let it make the produced value external: the result is SMEM.
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
        pipeline=PipelineSpec(num_stages=2),
    )
    assert classify_dropped_loop(pipelined) == "pipeline"
