"""task-76 A/C: value-form retention assumption + wiring labels.

A) The retention assumption (value-form SSA) is explicit and guarded: a DROPPED loop that
   writes a GMEM operand produces DCE_VALUE_FORM_ASSUMPTION_VIOLATED, while a dropped loop
   writing only private storage is allowed. C) Each predicate docstring states whether it is
   production-wired or test-only.
"""
from __future__ import annotations

import types

import pytest

from tilefoundry.analysis import effects
from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.core.param_def import MemoryEffect
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.pipeline import PipelineError
from tilefoundry.ir.types import DType, make_tensor_type


class _WriteParam:
    kind = "input"
    effect = MemoryEffect.WRITE


class _WriteOp:
    _op_schema = type("S", (), {"signature": (_WriteParam(),)})()


def _loop_with_operand_write(storage):
    target_type = make_tensor_type((64,), dtype=DType.f32, storage=storage)
    call = Call(target=_WriteOp(), args=(Var(name="dst", type=target_type),), type=target_type)
    index = make_tensor_type((), dtype=DType.i64)
    return LoopRegion(
        type=target_type,
        induction_var=Var(name="i", type=index),
        params=(),
        args=(),
        body=call,
        yield_values=(),
        extent=Constant(4, type=index),
        step=Constant(1, type=index),
    )


def _empty_function():
    return types.SimpleNamespace(body=None)


def test_dropped_gmem_operand_write_is_refused_by_name():
    loop = _loop_with_operand_write("gmem")
    with pytest.raises(PipelineError) as caught:
        effects.assert_value_form_retention(_empty_function(), [loop])
    assert caught.value.code == "DCE_VALUE_FORM_ASSUMPTION_VIOLATED", caught.value


def test_dropped_private_operand_write_is_allowed():
    loop = _loop_with_operand_write("smem")
    report = effects.assert_value_form_retention(_empty_function(), [loop])
    assert report["dropped"] == 1


def test_no_candidates_is_a_noop():
    report = effects.assert_value_form_retention(_empty_function(), [])
    assert report["candidates"] == 0


def test_reachable_loops_is_marked_production():
    assert "PRODUCTION" in (effects.reachable_loops.__doc__ or "")


def test_test_only_predicates_are_marked():
    for fn in (
        effects.value_is_external,
        effects.instruction_operand_writes,
        effects.loop_is_retained,
        effects.classify_dropped_loop,
    ):
        assert "TEST-ONLY" in (fn.__doc__ or ""), fn.__name__
