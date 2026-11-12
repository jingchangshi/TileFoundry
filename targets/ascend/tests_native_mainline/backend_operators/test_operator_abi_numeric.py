"""Operator ABI and VALUE tests for the native AR/MLA builders.

The defect these lock down: both builders returned PrimFunction(params=(..., out, ..., ...),
output_count=1), while PrimFunction.output_count counts TRAILING parameters. The callable
contract therefore read the output as an input and the last parameter (a scalar extent) as
the output. Shape-only assertions could not see it, so every check here goes through the
real ABI derivation and, where it matters, through actual numpy values executed by the
independent template evaluator.
"""
from __future__ import annotations

import numpy as np
import pytest

from tilefoundry.ir.core import Constant, Var
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Sequential
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.utils import make_tensor_type
from tilefoundry_ascend.attnres.build import AttnResSpec, build_attnres_module
from tilefoundry_ascend.codegen.abi import abi_from_module, abi_from_prim_function
from tilefoundry_ascend.mla.build import MLASpec, build_mla_module
from tf_ascend_runtime.abi import AbiRefusal, OrderedABI, validate_bindings

from native_reference import run_module


def _ar_module():
    return build_attnres_module(
        AttnResSpec(n=4, tokens=8, d=256, token_rows=2, block_n=1, d_chunk=256))


def _mla_module():
    return build_mla_module(MLASpec(batch=1, seq_q=8, seq_kv=16, heads_q=8, heads_kv=2,
                                    d_nope=64, d_rope=64, kv_block=4, block_q=4))


def _lane(values):
    return np.ascontiguousarray(np.asarray(values, dtype="float32"))


def test_attnres_abi_binds_y_as_trailing_output():
    module = _ar_module()
    abi = abi_from_module(module)
    assert abi.entry == "attention_residual"
    assert [spec.name for spec in abi.tensor_inputs] == ["x_tile", "v_tile"]
    assert [spec.name for spec in abi.scalars] == ["M_extent", "N_extent"]
    assert [spec.name for spec in abi.outputs] == ["y_tile"]
    assert module.entry_function().params[-1].name == "y_tile"
    assert abi.output_split == 4 and len(abi.parameters) == 5
    assert abi.outputs[0].storage == "gmem" and abi.outputs[0].role == "tensor"
    assert [spec.role for spec in abi.parameters] == ["tensor", "tensor", "scalar",
                                                      "scalar", "tensor"]
    descriptor = abi.as_dict()
    assert descriptor["ordered_arguments"][-1]["name"] == "N_extent"
    assert descriptor["output_parameters"][0]["name"] == "y_tile"


def test_attnres_values_reach_the_declared_output_only():
    module = _ar_module()
    x = _lane(np.linspace(-2.0, 2.0, 64))
    v = _lane(np.linspace(3.0, -1.0, 64))
    x_before, v_before = x.copy(), v.copy()
    outputs, log = run_module(module, (x, v), (8, 4))
    assert set(outputs) == {"y_tile"}
    assert np.array_equal(outputs["y_tile"], (x * v).astype("float32"))
    assert np.array_equal(x, x_before) and np.array_equal(v, v_before)
    assert log == []

    other_x = _lane(np.arange(64, dtype="float32"))
    other_v = _lane(np.full(64, 0.5, dtype="float32"))
    other_outputs, _ = run_module(module, (other_x, other_v), (1, 1))
    assert not np.array_equal(outputs["y_tile"], other_outputs["y_tile"])
    # Scalar extents select loop counts; the transport template must still be idempotent.
    same_outputs, _ = run_module(module, (other_x, other_v), (8, 4))
    assert np.array_equal(other_outputs["y_tile"], same_outputs["y_tile"])


def test_attnres_middle_output_order_is_refused_by_the_abi():
    """The exact baseline defect: output before the trailing scalars."""
    lane = make_tensor_type((64,), DType.f32, "gmem")
    x = Var("x_tile", type=lane)
    v = Var("v_tile", type=lane)
    y = Var("y_tile", type=lane)
    m_extent = Var("M_extent", type=TensorType.scalar(DType.i64))
    n_extent = Var("N_extent", type=TensorType.scalar(DType.i64))
    broken = PrimFunction("attention_residual", (x, v, y, m_extent, n_extent),
                          Sequential(()), output_count=1)
    with pytest.raises(AbiRefusal, match="ABI_OUTPUT_NOT_TENSOR"):
        abi_from_prim_function(broken)


def test_mla_abi_binds_out_as_trailing_output():
    module = _mla_module()
    abi = abi_from_module(module)
    assert [spec.name for spec in abi.tensor_inputs] == [
        "q_nope_tile", "q_rope_tile", "k_nope_tile", "k_rope_tile", "v_tile"]
    assert [spec.name for spec in abi.scalars] == ["job_count", "kv_blocks"]
    assert [spec.name for spec in abi.outputs] == ["out_tile"]
    assert module.entry_function().params[-1].name == "out_tile"
    assert len(abi.parameters) == 8 and abi.output_split == 7


def test_mla_values_reach_the_declared_output_only():
    module = _mla_module()
    q_nope = _lane(np.arange(64, dtype="float32"))
    q_rope = _lane(np.full(64, 7.0, dtype="float32"))
    k_nope = _lane(np.linspace(1.0, 2.0, 64).astype("float32"))
    k_rope = _lane(np.full(64, -3.0, dtype="float32"))
    v = _lane(np.full(64, 5.0, dtype="float32"))
    copies = [item.copy() for item in (q_nope, q_rope, k_nope, k_rope, v)]
    outputs, _log = run_module(module, (q_nope, q_rope, k_nope, k_rope, v), (2, 4))
    assert set(outputs) == {"out_tile"}
    assert np.array_equal(outputs["out_tile"], (q_nope * k_nope).astype("float32"))
    for original, current in zip(copies, (q_nope, q_rope, k_nope, k_rope, v)):
        assert np.array_equal(original, current)


def test_abi_descriptor_round_trips_and_refuses_tampering():
    abi = abi_from_module(_ar_module())
    rebuilt = OrderedABI.from_dict(abi.as_dict())
    assert rebuilt.digest() == abi.digest()
    assert [spec.name for spec in rebuilt.outputs] == ["y_tile"]

    tampered = abi.as_dict()
    tampered["digest"] = "0" * 64
    with pytest.raises(AbiRefusal, match="ABI_DESCRIPTOR_DIGEST"):
        OrderedABI.from_dict(tampered)

    foreign = abi.as_dict()
    foreign["schema"] = "ascend.native-abi/0"
    with pytest.raises(AbiRefusal, match="ABI_DESCRIPTOR_SCHEMA"):
        OrderedABI.from_dict(foreign)


def test_bindings_refuse_wrong_shape_wrong_dtype_and_non_contiguous():
    abi = abi_from_module(_ar_module())
    x = _lane(np.zeros(64, dtype="float32"))
    v = _lane(np.zeros(64, dtype="float32"))
    assert validate_bindings(abi, (x, v), (8, 4)) == (x, v, 8, 4)
    with pytest.raises(AbiRefusal, match="ABI_INPUT_SHAPE"):
        validate_bindings(abi, (_lane(np.zeros(32)), v), (8, 4))
    with pytest.raises(AbiRefusal, match="ABI_INPUT_DTYPE"):
        validate_bindings(abi, (np.zeros(64, dtype="int32"), v), (8, 4))
    with pytest.raises(AbiRefusal, match="ABI_INPUT_COUNT"):
        validate_bindings(abi, (x,), (8, 4))
    with pytest.raises(AbiRefusal, match="ABI_SCALAR_TYPE"):
        validate_bindings(abi, (x, v), (8, 4.0))

    class NonContiguous:
        shape = (64,)
        dtype = np.dtype("float32")

        @staticmethod
        def is_contiguous():
            return False

    with pytest.raises(AbiRefusal, match="ABI_INPUT_NONCONTIGUOUS"):
        validate_bindings(abi, (NonContiguous(), v), (8, 4))
