"""Tests for EnvelopeInspector positional dynamic binding and zero-divisor safety."""

from tilefoundry_ascend.ascend.schedule.envelope_inspector import (
    inspect_tilelang_primfunc,
    specialize_primfunc_for_anchor,
)


def test_envelope_inspector_binds_positional_shapes_and_avoids_zero_divisors() -> None:
    """Bind unnamed dynamic dims from ordered_tensor_shapes and clamp zero/unbound divisors."""
    import tilelang.ascend.language as T  # noqa: PLC0415
    from tilelang import tvm  # noqa: PLC0415

    m_dyn = tvm.tirx.SizeVar("unnamed_m", "int32")
    n_dyn = tvm.tirx.SizeVar("unnamed_n", "int32")

    @T.prim_func
    def dynamic_kernel(
        X: T.Tensor((m_dyn, n_dyn), "bfloat16"),
        Y: T.Tensor((m_dyn, n_dyn), "bfloat16"),
        tile_m: T.int32,
    ):
        with T.Kernel(T.ceildiv(m_dyn, tile_m)):
            x_ub = T.alloc_shared((64, 64), "bfloat16")
            T.copy(X[0:64, 0:64], x_ub)
            T.copy(x_ub, Y[0:64, 0:64])

    spec_fn, _ = specialize_primfunc_for_anchor(
        dynamic_kernel,
        var_bindings={"tile_m": 0},
        ordered_tensor_shapes=((256, 64), (256, 64)),
    )
    assert spec_fn is not None

    inspection = inspect_tilelang_primfunc(
        dynamic_kernel,
        var_bindings={"tile_m": 0},
        ordered_tensor_shapes=((256, 64), (256, 64)),
    )
    assert inspection.ledger is not None
    assert "Divide by zero" not in (inspection.lowering_error or "")
    assert inspection.ledger.grid_blocks >= 1


def test_envelope_inspector_binds_renamed_decode_split_and_clamps_hc_divisor() -> None:
    """Bind renamed decode_split dims positionally and clamp m_rows_dyn // hc_dim_dyn divisors."""
    import tilelang.ascend.language as T  # noqa: PLC0415
    from tilelang import tvm  # noqa: PLC0415

    decode_split = tvm.tirx.SizeVar("decode_split", "int32")
    m_rows_dyn = tvm.tirx.SizeVar("m_rows_dyn", "int32")
    d_dyn = tvm.tirx.SizeVar("d_dyn", "int32")

    @T.prim_func
    def split_kernel(
        Q: T.Tensor((decode_split, d_dyn), "bfloat16"),
        Out: T.Tensor((m_rows_dyn, d_dyn), "bfloat16"),
        hc_dim_dyn: T.int32,
    ):
        with T.Kernel(m_rows_dyn // hc_dim_dyn):
            q_ub = T.alloc_shared((64, 64), "bfloat16")
            for _r in T.serial(decode_split // hc_dim_dyn):
                T.copy(Q[0:64, 0:64], q_ub)
                T.copy(q_ub, Out[0:64, 0:64])

    spec_unbound, _ = specialize_primfunc_for_anchor(
        split_kernel,
        var_bindings=None,
        ordered_tensor_shapes=((128, 64), (128, 64)),
    )
    assert spec_unbound is not None

    insp_zero = inspect_tilelang_primfunc(
        split_kernel,
        var_bindings={"hc_dim_dyn": 0},
        ordered_tensor_shapes=((128, 64), (128, 64)),
    )
    assert insp_zero.ledger is not None
    assert "Divide by zero" not in (insp_zero.lowering_error or "")
    assert insp_zero.ledger.grid_blocks >= 1

    insp_unbound = inspect_tilelang_primfunc(
        split_kernel,
        var_bindings=None,
        ordered_tensor_shapes=((128, 64), (128, 64)),
    )
    assert insp_unbound.ledger is not None
    assert "Divide by zero" not in (insp_unbound.lowering_error or "")
    assert insp_unbound.ledger.grid_blocks >= 1
