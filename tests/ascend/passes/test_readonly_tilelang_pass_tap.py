"""Tests for ReadOnlyAscendPassTap, device_mod reconciliation, and CANN 9.3 lock."""

from __future__ import annotations

import pytest
import tilelang.ascend.language as T
from tilelang import tvm

from tilefoundry_ascend.ascend.schedule import (
    ReadOnlyAscendPassTap,
    inspect_tilelang_primfunc,
)
from tilefoundry_ascend.ascend.target import DEFAULT_CANN_SERIES, resolve_cann_series


def test_readonly_ascend_pass_tap_does_not_mutate_primfunc_and_captures_taps() -> None:
    """Verify ReadOnlyAscendPassTap preserves input PrimFunc hash/IR and records 4 taps."""
    @T.prim_func
    def gemm_kernel(
        X: T.Tensor((128, 128), "bfloat16"),
        W: T.Tensor((128, 128), "bfloat16"),
        Y: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(1):
            x_l1 = T.alloc_l1((128, 64), "bfloat16")
            w_l1 = T.alloc_l1((128, 64), "bfloat16")
            acc = T.alloc_l0c((128, 128), "float32")
            for kt in T.Pipelined(2, num_stages=2):
                T.copy(X[0:128, kt * 64 : (kt + 1) * 64], x_l1)
                T.copy(W[0:128, kt * 64 : (kt + 1) * 64], w_l1)
                T.gemm(x_l1, w_l1, acc, transpose_B=True, clear_accum=kt == 0)
            T.copy(acc, Y[0:128, 0:128])

    hash_before = tvm.ir.structural_hash(gemm_kernel)
    script_before = gemm_kernel.script()

    res = inspect_tilelang_primfunc(gemm_kernel)

    assert tvm.ir.structural_hash(gemm_kernel) == hash_before
    assert gemm_kernel.script() == script_before
    assert "tl.MaterializeMultiBuffer" in res.lowered_passes_observed
    assert "tl.InsertNd2Nz" in res.lowered_passes_observed
    assert "tl.NormalizeAscendFractalStorage" in res.lowered_passes_observed
    assert "tl.MergeUBAllocations" in res.lowered_passes_observed
    assert (
        "tl.AscendLowerTileOp" in res.lowered_passes_observed
        or "tl.LowerTileOp" in res.lowered_passes_observed
    )
    assert res.ledger.uses_dynamic_offset_pipelining is True
    assert res.ledger.waterlines["L1"].peak_allocated_bytes >= 2 * 128 * 64 * 2
    assert isinstance(ReadOnlyAscendPassTap(), object)


def test_reconcile_post_lower_device_mod_extracts_barriers_narrow_dma_and_scalar_loops() -> None:
    """Verify device_mod reconciliation captures PIPE_ALL, <32B DMA, and UB scalar loops."""
    @T.prim_func
    def narrow_barrier_scalar(
        X: T.Tensor((16, 64), "float32"),
        Y: T.Tensor((16, 64), "float32"),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((16, 64), "float32")
            for i in T.serial(16):
                T.copy(X[i, 0:4], ub[i, 0:4])
                T.ascend_pipe_barrier("PIPE_ALL")
                for j in T.serial(64):
                    ub[i, j] = ub[i, j] + T.float32(1.0)
                T.copy(ub[i, 0:4], Y[i, 0:4])

    res = inspect_tilelang_primfunc(
        narrow_barrier_scalar,
        pass_configs={"tl.enable_auto_schedule": False},
    )
    assert res.ledger.in_loop_pipe_all_barriers >= 1
    assert res.ledger.min_gm_dma_bytes == 16
    assert 0.0 < res.ledger.dma_burst_efficiency_ratio < 1.0
    assert res.ledger.ub_scalar_compute_loops >= 1


def test_reconcile_post_lower_device_mod_tracks_asymmetric_sid_aiv_pipe_v() -> None:
    """Verify device_mod reconciliation splits AIV0.PIPE_V vs AIV1.PIPE_V across sid branches."""
    @T.prim_func
    def asymmetric_sid_kernel():
        with T.MixedKernel(1, sids=2) as (bx, sid):
            x_ub = T.alloc_shared((64, 128), "float32")
            if sid == 0:
                with T.SimdVF():
                    m32 = T.simd.pset(32, "PAT_ALL")
                    for r in T.serial(6):
                        v = T.simd.vld(x_ub[r, 0], dist="NORM_B32")
                        T.simd.vsts(x_ub[r, 0], v, mask=m32, dist="NORM_B32")
            else:
                with T.SimdVF():
                    m32 = T.simd.pset(32, "PAT_ALL")
                    for r in T.serial(2):
                        v = T.simd.vld(x_ub[r, 0], dist="NORM_B32")
                        T.simd.vsts(x_ub[r, 0], v, mask=m32, dist="NORM_B32")

    res = inspect_tilelang_primfunc(asymmetric_sid_kernel)
    assert res.ledger.pipe_cycles["AIV0.PIPE_V"] == 6
    assert res.ledger.pipe_cycles["AIV1.PIPE_V"] == 2
    assert res.ledger.ub_scalar_compute_loops == 0


def test_resolve_cann_series_resolves_cann93(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify default, env, and path resolution consistently resolve to cann_9_3."""
    monkeypatch.delenv("TILEFOUNDRY_CANN_SERIES", raising=False)
    monkeypatch.delenv("TILEFOUNDRY_NPU_CANN_SERIES", raising=False)
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)

    assert resolve_cann_series() == DEFAULT_CANN_SERIES == "cann_9_3"

    monkeypatch.setenv("TILEFOUNDRY_CANN_SERIES", "cann_9_3")
    assert resolve_cann_series() == DEFAULT_CANN_SERIES == "cann_9_3"
    monkeypatch.delenv("TILEFOUNDRY_CANN_SERIES", raising=False)

    monkeypatch.setenv("TILEFOUNDRY_NPU_CANN_SERIES", "cann_9_3")
    assert resolve_cann_series() == "cann_9_3"
    monkeypatch.delenv("TILEFOUNDRY_NPU_CANN_SERIES", raising=False)

    monkeypatch.setenv("ASCEND_HOME_PATH", "/opt/Ascend/cann-9.3.0")
    assert resolve_cann_series() == "cann_9_3"
    assert resolve_cann_series(cann_home="/opt/Ascend/cann-9.3.0") == "cann_9_3"
    assert resolve_cann_series("cann_9_3") == "cann_9_3"
