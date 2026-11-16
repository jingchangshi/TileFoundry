"""Live 1-operator NPU funnel smoke test via AgentGates (Stage A -> Stage B -> Stage C)."""

import json
import os
from pathlib import Path

import pytest
import torch

from tilefoundry_ascend.agent.orchestrator import (
    compute_static_dynamic_residual,
    run_stage_a_static,
    run_stage_b_compile,
)

_SMOKE_DSL = """
import tilelang.ascend.language as T

def make_vec_add():
    @T.prim_func
    def main(
        A: T.Tensor((4096,), "float32"),
        B: T.Tensor((4096,), "float32"),
        C: T.Tensor((4096,), "float32"),
    ):
        with T.Kernel(8) as bx:
            a_ub = T.alloc_shared((512,), "float32")
            b_ub = T.alloc_shared((512,), "float32")
            c_ub = T.alloc_shared((512,), "float32")
            off = bx * 512
            T.copy(A[off : off + 512], a_ub)
            T.copy(B[off : off + 512], b_ub)
            with T.SimdVF():
                mask = T.simd.pset(32)
                for i in range(8):
                    va = T.simd.vld(a_ub[i * 64])
                    vb = T.simd.vld(b_ub[i * 64])
                    T.simd.vsts(c_ub[i * 64], T.simd.vadd(va, vb, mask), mask)
            T.copy(c_ub, C[off : off + 512])
    return main
"""


def _require_npu() -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU device is required for Stage C smoke test.")


def test_live_one_op_funnel_stage_a_b_c_on_npu() -> None:
    """Execute Stage A static -> Stage B bisheng -> Stage C NPU msprof on Ascend 950PR."""
    _require_npu()
    import math
    import tilelang
    import tilelang.ascend.language as T
    from tilelang.profiler import do_bench

    stage_a = run_stage_a_static(_SMOKE_DSL, factory_name="make_vec_add")
    assert stage_a.passed is True
    assert stage_a.ledger is not None
    assert stage_a.route is not None and stage_a.route.track == "developer"

    stage_b = run_stage_b_compile(
        _SMOKE_DSL,
        factory_name="make_vec_add",
        enable_device_compile=True,
    )
    assert stage_b.passed is True, stage_b.error_message

    prim_func = stage_a.prim_func
    kernel = tilelang.compile(prim_func, out_idx=-1)

    a = torch.randn((4096,), dtype=torch.float32, device="npu")
    b = torch.randn((4096,), dtype=torch.float32, device="npu")
    out = kernel(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), (a + b).cpu(), atol=1e-4, rtol=1e-4)

    prof = do_bench(lambda: kernel(a, b), backend="msprof_detail", _n_warmup=3, _n_repeat=5)
    telemetry = {
        "dur_us": float(prof.dur_us),
        "aic_mac_ratio": float(prof.aic_mad),
        "aiv_vec_ratio": float(prof.aiv_vec),
        "mte2_wait_ratio": float(prof.aiv_mte2),
        "sync_stall_ratio": 0.0,
    }
    residual = compute_static_dynamic_residual(stage_a.ledger, telemetry)
    assert residual.measured_latency_us > 0.0
    assert 0.0 <= residual.unexplained_gap_ratio <= 1.0

    report_path = os.environ.get("TILEFOUNDRY_FUNNEL_SMOKE_REPORT", "").strip()
    if report_path:
        Path(report_path).write_text(
            json.dumps({"telemetry": telemetry, "residual": residual.to_dict()}, indent=2),
            encoding="utf-8",
        )
