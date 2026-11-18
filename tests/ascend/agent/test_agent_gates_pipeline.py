"""Integration tests for GateFunnel Stage A/B custom kernels, --cce-res-usage, and HAP."""

from __future__ import annotations

from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    GateFunnel,
    StageBResult,
    StageCResult,
    parse_cce_res_usage,
    run_stage_a_static,
    run_stage_b_compile,
)
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    compute_eq3_hap_objective,
    compute_static_dynamic_residual,
    extract_structural_signature,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger

_PYTHON_KERNEL_MODULE = """
import torch

def kernel(x: torch.Tensor, gamma: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    xf = x.to(torch.float32)
    rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    return (xf * rms * gamma.to(torch.float32)).to(x.dtype)
"""

_ILLEGAL_ASCENDC_WRAPPER = """
ASCENDC_CODE = '''
// Illegal dav-3510 datapath: L1->GM and GM->L0A
'''
def kernel(x):
    return x
"""


def test_gate_funnel_python_kernel_module_and_cce_res_usage_parsing() -> None:
    """Verify Stage A/B support Python kernel(...) modules and parse --cce-res-usage."""
    sa_ok = run_stage_a_static(_PYTHON_KERNEL_MODULE)
    assert sa_ok.passed is True
    assert sa_ok.backend_kind == "python_kernel"
    assert sa_ok.ledger is not None and sa_ok.ledger.is_legal is True

    sb_ok = run_stage_b_compile(_PYTHON_KERNEL_MODULE)
    assert sb_ok.passed is True
    assert sb_ok.spill_bytes == 0
    assert sb_ok.vloopv2_hit is True

    sa_bad = run_stage_a_static(_ILLEGAL_ASCENDC_WRAPPER)
    assert sa_bad.passed is False
    assert "forbidden_dav3510_dma_path" in sa_bad.error_message

    spill, vloop = parse_cce_res_usage(
        "warning: Register spill: 128 bytes\nremark: VLOOPv2 miss: inner branch\n"
    )
    assert spill == 128
    assert vloop is False

    spill_clean, vloop_clean = parse_cce_res_usage("info: stack spill: 0 bytes\nVLOOPv2 hit\n")
    assert spill_clean == 0
    assert vloop_clean is True

    funnel = GateFunnel(
        skip_stage_b=False,
        device_runner=lambda _src, st_a: StageCResult(
            passed=True,
            accuracy_passed=True,
            measured_latency_us=5.0,
            residual=compute_static_dynamic_residual(
                st_a.ledger,
                {
                    "dur_us": 5.0,
                    "spill_bytes": 256,
                    "bank_conflict_ratio": 0.25,
                    "l2_hit_rate": 0.60,
                },
            ),
            telemetry={"dur_us": 5.0, "spill_bytes": 256},
        ),
    )
    ev = funnel.evaluate(_PYTHON_KERNEL_MODULE)
    assert ev.passed is True
    assert ev.stage_c is not None and ev.stage_c.residual is not None
    assert ev.stage_c.residual.spill_bytes == 256
    assert ev.stage_c.residual.bank_conflict_ratio == 0.25
    assert ev.stage_c.residual.l2_hit_rate == 0.60

    clean_score = compute_eq3_hap_objective(
        5.0, ref_perf_us=15.0, t_hw_us=2.5, spill_bytes=0, bank_conflict_ratio=0.0, l2_hit_rate=1.0
    )
    penalized_score = compute_eq3_hap_objective(
        5.0, ref_perf_us=15.0, t_hw_us=2.5, residual=ev.stage_c.residual
    )
    assert 0.0 < penalized_score < clean_score

    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 50},
        waterlines={},
    )
    res = compute_static_dynamic_residual(ledger, {"dur_us": 5.0, "spill_bytes": 64, "l2_hit_rate": 0.5})
    sig = extract_structural_signature(
        _PYTHON_KERNEL_MODULE,
        residual=res,
        stage_b=StageBResult(passed=True, spill_bytes=64, vloopv2_hit=False),
    )
    assert "hw:register_spill" in sig
    assert "hw:low_l2_hit" in sig
    assert "hw:vloopv2_miss" in sig
