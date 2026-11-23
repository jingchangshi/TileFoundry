"""Tests for numerical twin CPU parity and PyTorch blueprint navigation."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math

import pytest
import torch

from tilefoundry_ascend.agent import (
    BayesianRuleTracker,
    GateFunnel,
    MockLLMBackend,
    PortfolioRunner,
    TwoLaneBeamSearch,
    format_blueprint_delta_section,
    verify_numerical_twin,
    verify_reference_cpu_parity,
)
from tilefoundry_ascend.agent.orchestrator import (
    BeamCandidate,
    StageBResult,
    StageCResult,
    compute_static_dynamic_residual,
    extract_structural_signature,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    build_objective_physical_ledger,
    build_scope_waterline,
)
from tilefoundry_ascend.ascend.sil.pytorch_frontend import (
    PyTorchSemanticAnalyzer,
    RepresentativeCase,
)

_CANDIDATE_DSL = """
import tilelang.ascend.language as T

def make_kernel():
    @T.prim_func
    def main(
        X: T.Tensor((128, 128), "bfloat16"),
        W: T.Tensor((128, 128), "bfloat16"),
        Y: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(2):
            x_l1 = T.alloc_l1((128, 64), "bfloat16")
            w_l1 = T.alloc_l1((128, 64), "bfloat16")
            acc = T.alloc_l0c((128, 128), "float32")
            for kt in T.Pipelined(2, num_stages=2):
                T.copy(X[0:128, kt * 64 : (kt + 1) * 64], x_l1)
                T.copy(W[0:128, kt * 64 : (kt + 1) * 64], w_l1)
                T.gemm(x_l1, w_l1, acc, transpose_B=True, clear_accum=kt == 0)
            T.copy(acc, Y[0:128, 0:128])
    return main
"""


_AR_GOLDEN_SOURCE = """
import math
import torch

def forward(v_stack: torch.Tensor, x: torch.Tensor, w: torch.Tensor, gamma: torch.Tensor):
    vf = v_stack.to(torch.float32)
    xf = x.to(torch.float32)
    wf = w.to(torch.float32)
    gf = gamma.to(torch.float32)
    rms = torch.rsqrt(vf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    k = vf * rms * gf
    d_dim = vf.shape[-1]
    scores = (k * wf.view(1, 1, 1, d_dim)).sum(dim=-1) / math.sqrt(d_dim)
    probs = torch.softmax(scores, dim=0)
    agg = (probs.unsqueeze(-1) * vf).sum(dim=0)
    return (0.5 * xf + 0.5 * agg).to(v_stack.dtype)
"""


def _attention_residual_torch_ref(
    v_stack: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    gamma: torch.Tensor,
    *,
    alpha: float = 0.5,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    vf = v_stack.to(torch.float32)
    xf = x.to(torch.float32)
    wf = w.to(torch.float32)
    gf = gamma.to(torch.float32)
    rms = torch.rsqrt(vf.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    k = vf * rms * gf
    d_dim = vf.shape[-1]
    scores = (k * wf.view(1, 1, 1, d_dim)).sum(dim=-1) / math.sqrt(d_dim)
    probs = torch.softmax(scores, dim=0)
    agg = (probs.unsqueeze(-1) * vf).sum(dim=0)
    y = (1.0 - alpha) * xf + alpha * agg
    return y.to(v_stack.dtype)


def test_verify_numerical_twin_and_cpu_parity_attention_residual() -> None:
    """Verify pure Python/PyTorch twin parity passes matching candidates and rejects mismatches."""
    good_candidate = (
        "import math\n"
        "import torch\n"
        "def forward(v_stack, x, w, gamma):\n"
        "    vf, xf, wf, gf = v_stack.float(), x.float(), w.float(), gamma.float()\n"
        "    rms = torch.rsqrt(vf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)\n"
        "    k = vf * rms * gf\n"
        "    scores = (k * wf.view(1, 1, 1, vf.shape[-1])).sum(dim=-1) / math.sqrt(vf.shape[-1])\n"
        "    probs = torch.softmax(scores, dim=0)\n"
        "    agg = (probs.unsqueeze(-1) * vf).sum(dim=0)\n"
        "    return (0.5 * xf + 0.5 * agg).to(v_stack.dtype)\n"
    )
    bad_candidate = (
        "def forward(v_stack, x, w, gamma):\n"
        "    return x + 0.25\n"
    )
    specs = [
        ((4, 1, 4, 64), "bfloat16"),
        ((1, 4, 64), "bfloat16"),
        ((64,), "bfloat16"),
        ((64,), "bfloat16"),
    ]
    res_ok = verify_numerical_twin(
        _AR_GOLDEN_SOURCE,
        good_candidate,
        input_specs=specs,
        dual_profile=True,
    )
    assert res_ok.passed is True, res_ok.error_message
    assert "reference_cpu_parity_passed" in res_ok.detail

    res_bad = verify_numerical_twin(
        _AR_GOLDEN_SOURCE,
        bad_candidate,
        input_specs=specs,
        dual_profile=True,
    )
    assert res_bad.passed is False
    assert "twin_numeric_mismatch" in res_bad.error_message

    sample_inputs = [
        torch.randn((4, 1, 4, 64), dtype=torch.bfloat16),
        torch.randn((1, 4, 64), dtype=torch.bfloat16),
        torch.randn((64,), dtype=torch.bfloat16),
        torch.randn((64,), dtype=torch.bfloat16),
    ]
    ok, ok_msg = verify_reference_cpu_parity(
        _attention_residual_torch_ref,
        golden_source=_AR_GOLDEN_SOURCE,
        sample_inputs=sample_inputs,
    )
    assert ok is True, ok_msg
    assert "reference_cpu_parity_passed" in ok_msg

    bad_ok, bad_msg = verify_reference_cpu_parity(
        lambda v_stack, x, w, gamma: _attention_residual_torch_ref(v_stack, x, w, gamma) + 0.25,
        golden_source=_AR_GOLDEN_SOURCE,
        sample_inputs=sample_inputs,
    )
    assert bad_ok is False
    assert "twin_numeric_mismatch" in bad_msg


def test_verify_reference_cpu_parity_with_test_dims() -> None:
    """Verify shape-specialized test_dims inputs flow cleanly into verify_reference_cpu_parity."""
    golden_add = (
        "import torch\n"
        "def forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:\n"
        "    return (a + b).to(torch.float32)\n"
    )
    ok, msg = verify_reference_cpu_parity(
        lambda a, b: a + b,
        golden_source=golden_add,
        test_dims={"M": 8, "N": 16, "EXTRA": 99},
    )
    assert ok is True, msg

    bad_ok, bad_msg = verify_reference_cpu_parity(
        lambda a, b: a - b,
        golden_source=golden_add,
        test_dims={"M": 8, "N": 16},
    )
    assert bad_ok is False
    assert "twin_numeric_mismatch" in bad_msg


def test_format_blueprint_delta_section_and_portfolio_roofline_eta() -> None:
    """Verify navigation formatting, ledger delta, and roofline_eta residual threading."""
    mla_golden = (
        "import torch\n"
        "def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:\n"
        "    return torch.matmul(q, k.transpose(-1, -2)) @ v\n"
    )
    analyzer = PyTorchSemanticAnalyzer()
    semantics = analyzer.analyze_source(mla_golden, entry_name="forward")
    raw_pkt = analyzer.build_universal_blueprint(
        semantics,
        anchor_dims={"B": 1, "M": 16, "N": 64, "K": 64},
        representative_case=RepresentativeCase(
            case_id=1,
            input_shapes=((1, 16, 64), (1, 16, 64), (1, 16, 64)),
            input_dtypes=("bfloat16", "bfloat16", "bfloat16"),
            output_dtype="bfloat16",
            output_shapes=((1, 16, 64),),
        ),
    )
    tl_ledger = build_objective_physical_ledger(
        analysis_mode="exact_modulo_z3",
        is_legal=True,
        steady_state_ii=24,
        pipe_cycles={"AIC.PIPE_M": 20, "AIV0.PIPE_V": 48},
        waterlines={
            "UB": build_scope_waterline(
                scope="UB",
                capacity_bytes=192 * 1024,
                explicit_buffers={"tmp": 16384},
                implicit_buffers={},
            )
        },
        gm_roundtrip_bytes=8192,
    )
    packet = raw_pkt
    nav_text = format_blueprint_delta_section(packet, current_ledger=tl_ledger)
    assert "### LAYER 3: PHYSICAL ROOFLINE & BLUEPRINT DELTA" in nav_text
    assert "T.ascend_mmad" not in nav_text
    assert "steady_state_ii:" in nav_text
    assert "ub_peak_bytes:" in nav_text
    assert "gm_roundtrip_bytes:" in nav_text

    parent_res = compute_static_dynamic_residual(tl_ledger, {"dur_us": 20.0})
    parent = BeamCandidate(
        candidate_id="p0",
        round_idx=0,
        worker_kind="seed",
        rule_id=None,
        dsl_source=_CANDIDATE_DSL,
        track="unconstrained",
        ledger=tl_ledger,
        residual=parent_res,
        latency_us=20.0,
        accuracy_passed=True,
        structural_signature=extract_structural_signature(_CANDIDATE_DSL),
    )

    def _no_pmu_device(_dsl, stage_a):
        tel = {"dur_us": 10.0}
        return StageCResult(
            passed=True,
            accuracy_passed=True,
            measured_latency_us=10.0,
            residual=compute_static_dynamic_residual(stage_a.ledger, tel),
            telemetry=tel,
        )

    creative_reply = json.dumps(
        {
            "physical_hypothesis": "Fuse vector epilogue to shrink UB and GM roundtrips.",
            "expected_bottleneck_shift": "Shift from MTE2 wait to compute.",
            "dsl_source": _CANDIDATE_DSL.strip(),
        }
    )
    backend = MockLLMBackend(
        responses={"exploit_rule": [_CANDIDATE_DSL], "explore_creative": [creative_reply]},
        default_response=_CANDIDATE_DSL,
    )
    funnel = GateFunnel(
        compile_runner=lambda _dsl: StageBResult(passed=True),
        device_runner=_no_pmu_device,
    )
    runner = PortfolioRunner(
        backend=backend,
        funnel=funnel,
        tracker=BayesianRuleTracker.from_seed_yaml(),
        beam=TwoLaneBeamSearch(),
        k_rule_workers=1,
    )
    report = asyncio.run(
        runner.run_round(
            parent,
            round_idx=1,
            factory_name="make_kernel",
            ideal_roofline_us=5.0,
            blueprint=packet,
        )
    )
    assert report.evaluated_candidates
    for req in backend.calls:
        assert "### LAYER 3: PHYSICAL ROOFLINE & BLUEPRINT DELTA" in req.user_prompt
        assert "- steady_state_ii:" in req.user_prompt
    for cand in report.evaluated_candidates:
        assert cand.residual.predicted_bottleneck_util >= 0.0
        assert cand.residual.unexplained_gap_ratio > 0.0


def test_format_blueprint_delta_section_gates_cross_scale_deltas() -> None:
    """Verify shape mismatch suppresses raw byte/cycle deltas in favor of normalized usage."""
    ar_golden = (
        "import torch\n"
        "def forward(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:\n"
        "    return (x + w).to(torch.float32)\n"
    )
    tl_ledger = build_objective_physical_ledger(
        analysis_mode="exact_modulo_z3",
        is_legal=True,
        steady_state_ii=18,
        pipe_cycles={"AIV0.PIPE_V": 40},
        waterlines={
            "UB": build_scope_waterline(
                scope="UB",
                capacity_bytes=253952,
                explicit_buffers={"tmp": 32768},
                implicit_buffers={},
            )
        },
        gm_roundtrip_bytes=4096,
    )

    analyzer = PyTorchSemanticAnalyzer()
    semantics = analyzer.analyze_source(ar_golden, entry_name="forward")
    raw_pkt = analyzer.build_universal_blueprint(
        semantics,
        anchor_dims={"M": 4, "D": 64},
        representative_case=RepresentativeCase(
            case_id=1,
            input_shapes=((4, 64), (4, 64)),
            input_dtypes=("float32", "float32"),
            output_dtype="float32",
            output_shapes=((4, 64),),
        ),
    )
    exact_pkt = raw_pkt
    assert exact_pkt.shape_exact_match is True
    exact_nav = format_blueprint_delta_section(exact_pkt, current_ledger=tl_ledger)
    assert "[Structural Reference Only:" not in exact_nav
    assert "ub_peak_bytes:" in exact_nav
    assert "gm_roundtrip_bytes:" in exact_nav
    assert "delta=" in exact_nav

    fallback_pkt = dataclasses.replace(
        exact_pkt,
        shape_exact_match=False,
        requested_anchor_dims={"M": 8, "N": 1000},
        fallback_reason="shape_mismatch: representative_anchor",
    )
    assert fallback_pkt.fallback_reason is not None
    gated_nav = format_blueprint_delta_section(fallback_pkt, current_ledger=tl_ledger)
    assert "[Structural Reference Only:" in gated_nav
    assert "fallback_reason:" in gated_nav
    assert "scope_utilization[" in gated_nav
    assert "%" in gated_nav
    assert "ub_peak_bytes:" not in gated_nav
    assert "gm_roundtrip_bytes:" not in gated_nav
    assert "delta=" not in gated_nav


def test_up3_twin_reference_dual_profile_runs_both_profiles() -> None:
    """Verify dual_profile executes both nominal and stress CPU parity checks."""
    bias_golden = (
        "import torch\n"
        "def fwd(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:\n"
        "    return (x + w).to(torch.float32)\n"
    )
    seen_w: list[torch.Tensor] = []

    def _capture_exact(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        seen_w.append(w.detach().clone())
        return x + w

    ok_exact, msg_exact = verify_reference_cpu_parity(
        _capture_exact,
        golden_source=bias_golden,
        input_specs=[((4, 8), "float32"), ((4, 8), "float32")],
        dual_profile=True,
    )
    assert ok_exact is True, msg_exact
    assert "stress_profile=ok" in msg_exact
    assert len(seen_w) == 2
    assert not torch.allclose(seen_w[0], seen_w[1])
