"""Tests for NP3 runtime twin bridge, CPU oracle equivalence, and HIR/TIR navigation."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
from pathlib import Path

import torch

from tilefoundry.cli.source import load_authored_ir
from tilefoundry.runtime import RuntimeModule
from tilefoundry_ascend.agent import (
    BayesianRuleTracker,
    GateFunnel,
    MockLLMBackend,
    PortfolioRunner,
    TwoLaneBeamSearch,
    build_tilelang_runtime_twin,
    format_blueprint_delta_section,
    verify_twin_cpu_reference,
)
from tilefoundry_ascend.agent.orchestrator import (
    BeamCandidate,
    StageBResult,
    StageCResult,
    compute_static_dynamic_residual,
    extract_structural_signature,
)
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import compile_hir_blueprint
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    build_objective_physical_ledger,
    build_scope_waterline,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AR_MODEL_PATH = _REPO_ROOT / "examples" / "ascend950pr_ops" / "attention_residual" / "model.py"
_AR_SCHED_PATH = (
    _REPO_ROOT / "examples" / "ascend950pr_ops" / "attention_residual" / "scheduled.py"
)
_MLA_MODEL_PATH = _REPO_ROOT / "examples" / "ascend950pr_ops" / "mla" / "model.py"
_MLA_SCHED_PATH = _REPO_ROOT / "examples" / "ascend950pr_ops" / "mla" / "scheduled.py"

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


def test_build_tilelang_runtime_twin_signature_and_call() -> None:
    """Verify @runtime_module Twin exposes exact signature and returns CPU tensors."""
    authored_mod = load_authored_ir(str(_AR_MODEL_PATH))

    def _f32_runner(v_stack, x, w, gamma):
        return _attention_residual_torch_ref(v_stack, x, w, gamma).to(torch.float32)

    twin_cls = build_tilelang_runtime_twin(authored_mod, _f32_runner)
    assert issubclass(twin_cls, RuntimeModule)

    method = getattr(twin_cls, "forward")
    assert method.__name__ == "forward"
    assert getattr(method, "_tilefoundry_runtime_func", False) is True
    sig = inspect.signature(method)
    assert list(sig.parameters.keys()) == ["self", "v_stack", "x", "w", "gamma"]

    twin = twin_cls()
    v_stack = torch.randn((4, 1, 4, 64), dtype=torch.bfloat16)
    x = torch.randn((1, 4, 64), dtype=torch.bfloat16)
    w = torch.randn((64,), dtype=torch.bfloat16)
    gamma = torch.randn((64,), dtype=torch.bfloat16)

    out_direct = twin.forward(v_stack, x, w, gamma)
    out_call = twin(v_stack, x, w, gamma)
    assert out_direct.device.type == "cpu"
    assert out_direct.dtype == torch.bfloat16
    assert tuple(out_direct.shape) == (1, 4, 64)
    torch.testing.assert_close(out_direct.to(torch.float32), out_call.to(torch.float32))


def test_verify_twin_cpu_reference_attention_residual_match_and_mismatch() -> None:
    """Verify CPU evaluator oracle passes matching runner and fails perturbed runner."""
    ar_hir = _AR_MODEL_PATH.read_text(encoding="utf-8")

    ok, ok_msg = verify_twin_cpu_reference(
        ar_hir,
        _attention_residual_torch_ref,
        test_dims={"N": 4, "B": 1, "S": 4, "D": 64},
    )
    assert ok is True, ok_msg
    assert "twin_cpu_reference_passed" in ok_msg

    def _perturbed_runner(v_stack, x, w, gamma):
        return _attention_residual_torch_ref(v_stack, x, w, gamma) + 0.25

    bad_ok, bad_msg = verify_twin_cpu_reference(
        ar_hir,
        _perturbed_runner,
        test_dims={"N": 4, "B": 1, "S": 4, "D": 64},
    )
    assert bad_ok is False
    assert "twin_numeric_mismatch" in bad_msg


def test_verify_twin_cpu_reference_with_dimvars() -> None:
    """Verify DimVar HIR modules specialize cleanly in verify_twin_cpu_reference."""
    dimvar_hir = (
        "from __future__ import annotations\n"
        "from tilefoundry.dsl import *\n"
        "from tilefoundry_ascend import AscendTarget\n"
        "TOPOS = (Topology('block', 1), Topology('aic', 1), "
        "Topology('aiv', 2), Topology('lane', 64))\n"
        "M = DimVar('M', lo=1, hi=512)\n"
        "N = DimVar('N', lo=1, hi=512)\n"
        "@module(entry='add_scale', target=AscendTarget('ascend.950pr'), topologies=TOPOS)\n"
        "class DimAddMod:\n"
        "    @func\n"
        "    def add_scale(a: Tensor[(M, N), 'f32'], b: Tensor[(M, N), 'f32']) "
        "-> Tensor[(M, N), 'f32']:\n"
        "        return tf.cast(a + b, dtype='f32')\n"
    )
    ok, msg = verify_twin_cpu_reference(
        dimvar_hir,
        lambda a, b: a + b,
        test_dims={"M": 8, "N": 16, "EXTRA": 99},
    )
    assert ok is True, msg

    bad_ok, bad_msg = verify_twin_cpu_reference(
        dimvar_hir,
        lambda a, b: a - b,
        test_dims={"M": 8, "N": 16},
    )
    assert bad_ok is False
    assert "twin_numeric_mismatch" in bad_msg


def test_format_blueprint_delta_section_and_portfolio_roofline_eta() -> None:
    """Verify navigation formatting, ledger delta, and roofline_eta residual threading."""
    mla_hir = _MLA_MODEL_PATH.read_text(encoding="utf-8")
    mla_sched = _MLA_SCHED_PATH.read_text(encoding="utf-8")
    packet = compile_hir_blueprint(
        mla_hir,
        scheduled_hir_source=mla_sched,
        anchor_dims={"B": 1, "S": 16, "S_kv": 16, "N_q": 4, "N_kv": 1},
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
        assert "### TIR BLUEPRINT VS CURRENT TILELANG LEDGER DELTA" in req.user_prompt
    for cand in report.evaluated_candidates:
        assert cand.residual.predicted_bottleneck_util >= 0.0
        assert cand.residual.unexplained_gap_ratio > 0.0


def test_format_blueprint_delta_section_gates_cross_scale_deltas() -> None:
    """Verify shape mismatch suppresses raw byte/cycle deltas in favor of normalized usage."""
    ar_hir = _AR_MODEL_PATH.read_text(encoding="utf-8")
    ar_sched = _AR_SCHED_PATH.read_text(encoding="utf-8")
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

    exact_pkt = compile_hir_blueprint(
        ar_hir,
        scheduled_hir_source=ar_sched,
        anchor_dims={"N": 4, "B": 1, "S": 4, "D": 64},
    )
    assert exact_pkt.shape_exact_match is True
    exact_nav = format_blueprint_delta_section(exact_pkt, current_ledger=tl_ledger)
    assert "[Structural Reference Only:" not in exact_nav
    assert "ub_peak_bytes:" in exact_nav
    assert "gm_roundtrip_bytes:" in exact_nav
    assert "delta=" in exact_nav

    fallback_pkt = compile_hir_blueprint(
        ar_hir,
        scheduled_hir_source=ar_sched,
        anchor_dims={"N": 8, "B": 1, "S": 4, "D": 1000},
    )
    assert fallback_pkt.shape_exact_match is False
    assert fallback_pkt.resolved_hir_dims == {"N": 4, "B": 1, "S": 4, "D": 64}
    assert fallback_pkt.fallback_reason is not None
    gated_nav = format_blueprint_delta_section(fallback_pkt, current_ledger=tl_ledger)
    assert "[Structural Reference Only:" in gated_nav
    assert "fallback_reason:" in gated_nav
    assert "scope_utilization[" in gated_nav
    assert "%" in gated_nav
    assert "ub_peak_bytes:" not in gated_nav
    assert "gm_roundtrip_bytes:" not in gated_nav
    assert "delta=" not in gated_nav


def test_up3_twin_reference_contract_profile_and_dual_profile() -> None:
    """Verify contract profile zero-inits w and dual_profile logs stress deltas."""
    import torch  # noqa: PLC0415

    w_hir = (
        "from __future__ import annotations\n"
        "from tilefoundry.dsl import *\n"
        "from tilefoundry_ascend import AscendTarget\n"
        "TOPOS = (Topology('block', 1), Topology('aic', 1), "
        "Topology('aiv', 2), Topology('lane', 64))\n"
        "@module(entry='fwd', target=AscendTarget('ascend.950pr'), topologies=TOPOS)\n"
        "class BiasMod:\n"
        "    @func\n"
        "    def fwd(x: Tensor[(4, 8), 'f32'], w: Tensor[(4, 8), 'f32']) -> Tensor[(4, 8), 'f32']:\n"
        "        return tf.cast(x + w, dtype='f32')\n"
    )
    ok_fast, msg_fast = verify_twin_cpu_reference(
        w_hir,
        lambda x, _w: x,
        dual_profile=True,
    )
    assert ok_fast is True, msg_fast
    assert "stress_profile=warn" in msg_fast

    seen_w: list[torch.Tensor] = []

    def _capture_exact(x, w):
        seen_w.append(w.detach().clone())
        return x + w

    ok_exact, msg_exact = verify_twin_cpu_reference(
        w_hir,
        _capture_exact,
        dual_profile=True,
    )
    assert ok_exact is True, msg_exact
    assert "stress_profile=ok" in msg_exact
    assert len(seen_w) == 2
    assert torch.all(seen_w[0] == 0)
    assert torch.any(seen_w[1] != 0)
