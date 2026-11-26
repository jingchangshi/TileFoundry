"""Offline end-to-end tests driving dual-loop evolution from PyTorch golden fixtures."""

import asyncio
import json

from tilefoundry_ascend.agent import (
    BayesianRuleTracker,
    GateFunnel,
    MockLLMBackend,
    run_e2e_evolution,
)
from tilefoundry_ascend.agent.orchestrator import (
    StageBResult,
    StageCResult,
    compute_static_dynamic_residual,
)

_AR_GOLDEN = """
import math
import torch

def forward(v_stack, x, w, gamma):
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

_MLA_GOLDEN = """
import torch

def forward(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)
"""

_BASELINE_DSL = """
import tilelang.ascend.language as T

def make_kernel():
    @T.prim_func
    def main(
        X: T.Tensor((128, 128), "bfloat16"),
        W: T.Tensor((128, 128), "bfloat16"),
        Y: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(1):
            x_l1 = T.alloc_l1((128, 64), "bfloat16")
            w_l1 = T.alloc_l1((128, 64), "bfloat16")
            acc = T.alloc_l0c((128, 128), "float32")
            for kt in T.Pipelined(2, num_stages=1):
                T.copy(X[0:128, kt * 64 : (kt + 1) * 64], x_l1)
                T.copy(W[0:128, kt * 64 : (kt + 1) * 64], w_l1)
                T.gemm(x_l1, w_l1, acc, transpose_B=True, clear_accum=kt == 0)
            T.copy(acc, Y[0:128, 0:128])
    return main
"""

_RULE_IMPROVED_DSL = """
import tilelang.ascend.language as T

def make_kernel():
    @T.prim_func
    def main(
        X: T.Tensor((128, 128), "bfloat16"),
        W: T.Tensor((128, 128), "bfloat16"),
        Y: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(4):
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

_CREATIVE_SSBUF_RESPONSE = (
    "# target_bottleneck: Reduce cross-core sync_stall_ratio with 2-stage double buffering.\n"
    "# structural_hypothesis: Stage dynamic rescale descriptors through 8B-aligned SSBUF.\n"
    "```python\n"
    "import tilelang.ascend.language as T\n\n"
    "def make_kernel():\n"
    "    @T.prim_func\n"
    "    def main(\n"
    '        X: T.Tensor((128, 128), "bfloat16"),\n'
    '        W: T.Tensor((128, 128), "bfloat16"),\n'
    '        Y: T.Tensor((128, 128), "float32"),\n'
    "    ):\n"
    "        with T.MixedKernel(8) as (_, sid):\n"
    '            ss = T.alloc_ss((2,), "uint32")\n'
    '            x_l1 = T.alloc_l1((128, 64), "bfloat16")\n'
    '            w_l1 = T.alloc_l1((128, 64), "bfloat16")\n'
    '            acc = T.alloc_l0c((128, 128), "float32")\n'
    '            ss[sid] = T.Cast("uint32", 0)\n'
    "            for kt in T.Pipelined(2, num_stages=2):\n"
    "                T.copy(X[0:128, kt * 64 : (kt + 1) * 64], x_l1)\n"
    "                T.copy(W[0:128, kt * 64 : (kt + 1) * 64], w_l1)\n"
    "                T.gemm(x_l1, w_l1, acc, transpose_B=True, clear_accum=kt == 0)\n"
    "            T.copy(acc, Y[0:128, 0:128])\n"
    "    return main\n"
    "```\n"
)


def _surrogate_device_runner(dsl_source: str, stage_a) -> StageCResult:
    assert stage_a.ledger is not None
    if "alloc_ss" in dsl_source:
        lat = 52.0
        tel = {"dur_us": lat, "aic_mad": 0.78, "aiv_vec": 0.34, "aiv_mte2": 0.14}
    elif "num_stages=2" in dsl_source:
        lat = 71.0
        tel = {"dur_us": lat, "aic_mad": 0.68, "aiv_vec": 0.30, "aiv_mte2": 0.22}
    else:
        lat = 115.0
        tel = {"dur_us": lat, "aic_mad": 0.42, "aiv_vec": 0.22, "aiv_mte2": 0.48}
    return StageCResult(
        passed=True,
        accuracy_passed=True,
        measured_latency_us=lat,
        residual=compute_static_dynamic_residual(stage_a.ledger, tel),
        telemetry=tel,
    )


def test_e2e_evolution_offline_golden_to_converged_trajectories() -> None:
    """Verify multi-round evolution from PyTorch golden forward() fixtures."""

    def _make_backend() -> MockLLMBackend:
        return MockLLMBackend(
            responses={
                "exploit_rule": [
                    _BASELINE_DSL,
                    _RULE_IMPROVED_DSL,
                    _RULE_IMPROVED_DSL,
                    _RULE_IMPROVED_DSL,
                    _RULE_IMPROVED_DSL,
                ],
                "explore_creative": [
                    _CREATIVE_SSBUF_RESPONSE,
                    _CREATIVE_SSBUF_RESPONSE,
                ],
            },
            default_response=_RULE_IMPROVED_DSL,
        )

    funnel = GateFunnel(
        compile_runner=lambda _dsl: StageBResult(passed=True),
        device_runner=_surrogate_device_runner,
    )

    async def _run_both():
        ar_traj = await run_e2e_evolution(
            problem_name="forward",
            hir_source=_AR_GOLDEN,
            anchor_dims={"B": 1, "S": 4, "D": 64, "N": 4},
            shape_contract="make_kernel() returning fixed static benchmark tensor signature",
            backend=_make_backend(),
            funnel=funnel,
            tracker=BayesianRuleTracker.from_seed_yaml(),
            num_rounds=2,
            k_rule_workers=2,
        )
        mla_traj = await run_e2e_evolution(
            problem_name="forward",
            hir_source=_MLA_GOLDEN,
            anchor_dims={"B": 1, "S_q": 16, "S_kv": 16, "N_q": 4, "N_kv": 1, "D_qk": 64},
            shape_contract="make_kernel() returning fixed static benchmark tensor signature",
            backend=_make_backend(),
            funnel=funnel,
            tracker=BayesianRuleTracker.from_seed_yaml(),
            num_rounds=2,
            k_rule_workers=2,
        )
        return ar_traj, mla_traj

    ar_traj, mla_traj = asyncio.run(_run_both())
    for traj in (ar_traj, mla_traj):
        assert traj.total_rounds == 2
        assert traj.baseline_latency_us == 115.0
        assert traj.best_latency_us == 52.0
        assert traj.speedup_vs_baseline > 2.0
        assert "alloc_ss" in traj.best_dsl_source
        assert len(traj.rounds) == 2
        assert traj.rounds[0].best_so_far_latency_us < traj.baseline_latency_us
        payload = json.loads(traj.to_json())
        assert payload["best_candidate_id"].endswith("_creative")
        assert len(payload["rounds"]) == 2


def test_np3_golden_navigation_router_off_and_universal_blueprint(monkeypatch) -> None:
    """Verify router-off PyTorch golden navigation and zero capability_gate calls."""
    import tilefoundry_ascend.agent.router.capability_gate as cap_gate  # noqa: PLC0415
    from tilefoundry_ascend.agent.orchestrator import (  # noqa: PLC0415
        TwoLaneBeamSearch,
        build_hir_seed_request,
    )
    from tilefoundry_ascend.ascend.sil.pytorch_frontend import (  # noqa: PLC0415
        PyTorchSemanticAnalyzer,
    )
    from tilefoundry_ascend.ascend.target import AscendTarget  # noqa: PLC0415

    router_calls: list[str] = []

    def _forbid_gate(*_args, **_kwargs):
        router_calls.append("gate")
        raise AssertionError("CapabilityGate.route must not be called")

    def _forbid_route(*_args, **_kwargs):
        router_calls.append("route")
        raise AssertionError("route_execution_track must not be called")

    monkeypatch.setattr(cap_gate.CapabilityGate, "route", _forbid_gate)
    monkeypatch.setattr(cap_gate, "route_execution_track", _forbid_route)

    analyzer = PyTorchSemanticAnalyzer(AscendTarget("ascend.950pr"))
    semantics = analyzer.analyze_source(_AR_GOLDEN, entry_name="forward")
    blueprint = analyzer.build_universal_blueprint(
        semantics,
        {"M": 16, "D": 64},
    )
    seed_req = build_hir_seed_request(
        _AR_GOLDEN,
        "make_kernel()",
        blueprint=blueprint,
    )
    assert "### L0 RECOMMENDED EXECUTION TRACK" not in seed_req.user_prompt
    assert "### LAYER 1: HARDWARE REDLINES & 7-FACET SHELL CONTRACT" in seed_req.user_prompt
    assert "### LAYER 3: PHYSICAL ROOFLINE & BLUEPRINT DELTA" in seed_req.user_prompt
    assert "### LAYER 2: TARGET DIALECT API SURFACE" in seed_req.user_prompt

    backend = MockLLMBackend(
        responses={
            "exploit_rule": [_BASELINE_DSL, _RULE_IMPROVED_DSL, _RULE_IMPROVED_DSL],
            "explore_creative": [_CREATIVE_SSBUF_RESPONSE],
        },
        default_response=_RULE_IMPROVED_DSL,
    )
    funnel = GateFunnel(
        compile_runner=lambda _dsl: StageBResult(passed=True),
        device_runner=_surrogate_device_runner,
    )
    beam = TwoLaneBeamSearch(exploit_width=1, stepping_stone_width=1)
    traj = asyncio.run(
        run_e2e_evolution(
            problem_name="forward",
            hir_source=_AR_GOLDEN,
            anchor_dims={"M": 4, "D": 64},
            shape_contract="make_kernel()",
            backend=backend,
            funnel=funnel,
            beam=beam,
            num_rounds=1,
            k_rule_workers=2,
        )
    )
    assert router_calls == []
    assert backend.calls
    for req in backend.calls:
        assert "### L0 RECOMMENDED EXECUTION TRACK" not in req.user_prompt
    assert "### LAYER 3: PHYSICAL ROOFLINE & BLUEPRINT DELTA" in backend.calls[0].user_prompt
    assert "### LAYER 2: TARGET DIALECT API SURFACE" in backend.calls[0].user_prompt
    assert traj.feature_vector == {}
    assert traj.track == "unconstrained"
    assert beam.exploit_beam[0].features is None
    assert traj.hir_blueprint_summary
    assert traj.hir_blueprint_summary["roofline"]["ideal_roofline_us"] > 0.0
    payload = json.loads(traj.to_json())
    assert payload["feature_vector"] == {}
    assert payload["hir_blueprint_summary"]["entry_name"] == "forward"
