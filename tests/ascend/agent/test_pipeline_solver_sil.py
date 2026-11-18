"""Clean SIL-orchestrator integration tests replacing legacy sil placeholder."""

from __future__ import annotations

import asyncio
import json

from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    GateFunnel,
    StageBResult,
    StageCResult,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import MockLLMBackend
from tilefoundry_ascend.agent.orchestrator.portfolio_runner import (
    PortfolioRunner,
    build_creative_request,
    build_exploit_request,
    build_syntax_repair_request,
)
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    BeamCandidate,
    TwoLaneBeamSearch,
    compute_static_dynamic_residual,
    extract_structural_signature,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger
from tilefoundry_ascend.ascend.sil import (
    AdvisoryEnvelope,
    OpAdvisoryExtractor,
    VerificationReport,
    synthesize_default_policy,
    verify_policy_envelope,
)

_SAMPLE_DSL = """
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


def test_sil_advisory_and_verification_flow_into_portfolio_prompts() -> None:
    """Verify OpAdvisoryExtractor, SILVerifier, and PortfolioRunner SIL diagnostic injection."""
    extractor = OpAdvisoryExtractor()
    advisory: AdvisoryEnvelope = extractor.extract("rms_norm")
    policy_env = synthesize_default_policy(advisory)
    report: VerificationReport = verify_policy_envelope(policy_env, advisory)
    assert report.passed is True

    synthetic_ver = VerificationReport(
        passed=False,
        errors=("ub_256kb_unlock invalid in simtvf mode",),
        warnings=("512B UB stride bank conflict risk",),
        suggested_fixes={"ub_bank_pad_datablocks": 1, "ub_rw_shift_datablocks": 8},
    )

    ledger = build_objective_physical_ledger(
        analysis_mode="exact_modulo_z3",
        is_legal=True,
        steady_state_ii=20,
        pipe_cycles={"AIC.PIPE_M": 20, "AIV0.PIPE_V": 40},
        waterlines={},
    )
    parent = BeamCandidate(
        candidate_id="seed_0",
        round_idx=0,
        worker_kind="seed",
        rule_id=None,
        dsl_source=_SAMPLE_DSL,
        track="unconstrained",
        ledger=ledger,
        residual=compute_static_dynamic_residual(ledger, {"dur_us": 20.0}),
        latency_us=20.0,
        accuracy_passed=True,
        structural_signature=extract_structural_signature(_SAMPLE_DSL),
    )

    tracker = BayesianRuleTracker.from_seed_yaml()
    rule = tracker.active_rules()[0]
    exp_req = build_exploit_request(
        parent, rule, sil_advisory=advisory, sil_verification=synthetic_ver
    )
    cre_req = build_creative_request(
        parent, sil_advisory=advisory, sil_verification=synthetic_ver
    )
    rep_req = build_syntax_repair_request(
        _SAMPLE_DSL, "ub overflow", sil_verification=synthetic_ver
    )

    assert "### SIL ADVISORY ENVELOPE" in exp_req.user_prompt
    assert "rms_norm" in exp_req.user_prompt
    assert "### SIL VERIFICATION DIAGNOSTICS" in exp_req.user_prompt
    assert "ub_bank_pad_datablocks" in exp_req.user_prompt
    assert "### SIL ADVISORY ENVELOPE" in cre_req.user_prompt
    assert "### SIL VERIFICATION DIAGNOSTICS" in rep_req.user_prompt

    creative_reply = json.dumps(
        {
            "physical_hypothesis": "Apply +1 DataBlock UB pad.",
            "target_local_metric": "aiv_vec",
            "dsl_source": _SAMPLE_DSL.strip(),
        }
    )
    backend = MockLLMBackend(
        responses={"exploit_rule": [_SAMPLE_DSL], "explore_creative": [creative_reply]},
        default_response=_SAMPLE_DSL,
    )
    funnel = GateFunnel(
        compile_runner=lambda _dsl: StageBResult(passed=True),
        device_runner=lambda _dsl, st_a: StageCResult(
            passed=True,
            accuracy_passed=True,
            measured_latency_us=12.0,
            residual=compute_static_dynamic_residual(st_a.ledger, {"dur_us": 12.0}),
            telemetry={"dur_us": 12.0, "aiv_vec": 0.75},
        ),
    )
    runner = PortfolioRunner(
        backend=backend,
        funnel=funnel,
        tracker=tracker,
        beam=TwoLaneBeamSearch(),
        k_rule_workers=1,
    )
    round_report = asyncio.run(
        runner.run_round(
            parent,
            round_idx=1,
            factory_name="make_kernel",
            sil_advisory=advisory,
            sil_verification=synthetic_ver,
        )
    )
    assert round_report.evaluated_candidates
    assert f"ce:{advisory.topology_axes.compute_engine}" in tracker.axis_posteriors
    for req in backend.calls:
        assert "### SIL ADVISORY ENVELOPE" in req.user_prompt
