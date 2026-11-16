"""Offline end-to-end tests driving dual-loop evolution from attention_residual and mla HIR."""

import asyncio
import json
from pathlib import Path

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

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AR_MODEL_PATH = _REPO_ROOT / "examples" / "ascend950pr_ops" / "attention_residual" / "model.py"
_MLA_MODEL_PATH = _REPO_ROOT / "examples" / "ascend950pr_ops" / "mla" / "model.py"

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

_CREATIVE_SSBUF_JSON = json.dumps(
    {
        "physical_hypothesis": "Stage dynamic rescale descriptors through 8B-aligned SSBUF and dual_copy.",
        "expected_bottleneck_shift": "Reduce cross-core sync_stall_ratio with 2-stage double buffering.",
        "dsl_source": """
import tilelang.ascend.language as T

def make_kernel():
    @T.prim_func
    def main(
        X: T.Tensor((128, 128), "bfloat16"),
        W: T.Tensor((128, 128), "bfloat16"),
        Y: T.Tensor((128, 128), "float32"),
    ):
        with T.MixedKernel(8) as (_, sid):
            ss = T.alloc_ss((2,), "uint32")
            x_l1 = T.alloc_l1((128, 64), "bfloat16")
            w_l1 = T.alloc_l1((128, 64), "bfloat16")
            acc = T.alloc_l0c((128, 128), "float32")
            ss[sid] = T.Cast("uint32", 0)
            for kt in T.Pipelined(2, num_stages=2):
                T.copy(X[0:128, kt * 64 : (kt + 1) * 64], x_l1)
                T.copy(W[0:128, kt * 64 : (kt + 1) * 64], w_l1)
                T.gemm(x_l1, w_l1, acc, transpose_B=True, clear_accum=kt == 0)
            T.copy(acc, Y[0:128, 0:128])
    return main
""".strip(),
    }
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


def test_e2e_evolution_offline_hir_to_converged_trajectories() -> None:
    """Verify multi-round evolution from real attention_residual and mla HIR files."""
    ar_hir = _AR_MODEL_PATH.read_text(encoding="utf-8")
    mla_hir = _MLA_MODEL_PATH.read_text(encoding="utf-8")
    assert "AttentionResidualModel" in ar_hir
    assert "MlaModel" in mla_hir

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
                    _CREATIVE_SSBUF_JSON,
                    _CREATIVE_SSBUF_JSON,
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
            problem_name="attention_residual_hir",
            hir_source=ar_hir,
            shape_contract="make_kernel() returning fixed static benchmark tensor signature",
            backend=_make_backend(),
            funnel=funnel,
            tracker=BayesianRuleTracker.from_seed_yaml(),
            num_rounds=2,
            k_rule_workers=2,
        )
        mla_traj = await run_e2e_evolution(
            problem_name="mla_hir",
            hir_source=mla_hir,
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
