"""Tests for CostModelAdvisor isolation, shadow mode, operator filtering, and zero-cost lattice."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from tilefoundry_ascend.agent.campaign.cli import build_arg_parser
from tilefoundry_ascend.agent.cost_advisor import (
    CostModelAdvisor,
    NullCostModelAdvisor,
    PluginCostModelAdvisor,
    create_cost_advisor,
)
from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.orchestrator.gate_funnel import GateFunnel, StageBResult, StageCResult
from tilefoundry_ascend.agent.orchestrator.llm_client import GenerationRequest, KernelLLMBackend
from tilefoundry_ascend.agent.orchestrator.portfolio_runner import PortfolioRunner
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    BeamCandidate,
    StaticDynamicResidual,
    compute_static_dynamic_residual,
)
from tilefoundry_ascend.agent.orchestrator.tilesize_prescreen import (
    LatticeSweepCandidate,
    sweep_executable_lattice_zero_llm,
)
from tilefoundry_ascend.ascend.cost_model import (
    CandidateArtifactRecord,
    CaseObservationRecord,
    CostGateDecision,
    CostModelMode,
    ExperienceLake,
    HybridAscendCostModel,
    evaluate_cost_gate_policy,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes
from tilefoundry_ascend.ascend.target import AscendTarget


class _DummyBackend(KernelLLMBackend):
    """Minimal offline backend stub for PortfolioRunner prompt inspection."""

    async def generate(self, request: GenerationRequest) -> str:
        _ = request
        return "{}"


@dataclass(frozen=True)
class _AnchorStub:
    case_num: int
    regime_tag: str
    case_dims: dict[str, int]
    t_hw_us: float


_SYNTHETIC_CUBE_DSL = """
import tilelang
import tilelang.ascend.language as T

def _select_cube_config(m_dim: int = 256, n_dim: int = 256, k_dim: int = 128) -> dict[str, int]:
    return {"bm": 64, "bn": 64, "bk": 32, "num_stages": 1}

def _build_cube_prim_func(
    m_dim: int = 256,
    n_dim: int = 256,
    k_dim: int = 128,
    bm: int = 64,
    bn: int = 64,
    bk: int = 32,
    num_stages: int = 1,
    **_extra,
):
    stages = max(1, int(num_stages))

    @T.prim_func
    def main(
        A: T.Tensor((m_dim, k_dim), "float16"),
        B: T.Tensor((n_dim, k_dim), "float16"),
        C: T.Tensor((m_dim, n_dim), "float32"),
    ):
        with T.Kernel((m_dim // bm) * (n_dim // bn)) as cid:
            n_blocks = n_dim // bn
            bx = cid // n_blocks
            by = cid % n_blocks
            a_l1 = T.alloc_l1((bm, bk), "float16")
            b_l1 = T.alloc_l1((bn, bk), "float16")
            c_l0c = T.alloc_l0c((bm, bn), "float32")
            for ko in T.Pipelined(k_dim // bk, num_stages=stages):
                T.copy(A[bx * bm : (bx + 1) * bm, ko * bk : (ko + 1) * bk], a_l1)
                T.copy(B[by * bn : (by + 1) * bn, ko * bk : (ko + 1) * bk], b_l1)
                T.gemm(a_l1, b_l1, c_l0c, transpose_B=True, clear_accum=(ko == 0))
            T.copy(c_l0c, C[bx * bm : (bx + 1) * bm, by * bn : (by + 1) * bn])

    return main
"""


def test_default_off_mode_and_null_advisor_zero_leakage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify default mode=off never initializes HybridAscendCostModel or leaks guidance."""
    default_decision = evaluate_cost_gate_policy(round_idx=1, incumbent_roofline_eta=0.1)
    assert default_decision.mode == CostModelMode.OFF
    assert default_decision.engaged is False

    args = build_arg_parser().parse_args(["--operator", "rms_norm"])
    assert args.cost_model_mode == "off"
    shadow_args = build_arg_parser().parse_args(
        ["--operator", "rms_norm", "--cost-model-mode", "shadow"]
    )
    assert shadow_args.cost_model_mode == "shadow"

    def _forbid(*_a: object, **_k: object) -> object:
        raise AssertionError("HybridAscendCostModel must not be initialized or called in off mode")

    monkeypatch.setattr(HybridAscendCostModel, "__init__", _forbid)
    monkeypatch.setattr(HybridAscendCostModel, "predict", _forbid)

    advisor = create_cost_advisor("off", calibration_path=tmp_path / "cost_model_calibration.jsonl")
    assert isinstance(advisor, NullCostModelAdvisor)
    assert isinstance(advisor, CostModelAdvisor)
    assert advisor.mode == CostModelMode.OFF
    assert advisor.underlying_cost_model is None
    assert advisor.predictions_invoked == 0

    assert advisor.sync_experience_lake(tmp_path) == {}
    decision = advisor.evaluate_round_policy(
        enable_prescreen_flag=True,
        round_idx=1,
        stagnation_rounds=0,
        incumbent_roofline_eta=0.2,
        target_roofline_eta=0.85,
        prev_local_rel_gain=None,
        mean_recoverable_excess_us=20.0,
    )
    assert decision.mode == CostModelMode.OFF
    assert decision.engaged is False
    assert decision.allow_prompt_guidance is False
    assert decision.allow_candidate_pruning is False

    assert advisor.predict_candidate_us({"cube_flops": 1e9, "gm_bytes": 1e6}) is None
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=16,
        pipe_cycles={"AIC.PIPE_M": 400, "AIV0.MTE2": 1200},
        gm_roundtrip_bytes=32768,
        grid_blocks=28,
        waterlines={},
    )
    assert advisor.predict_anchor_mean_us(ledger, anchors=[]) == 0.0
    assert advisor.predictions_invoked == 0

    assert (
        advisor.build_prompt_guidance(
            decision=decision,
            base_cfg={"bm": 64, "bn": 128, "bk": 64},
            dims={"M": 256, "N": 256, "K": 128},
        )
        == ""
    )

    kept, rejected = advisor.filter_stage_a_passers(
        ["cand_slow", "cand_fast"],
        decision=decision,
        parent_pred_us=10.0,
        parent_axes=None,
        parent_workspace_stages=0,
        candidate_pred_us_by_id={"cand_slow": 500.0, "cand_fast": 5.0},
        candidate_axes_by_id={},
        candidate_workspace_stages_by_id={},
        creative_ids=(),
    )
    assert kept == ["cand_slow", "cand_fast"]
    assert rejected == []

    advisor.observe_surrogate_batch(
        observations=[{"workload": {"cube_flops": 1e9, "gm_bytes": 1e6}, "measured_us": 25.0}]
    )
    assert not (tmp_path / "cost_model_calibration.jsonl").exists()

    mem_res = StaticDynamicResidual(
        measured_latency_us=20.0,
        predicted_bottleneck_util=0.5,
        measured_aic_mac_ratio=0.1,
        measured_aiv_vec_ratio=0.1,
        measured_mte2_wait_ratio=0.7,
        measured_sync_stall_ratio=0.05,
        unexplained_gap_ratio=0.1,
    )
    comp_res = StaticDynamicResidual(
        measured_latency_us=20.0,
        predicted_bottleneck_util=0.5,
        measured_aic_mac_ratio=0.8,
        measured_aiv_vec_ratio=0.1,
        measured_mte2_wait_ratio=0.1,
        measured_sync_stall_ratio=0.05,
        unexplained_gap_ratio=0.1,
    )
    sync_res = StaticDynamicResidual(
        measured_latency_us=20.0,
        predicted_bottleneck_util=0.5,
        measured_aic_mac_ratio=0.1,
        measured_aiv_vec_ratio=0.1,
        measured_mte2_wait_ratio=0.1,
        measured_sync_stall_ratio=0.75,
        unexplained_gap_ratio=0.1,
    )
    fake_pred = SimpleNamespace(dominant_bound="compiler_spill")
    assert advisor.resolve_dominant_bound(ledger, mem_res, fallback_cost_pred=fake_pred) == "memory_bound"
    assert advisor.resolve_dominant_bound(ledger, comp_res, fallback_cost_pred=fake_pred) == "compute_bound"
    assert advisor.resolve_dominant_bound(ledger, sync_res, fallback_cost_pred=fake_pred) == "sync_bound"
    assert advisor.resolve_dominant_bound(ledger, None, fallback_cost_pred=fake_pred) == "memory_bound"

    parent = BeamCandidate(
        candidate_id="seed_0",
        round_idx=0,
        worker_kind="seed",
        rule_id=None,
        dsl_source=_SYNTHETIC_CUBE_DSL,
        track="developer",
        ledger=ledger,
        residual=compute_static_dynamic_residual(ledger, {"dur_us": 40.0}),
        latency_us=40.0,
        accuracy_passed=True,
        features={},
    )
    runner = PortfolioRunner(
        backend=_DummyBackend(),
        funnel=GateFunnel(
            compile_runner=lambda _s: StageBResult(passed=True),
            device_runner=lambda _s, st: StageCResult(
                passed=True,
                accuracy_passed=True,
                measured_latency_us=20.0,
                residual=compute_static_dynamic_residual(st.ledger, {"dur_us": 20.0}),
                telemetry={"dur_us": 20.0},
            ),
        ),
        tracker=BayesianRuleTracker.from_seed_yaml(),
        enable_tilesize_prescreen=True,
    )
    specs = runner.plan_workers(
        parent,
        round_idx=1,
        factory_kwargs={"bm": 64, "bn": 128, "bk": 64},
    )
    creative = next(spec for spec in specs if spec.worker_kind == "creative_explore")
    assert "TILE PRESCREEN GUIDANCE" not in creative.request.user_prompt


def test_shadow_mode_observes_without_prompt_or_pruning_leakage(tmp_path: Path) -> None:
    """Verify shadow mode records calibration while disabling prompt guidance and pruning."""
    calib_path = tmp_path / "cost_model_calibration.jsonl"
    advisor = create_cost_advisor(
        "shadow",
        operator_name="rms_norm",
        calibration_path=calib_path,
    )
    assert isinstance(advisor, PluginCostModelAdvisor)
    assert advisor.mode == CostModelMode.SHADOW
    assert isinstance(advisor.underlying_cost_model, HybridAscendCostModel)

    decision = advisor.evaluate_round_policy(
        enable_prescreen_flag=True,
        round_idx=1,
        stagnation_rounds=0,
        incumbent_roofline_eta=0.2,
        target_roofline_eta=0.85,
        prev_local_rel_gain=None,
        mean_recoverable_excess_us=15.0,
    )
    assert decision.mode == CostModelMode.SHADOW
    assert decision.engaged is False
    assert decision.reason == "shadow_observation_only"
    assert decision.allow_prompt_guidance is False
    assert decision.allow_candidate_pruning is False

    wl = {"vec_flops": 2e9, "gm_bytes": 1e7, "total_tiles": 28}
    pred = advisor.predict_candidate_us(wl)
    assert pred is not None
    assert advisor.predictions_invoked == 1

    advisor.observe_surrogate_batch(
        observations=[{"workload": wl, "measured_us": 18.0}],
    )
    assert calib_path.is_file()
    assert advisor.underlying_cost_model.sample_count == 1

    forced_decision = CostGateDecision(
        mode=CostModelMode.AUTO,
        engaged=True,
        reason="forced",
        coarse_reject_ratio=2.0,
        allow_prompt_guidance=True,
        allow_candidate_pruning=True,
    )
    assert (
        advisor.build_prompt_guidance(
            decision=forced_decision,
            base_cfg={"bm": 64, "bn": 128, "bk": 64},
            dims={"M": 256, "N": 256, "K": 128},
        )
        == ""
    )
    kept, rejected = advisor.filter_stage_a_passers(
        ["cand_slow"],
        decision=forced_decision,
        parent_pred_us=10.0,
        parent_axes=None,
        parent_workspace_stages=0,
        candidate_pred_us_by_id={"cand_slow": 100.0},
        candidate_axes_by_id={},
        candidate_workspace_stages_by_id={},
        creative_ids=(),
    )
    assert kept == ["cand_slow"]
    assert rejected == []

    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=8,
        pipe_cycles={"AIV0.PIPE_V": 900, "AIV0.MTE2": 100},
        gm_roundtrip_bytes=4096,
        grid_blocks=28,
        waterlines={},
    )
    assert (
        advisor.resolve_dominant_bound(
            ledger,
            None,
            fallback_cost_pred=SimpleNamespace(dominant_bound="hbm"),
        )
        == "compute_bound"
    )


def test_auto_mode_operator_filter_multi_anchor_and_pruning(tmp_path: Path) -> None:
    """Verify operator_filter isolates ExperienceLake replay and anchors average cleanly."""
    lake = ExperienceLake(tmp_path / "lake")
    base_ledger_dict = {
        "grid_blocks": 28,
        "pipe_cycles": {"AIV0.PIPE_V": 400},
        "gm_roundtrip_bytes": 16384,
        "gm_traffic_scope": "per_tile",
        "dominant_contig_bytes": 32768,
        "steady_state_ii": 4,
    }
    lake.register_candidate(
        CandidateArtifactRecord(
            cand_sha="sha_rms",
            operator="rms_norm",
            regime_tag="r0",
            base_ledger=base_ledger_dict,
        )
    )
    lake.register_candidate(
        CandidateArtifactRecord(
            cand_sha="sha_mm",
            operator="matmul",
            regime_tag="r0",
            base_ledger=base_ledger_dict,
        )
    )
    lake.append_observation(
        CaseObservationRecord(
            run_id="run_1",
            operator="rms_norm",
            regime_tag="r0",
            cand_sha="sha_rms",
            eval_tier="l2_pass",
            case_num=0,
            measured_us=12.0,
            accuracy_passed=True,
            case_dims={"M": 1024, "N": 2048},
        )
    )
    lake.append_observation(
        CaseObservationRecord(
            run_id="run_2",
            operator="matmul",
            regime_tag="r0",
            cand_sha="sha_mm",
            eval_tier="l2_pass",
            case_num=0,
            measured_us=45.0,
            accuracy_passed=True,
            case_dims={"M": 512, "N": 512, "K": 512},
        )
    )

    advisor = create_cost_advisor(
        "auto",
        operator_name="rms_norm",
        experience_lake=lake,
    )
    assert isinstance(advisor, PluginCostModelAdvisor)
    stats = advisor.sync_experience_lake(lake)
    assert stats["observations"] == 1
    assert isinstance(advisor.underlying_cost_model, HybridAscendCostModel)
    assert advisor.underlying_cost_model.sample_count == 1

    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=16,
        pipe_cycles={"AIC.PIPE_M": 800},
        gm_roundtrip_bytes=32768,
        grid_blocks=28,
        waterlines={},
    )
    anchors = [
        _AnchorStub(case_num=0, regime_tag="r0", case_dims={"M": 256, "N": 256, "K": 128}, t_hw_us=5.0),
        _AnchorStub(case_num=1, regime_tag="r0", case_dims={"M": 1024, "N": 1024, "K": 256}, t_hw_us=25.0),
    ]
    mean_us = advisor.predict_anchor_mean_us(ledger, anchors=anchors)
    assert mean_us > 0.0
    assert advisor.predictions_invoked == 2

    decision = advisor.evaluate_round_policy(
        enable_prescreen_flag=True,
        round_idx=1,
        stagnation_rounds=0,
        incumbent_roofline_eta=0.2,
        target_roofline_eta=0.85,
        prev_local_rel_gain=None,
        mean_recoverable_excess_us=20.0,
        coarse_reject_ratio=3.0,
    )
    assert decision.engaged is True
    guidance = advisor.build_prompt_guidance(
        decision=decision,
        base_cfg={"bm": 64, "bn": 128, "bk": 64},
        dims={"M": 256, "N": 256, "K": 128},
        regime_tag="r0",
        top_k=3,
    )
    assert "Tile Prescreen Summary" in guidance

    axes_a = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
    axes_b = TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd")
    kept, rejected = advisor.filter_stage_a_passers(
        ["fast_exploit", "slow_exploit", "slow_creative", "slow_shift"],
        decision=decision,
        parent_pred_us=10.0,
        parent_axes=axes_a,
        parent_workspace_stages=0,
        candidate_pred_us_by_id={
            "fast_exploit": 15.0,
            "slow_exploit": 45.0,
            "slow_creative": 60.0,
            "slow_shift": 60.0,
        },
        candidate_axes_by_id={"slow_shift": axes_b},
        candidate_workspace_stages_by_id={},
        creative_ids={"slow_creative"},
    )
    assert kept == ["fast_exploit", "slow_creative", "slow_shift"]
    assert len(rejected) == 1
    assert rejected[0][0] == "slow_exploit"


def test_zero_cost_lattice_sweep_without_cost_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify sweep_executable_lattice_zero_llm works cleanly when cost_model=None."""
    def _forbid(*_a: object, **_k: object) -> object:
        raise AssertionError("HybridAscendCostModel must not be called when cost_model=None")

    monkeypatch.setattr(HybridAscendCostModel, "__init__", _forbid)
    monkeypatch.setattr(HybridAscendCostModel, "predict", _forbid)

    results = sweep_executable_lattice_zero_llm(
        _SYNTHETIC_CUBE_DSL,
        selector_fn_name="_select_cube_config",
        builder_fn_name="_build_cube_prim_func",
        sample_builder_kwargs={"m_dim": 256, "n_dim": 256, "k_dim": 128},
        dims={"M": 256, "N": 256, "K": 128},
        cost_model=None,
        target=AscendTarget("ascend.950pr"),
        max_points=12,
        top_k=3,
    )
    assert len(results) >= 1
    assert len(results) <= 3
    for item in results:
        assert isinstance(item, LatticeSweepCandidate)
        assert item.pred_us == 0.0
        assert item.l1_bytes > 0
        assert item.dominant_bound in ("compute_bound", "memory_bound", "balanced")
