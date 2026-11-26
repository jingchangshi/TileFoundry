"""Tests for zero-magic cost advisor, physical prescreen, and closed-schema distiller."""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, fields
from pathlib import Path

import pytest

from tilefoundry_ascend.agent.cost_advisor import (
    CostModelAdvisor,
    NullCostModelAdvisor,
    PluginCostModelAdvisor,
    create_cost_advisor,
)
from tilefoundry_ascend.agent.flywheel import ablation_distiller as ad
from tilefoundry_ascend.agent.flywheel.ablation_distiller import (
    ALLOWED_SCHEMA_SIGNAL_VOCABULARY,
    FlywheelDistiller,
    MicroProbeSpec,
    _validate_distilled_rule,
    _validate_schema_closed_identifiers,
    is_paired_gain_significant,
    resolve_paired_sigma_us,
    should_trigger_flywheel_checkpoint,
    verify_micro_probe_hardware,
)
from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.kb.schema import AscendCausalRule, SymptomFamily
from tilefoundry_ascend.agent.orchestrator import gate_funnel as gf
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    GateFunnel,
    StageAResult,
    StageBResult,
    StageCResult,
)
from tilefoundry_ascend.agent.orchestrator.tilesize_prescreen import (
    TileConfigCandidate,
    evaluate_tile_candidate,
    prescreen_live_ir_neighborhood,
    prescreen_tile_configs,
    sweep_executable_lattice_zero_llm,
)
from tilefoundry_ascend.agent.orchestrator.residual_beam import compute_static_dynamic_residual
from tilefoundry_ascend.ascend.cost_model.cost_gate_policy import (
    CostGateDecision,
    CostModelMode,
    evaluate_cost_gate_policy,
)
from tilefoundry_ascend.ascend.cost_model.hybrid_cost_model import HybridAscendCostModel
from tilefoundry_ascend.ascend.passes.observation_space import (
    ComputeIssueAxis,
    ObservationVector5D,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    ObjectivePhysicalLedger,
    build_objective_physical_ledger,
)
from tilefoundry_ascend.ascend.sil.schema import CasePhysicsRatios, TopologyAxes
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget


@dataclass(frozen=True)
class _AnchorStub:
    case_num: int
    regime_tag: str
    case_dims: dict[str, int]
    t_hw_us: float


_CUBE_DSL = """
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


def test_cost_gate_policy_modes_and_student_t_threshold() -> None:
    """Verify CostModelMode enum, t-stat gating at N=8, and permanent no-pruning invariant."""
    assert {m.value for m in CostModelMode} == {"off", "shadow", "advisory"}
    decision_fields = {f.name for f in fields(CostGateDecision)}
    assert decision_fields == {
        "mode",
        "engaged",
        "reason",
        "allow_prompt_guidance",
        "allow_candidate_pruning",
    }

    off_dec = evaluate_cost_gate_policy(
        mode=CostModelMode.OFF,
        bucket_clean_samples=32,
        out_of_sample_spearman_rho=0.95,
    )
    assert off_dec.mode == CostModelMode.OFF
    assert off_dec.engaged is False
    assert off_dec.reason == "disabled_by_config"
    assert off_dec.allow_prompt_guidance is False
    assert off_dec.allow_candidate_pruning is False

    shadow_dec = evaluate_cost_gate_policy(
        mode=CostModelMode.SHADOW,
        bucket_clean_samples=32,
        out_of_sample_spearman_rho=0.95,
    )
    assert shadow_dec.mode == CostModelMode.SHADOW
    assert shadow_dec.engaged is False
    assert shadow_dec.reason == "shadow_observation_only"
    assert shadow_dec.allow_prompt_guidance is False
    assert shadow_dec.allow_candidate_pruning is False

    too_few = evaluate_cost_gate_policy(
        mode=CostModelMode.ADVISORY,
        bucket_clean_samples=7,
        out_of_sample_spearman_rho=0.95,
    )
    assert too_few.engaged is False
    assert too_few.allow_prompt_guidance is False
    assert too_few.allow_candidate_pruning is False

    missing_rho = evaluate_cost_gate_policy(
        mode=CostModelMode.ADVISORY,
        bucket_clean_samples=8,
        out_of_sample_spearman_rho=None,
    )
    assert missing_rho.engaged is False
    assert missing_rho.allow_candidate_pruning is False

    below_t = evaluate_cost_gate_policy(
        mode=CostModelMode.ADVISORY,
        bucket_clean_samples=8,
        out_of_sample_spearman_rho=0.60,
    )
    assert below_t.engaged is False
    assert below_t.allow_prompt_guidance is False
    assert below_t.allow_candidate_pruning is False

    above_t = evaluate_cost_gate_policy(
        mode=CostModelMode.ADVISORY,
        bucket_clean_samples=8,
        out_of_sample_spearman_rho=0.65,
    )
    assert above_t.engaged is True
    assert above_t.allow_prompt_guidance is True
    assert above_t.allow_candidate_pruning is False

    perfect_rho = evaluate_cost_gate_policy(
        mode=CostModelMode.ADVISORY,
        bucket_clean_samples=8,
        out_of_sample_spearman_rho=1.0,
    )
    assert perfect_rho.engaged is True
    assert perfect_rho.allow_prompt_guidance is True
    assert perfect_rho.allow_candidate_pruning is False

    flag_off = evaluate_cost_gate_policy(
        mode=CostModelMode.ADVISORY,
        enable_prescreen_flag=False,
        bucket_clean_samples=16,
        out_of_sample_spearman_rho=0.95,
    )
    assert flag_off.mode == CostModelMode.OFF
    assert flag_off.engaged is False
    assert flag_off.allow_candidate_pruning is False


def test_cost_advisor_off_zero_predictions_and_no_pruning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify zero predictions in OFF mode and zero pruning of creative/cross-topology."""
    def _forbid(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("HybridAscendCostModel must not run in OFF mode")

    monkeypatch.setattr(HybridAscendCostModel, "__init__", _forbid)
    monkeypatch.setattr(HybridAscendCostModel, "predict", _forbid)

    advisor = create_cost_advisor("off", calibration_path=tmp_path / "calib.jsonl")
    assert isinstance(advisor, NullCostModelAdvisor)
    assert isinstance(advisor, CostModelAdvisor)
    assert advisor.predictions_invoked == 0
    assert advisor.underlying_cost_model is None
    assert advisor.predict_candidate_us({"vec_flops": 1e9}) is None
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=8,
        pipe_cycles={"AIV0.PIPE_V": 200},
        gm_roundtrip_bytes=4096,
        grid_blocks=28,
        waterlines={},
    )
    assert advisor.predict_anchor_mean_us(ledger, anchors=[]) == 0.0
    assert advisor.predictions_invoked == 0

    dec = advisor.evaluate_round_policy(
        enable_prescreen_flag=True,
        bucket_clean_samples=20,
        out_of_sample_spearman_rho=0.9,
    )
    assert dec.engaged is False
    assert dec.allow_candidate_pruning is False
    assert (
        advisor.build_prompt_guidance(
            decision=dec,
            base_cfg={"bm": 64, "bn": 128, "bk": 64},
            dims={"M": 256, "N": 256, "K": 128},
        )
        == ""
    )

    monkeypatch.undo()
    adv_advisor = create_cost_advisor("advisory", operator_name="custom_op")
    assert isinstance(adv_advisor, PluginCostModelAdvisor)
    assert adv_advisor.mode == CostModelMode.ADVISORY
    engaged_dec = adv_advisor.evaluate_round_policy(
        enable_prescreen_flag=True,
        bucket_clean_samples=8,
        out_of_sample_spearman_rho=0.70,
    )
    assert engaged_dec.engaged is True
    assert engaged_dec.allow_prompt_guidance is True
    assert engaged_dec.allow_candidate_pruning is False

    axes_a = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
    axes_b = TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd")
    kept, rejected = adv_advisor.filter_stage_a_passers(
        ["exploit_slow", "explore_creative", "cross_topo"],
        decision=engaged_dec,
        parent_pred_us=10.0,
        parent_axes=axes_a,
        parent_workspace_stages=0,
        candidate_pred_us_by_id={
            "exploit_slow": 500.0,
            "explore_creative": 900.0,
            "cross_topo": 800.0,
        },
        candidate_axes_by_id={"cross_topo": axes_b},
        candidate_workspace_stages_by_id={},
        creative_ids={"explore_creative"},
    )
    assert kept == ["exploit_slow", "explore_creative", "cross_topo"]
    assert rejected == []

    shadow_advisor = create_cost_advisor("shadow", operator_name="custom_op")
    shadow_dec = shadow_advisor.evaluate_round_policy(
        enable_prescreen_flag=True,
        bucket_clean_samples=12,
        out_of_sample_spearman_rho=0.95,
    )
    assert shadow_dec.engaged is False
    assert (
        shadow_advisor.build_prompt_guidance(
            decision=engaged_dec,
            base_cfg={"bm": 64, "bn": 128, "bk": 64},
            dims={"M": 256, "N": 256, "K": 128},
        )
        == ""
    )


def test_tilesize_prescreen_zero_magic_hard_physics_in_off_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify zero-parameter hard feasibility checks and Roofline/tail-waste ranking in off mode."""
    facts = AscendTarget("ascend.950pr").get_facts(AscendSpaceFacts)

    zero_denom = evaluate_tile_candidate(TileConfigCandidate(bm=0, bn=64, bk=64), facts)
    assert zero_denom.is_feasible is False
    assert zero_denom.rejection_reason == "non_positive_tile_dimension"

    unaligned = evaluate_tile_candidate(TileConfigCandidate(bm=48, bn=64, bk=64), facts)
    assert unaligned.is_feasible is True
    bad_align = evaluate_tile_candidate(TileConfigCandidate(bm=15, bn=64, bk=64), facts)
    assert bad_align.is_feasible is False
    assert bad_align.rejection_reason == "unaligned_mmad_16x16x16_atom"

    stack_bad = evaluate_tile_candidate(
        TileConfigCandidate(bm=64, bn=64, bk=64, stack_bytes=128), facts
    )
    assert stack_bad.is_feasible is False
    assert stack_bad.rejection_reason is not None
    assert "stack_overflow" in stack_bad.rejection_reason

    cap_bad = evaluate_tile_candidate(TileConfigCandidate(bm=256, bn=512, bk=128), facts)
    assert cap_bad.is_feasible is False

    no_split = evaluate_tile_candidate(
        TileConfigCandidate(bm=128, bn=128, bk=64, stages=2, l1_half_bank_split=False),
        facts,
    )
    assert no_split.half_l1_penalty == 0.0

    exact_fit = TileConfigCandidate(bm=64, bn=64, bk=64, stages=2)
    tail_waste = TileConfigCandidate(bm=112, bn=112, bk=64, stages=2)
    ranked = prescreen_tile_configs(
        [tail_waste, exact_fit, bad_align.candidate, cap_bad.candidate],
        target=AscendTarget("ascend.950pr"),
        m_dim=256,
        n_dim=256,
        k_dim=128,
        cost_model_mode="off",
    )
    assert len(ranked) == 2
    assert ranked[0].candidate == exact_fit
    assert ranked[0].tail_waste_ratio == pytest.approx(0.0)
    assert ranked[1].tail_waste_ratio > 0.0
    assert ranked[0].t_roof_us <= ranked[1].t_roof_us

    def _forbid_predict(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("CostModel.predict must not run when cost_model_mode == 'off'")

    model = HybridAscendCostModel(AscendTarget("ascend.950pr"))
    monkeypatch.setattr(model, "predict", _forbid_predict)

    def _fake_stage_a(*_args: object, **kwargs: object) -> StageAResult:
        fkw = kwargs.get("factory_kwargs")
        bm = int(fkw.get("bm", 64)) if isinstance(fkw, dict) else 64
        bn = int(fkw.get("bn", 64)) if isinstance(fkw, dict) else 64
        bk = int(fkw.get("bk", 32)) if isinstance(fkw, dict) else 32
        ledger = build_objective_physical_ledger(
            analysis_mode="readonly_envelope",
            is_legal=True,
            steady_state_ii=16,
            pipe_cycles={"AIC.PIPE_M": max(1, (2 * bm * bn * bk) // 4096)},
            gm_roundtrip_bytes=bm * bk * 2 + bk * bn * 2,
            grid_blocks=28,
            waterlines={},
        )
        return StageAResult(passed=True, prim_func=None, ledger=ledger)

    monkeypatch.setattr(gf, "run_stage_a_static", _fake_stage_a)

    sweep_dsl = """
def _select_cube_config(m_dim: int = 256, n_dim: int = 256, k_dim: int = 128) -> dict[str, int]:
    return {"bm": 64, "bn": 64, "bk": 32, "num_stages": 1}

def _build_cube_prim_func(bm: int = 64, bn: int = 64, bk: int = 32, num_stages: int = 1, **_extra):
    return None
"""
    sweep_results = sweep_executable_lattice_zero_llm(
        sweep_dsl,
        selector_fn_name="_select_cube_config",
        builder_fn_name="_build_cube_prim_func",
        sample_builder_kwargs={"m_dim": 256, "n_dim": 256, "k_dim": 128},
        dims={"M": 256, "N": 256, "K": 128},
        cost_model=model,
        cost_model_mode="off",
        max_points=12,
        top_k=3,
    )
    assert len(sweep_results) >= 1
    assert all(item.pred_us == 0.0 for item in sweep_results)
    assert sweep_results == sorted(
        sweep_results,
        key=lambda item: (round(item.t_roof_us, 12), round(item.tail_waste_ratio, 12)),
    )

    summary = prescreen_live_ir_neighborhood(
        sweep_dsl,
        factory_name="_build_cube_prim_func",
        center=TileConfigCandidate(bm=64, bn=64, bk=32, stages=1),
        base_kwargs={"m_dim": 256, "n_dim": 256, "k_dim": 128},
        dims={"M": 256, "N": 256, "K": 128},
        max_variants=3,
        cost_model=model,
        cost_model_mode="off",
    )
    assert "Live-IR Tile Prescreen Summary" in summary
    assert "pred_us=" not in summary


def test_ablation_distiller_closed_vocabulary_and_dynamic_sanitization() -> None:
    """Verify schema-closed AST identifier gate and dynamic operator-name sanitization."""
    src_text = inspect.getsource(ad)
    assert "_STATIC_DISALLOWED_PROVENANCE_TOKENS" not in src_text
    assert "_SUBSTRING_PROVENANCE_TOKENS" not in src_text
    for forbidden_op in (
        "attention_residual",
        "attn_res",
        "mla_model",
        "grouped_matmul",
        "deepgemm",
        "flash_attention",
    ):
        assert forbidden_op not in src_text

    for cls in (
        ObservationVector5D,
        ComputeIssueAxis,
        ObjectivePhysicalLedger,
        CasePhysicsRatios,
        TopologyAxes,
    ):
        for f in fields(cls):
            assert f.name in ALLOWED_SCHEMA_SIGNAL_VOCABULARY

    assert (
        _validate_schema_closed_identifiers(
            "measured_latency_us < prev.measured_latency_us and any_overflow_bytes == 0"
        )
        == []
    )
    assert _validate_schema_closed_identifiers("unknown_magic_signal > 0") == [
        "unknown_magic_signal"
    ]
    assert _validate_schema_closed_identifiers("measured_latency_us < prev.bogus_field") == [
        "bogus_field"
    ]

    bad_ident_rule = AscendCausalRule(
        rule_id="R_BAD_ID",
        regime_guard="is_legal",
        precondition="degraded_buffer_count > 0 and unknown_custom_knob > 1",
        transform_template="Use double buffering on on-chip SRAM buffers.",
        expected_postcondition="degraded_buffer_count < prev.degraded_buffer_count",
        source_provenance="flywheel:ablation",
        compatible_backends=("tilelang_dsl",),
        symptom_family=SymptomFamily.SYNC_PSEUDO_DOUBLE_BUFFER.value,
    )
    ok_bad_id, err_bad_id = _validate_distilled_rule(bad_ident_rule)
    assert ok_bad_id is False
    assert "unknown_custom_knob" in err_bad_id

    bad_post_id_rule = AscendCausalRule(
        rule_id="R_BAD_POST_ID",
        regime_guard="is_legal",
        precondition="degraded_buffer_count > 0",
        transform_template="Use double buffering on on-chip SRAM buffers.",
        expected_postcondition="mystery_metric < prev.mystery_metric",
        source_provenance="flywheel:ablation",
        compatible_backends=("tilelang_dsl",),
        symptom_family=SymptomFamily.SYNC_PSEUDO_DOUBLE_BUFFER.value,
    )
    ok_post_id, err_post_id = _validate_distilled_rule(bad_post_id_rule)
    assert ok_post_id is False
    assert "mystery_metric" in err_post_id

    leaky_dynamic_rule = AscendCausalRule(
        rule_id="R_DYN_LEAK",
        regime_guard="is_legal",
        precondition="degraded_buffer_count > 0",
        transform_template="Apply tile staging for CustomSyntheticKernel workload.",
        expected_postcondition="degraded_buffer_count < prev.degraded_buffer_count",
        source_provenance="flywheel:ablation",
        compatible_backends=("tilelang_dsl",),
        symptom_family=SymptomFamily.SYNC_PSEUDO_DOUBLE_BUFFER.value,
    )
    ok_dyn, err_dyn = _validate_distilled_rule(
        leaky_dynamic_rule,
        operator_name="custom_synthetic_kernel",
    )
    assert ok_dyn is False
    assert "disallowed token" in err_dyn

    path_leaky_rule = AscendCausalRule(
        rule_id="R_PATH_LEAK",
        regime_guard="is_legal",
        precondition="degraded_buffer_count > 0",
        transform_template="Load schedule from /workspace/user_a/cache/sched.json.",
        expected_postcondition="degraded_buffer_count < prev.degraded_buffer_count",
        source_provenance="flywheel:ablation",
        compatible_backends=("tilelang_dsl",),
        symptom_family=SymptomFamily.SYNC_PSEUDO_DOUBLE_BUFFER.value,
    )
    ok_path, err_path = _validate_distilled_rule(path_leaky_rule)
    assert ok_path is False
    assert "filesystem path" in err_path


def test_ablation_distiller_2sigma_and_3sigma_paired_significance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify 3*sigma checkpoint/probe gating and 2*sigma LOO attribution gating."""
    assert resolve_paired_sigma_us(100.0, sigma_paired_us=4.0) == pytest.approx(4.0)
    assert resolve_paired_sigma_us(100.0, paired_rel_mad=0.02) == pytest.approx(
        1.4826 * 0.02 * 100.0
    )

    assert should_trigger_flywheel_checkpoint(100.0, 86.0, sigma_paired_us=5.0) is False
    assert should_trigger_flywheel_checkpoint(100.0, 84.0, sigma_paired_us=5.0) is True
    assert is_paired_gain_significant(9.0, sigma_paired_us=5.0, sigma_multiplier=2.0) is False
    assert is_paired_gain_significant(11.0, sigma_paired_us=5.0, sigma_multiplier=2.0) is True

    probe = MicroProbeSpec(
        probe_id="p1",
        probe_kind="barrier_prune",
        factory_name="make_flywheel_probe",
        baseline_probe_source="x = 1",
        optimized_probe_source="x = 2",
        line_count_baseline=1,
        line_count_optimized=1,
        tensor_shape=(128,),
    )
    ok_noisy, _ = verify_micro_probe_hardware(
        probe,
        probe_runner=lambda _s: {
            "baseline_dur_us": 100.0,
            "optimized_dur_us": 88.0,
            "sigma_paired_us": 5.0,
            "numerical_ok": 1.0,
        },
    )
    assert ok_noisy is False

    ok_sig, metrics_sig = verify_micro_probe_hardware(
        probe,
        probe_runner=lambda _s: {
            "baseline_dur_us": 100.0,
            "optimized_dur_us": 80.0,
            "sigma_paired_us": 5.0,
            "numerical_ok": 1.0,
        },
    )
    assert ok_sig is True
    assert metrics_sig["delta_us"] == pytest.approx(20.0)
    assert math.isclose(metrics_sig["sigma_paired_us"], 5.0)

    base_dsl = """
import tilelang.ascend.language as T

def make_kernel():
    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), C: T.Tensor((128,), "float32")):
        with T.Kernel(1) as bx:
            a_ub = T.alloc_shared((128,), "float32")
            T.copy(A[0:128], a_ub)
            T.ascend_pipe_barrier("PIPE_ALL")
            T.copy(a_ub, C[0:128])
    return main
"""
    win_dsl = """
import tilelang.ascend.language as T

def make_kernel():
    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), C: T.Tensor((128,), "float32")):
        with T.Kernel(1) as bx:
            a_ub = T.alloc_shared((128,), "float32")
            T.copy(A[0:128], a_ub)
            T.copy(a_ub, C[0:128])
    return main
"""
    def _fake_distill_stage_a(dsl_src: str, **_kwargs: object) -> StageAResult:
        barriers = 1 if "ascend_pipe_barrier" in dsl_src else 0
        ledger = build_objective_physical_ledger(
            analysis_mode="readonly_envelope",
            is_legal=True,
            steady_state_ii=8,
            pipe_cycles={"AIV0.PIPE_V": 400},
            gm_roundtrip_bytes=4096,
            grid_blocks=28,
            in_loop_pipe_all_barriers=barriers,
            waterlines={},
        )
        return StageAResult(passed=True, prim_func=None, ledger=ledger)

    monkeypatch.setattr(gf, "run_stage_a_static", _fake_distill_stage_a)
    monkeypatch.setattr(ad, "run_stage_a_static", _fake_distill_stage_a)

    funnel = GateFunnel(
        compile_runner=lambda _s: StageBResult(passed=True),
        device_runner=lambda s, st: StageCResult(
            passed=True,
            accuracy_passed=True,
            measured_latency_us=88.0 if "ascend_pipe_barrier" in s else 80.0,
            residual=compute_static_dynamic_residual(st.ledger, {"dur_us": 80.0}),
            telemetry={"dur_us": 80.0},
        ),
    )

    tracker_noise = BayesianRuleTracker()
    distiller = FlywheelDistiller(
        funnel=funnel,
        tracker=tracker_noise,
        probe_runner=lambda _s: {
            "baseline_dur_us": 100.0,
            "optimized_dur_us": 75.0,
            "sigma_paired_us": 5.0,
            "numerical_ok": 1.0,
        },
        sigma_paired_us=5.0,
    )
    rep_sub_2sigma = distiller.distill_leave_one_out(
        baseline_dsl=base_dsl,
        winner_dsl=win_dsl,
        baseline_latency_us=100.0,
        winner_latency_us=80.0,
        factory_name="make_kernel",
        operator_name="synthetic_op",
    )
    assert rep_sub_2sigma.admitted_rule_ids == ()

    tracker_sig = BayesianRuleTracker()
    distiller_sig = FlywheelDistiller(
        funnel=funnel,
        tracker=tracker_sig,
        sigma_paired_us=3.0,
    )
    rep_above_2sigma = distiller_sig.distill_leave_one_out(
        baseline_dsl=base_dsl,
        winner_dsl=win_dsl,
        baseline_latency_us=100.0,
        winner_latency_us=80.0,
        factory_name="make_kernel",
        operator_name="synthetic_op",
    )
    assert len(rep_above_2sigma.admitted_rule_ids) == 1
    assert len(rep_above_2sigma.kept_positive_chunks) == 1

    rep_checkpoint_fail = distiller_sig.distill_from_pair(
        baseline_dsl=base_dsl,
        winner_dsl=win_dsl,
        baseline_latency_us=88.0,
        winner_latency_us=80.0,
        factory_name="make_kernel",
        operator_name="synthetic_op",
        verify_on_candidate=True,
    )
    assert rep_checkpoint_fail.admitted_rule_ids == ()
