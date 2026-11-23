"""Unit tests for propagating real CasePhysicsRatios into portfolio prompts and decisions."""

from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.orchestrator import compute_static_dynamic_residual
from tilefoundry_ascend.agent.orchestrator.gate_funnel import GateFunnel
from tilefoundry_ascend.agent.orchestrator.llm_client import MockLLMBackend
from tilefoundry_ascend.agent.orchestrator.llm_constraints import (
    format_hardware_and_sil_layer,
)
from tilefoundry_ascend.agent.orchestrator.portfolio_runner import (
    PortfolioRunner,
    _derive_active_topology_axes,
    _format_regime_ratio_line,
    _project_features_with_ratios,
    format_sil_diagnostics_section,
)
from tilefoundry_ascend.agent.orchestrator.residual_beam import BeamCandidate
from tilefoundry_ascend.agent.router.capability_gate import EscalationDecision
from tilefoundry_ascend.ascend.passes import (
    build_objective_physical_ledger,
    build_scope_waterline,
)
from tilefoundry_ascend.ascend.sil.schema import (
    AdvisoryEnvelope,
    CasePhysicsRatios,
    RegimeSlice,
    TopologyAxes,
)

_RATIO_CASE_1 = CasePhysicsRatios(
    case_id=1,
    stride_ratio=1.0,
    contig_atom_ratio=1.0,
    is_atom_aligned=True,
    working_set_sram_ratio=0.5,
    working_set_vreg_ratio=0.4,
    wave_occupancy_ratio=1.0,
    oi_ridge_ratio=1.5,
    inner_contiguous_bytes=64,
)

_RATIO_CASE_2 = CasePhysicsRatios(
    case_id=2,
    stride_ratio=1.0,
    contig_atom_ratio=0.5,
    is_atom_aligned=False,
    working_set_sram_ratio=0.4,
    working_set_vreg_ratio=0.2,
    wave_occupancy_ratio=0.25,
    oi_ridge_ratio=0.1,
    inner_contiguous_bytes=256,
)


def _advisory_envelope() -> AdvisoryEnvelope:
    return AdvisoryEnvelope(
        op_name="softmax",
        cannbot_task="level2/softmax",
        level=2,
        topology_axes=TopologyAxes(
            compute_engine="aic_only",
            memory_ring="gm_l1_l0_fixpipe",
            iteration_regularity="affine_simd",
        ),
        input_dtypes=("float16",),
        output_dtypes=("float16",),
        roofline={"ideal_roofline_us": 1.7e-05},
        structural_features={},
        regimes=(
            RegimeSlice(
                regime_tag="aligned_tail",
                case_nums=(1, 2),
                representative_dims={"m": 128},
                has_unaligned_tail=True,
                case_ratios=(_RATIO_CASE_1, _RATIO_CASE_2),
                representative_case_id=2,
            ),
            RegimeSlice(
                regime_tag="unknown_shape",
                case_nums=(3,),
                representative_dims={"m": 64},
                has_unaligned_tail=False,
            ),
        ),
    )


def test_hardware_layer_renders_representative_real_ratios() -> None:
    """Every regime's representative real ratios reach the shared 3-layer prompt text."""
    text = format_hardware_and_sil_layer(sil_advisory=_advisory_envelope())
    lines = text.splitlines()
    aligned = [line for line in lines if "Regime aligned_tail" in line]
    assert len(aligned) == 1
    line = aligned[0]
    assert "- Regime aligned_tail cases=[1, 2] representative_case=2 " in line
    payload = json.loads(line.split("CasePhysicsRatios: ", 1)[1])
    assert payload == _RATIO_CASE_2.to_dict()
    assert json.dumps(_RATIO_CASE_1.to_dict(), sort_keys=True) not in text
    assert "- Regime unknown_shape cases=[3] CasePhysicsRatios: unavailable" in text


def test_sil_diagnostics_section_renders_serialized_real_ratios() -> None:
    """The syntax-repair diagnostics render representative ratios from serialized envelopes."""
    text = format_sil_diagnostics_section(sil_advisory=_advisory_envelope().to_dict())
    lines = [line for line in text.splitlines() if line.startswith("- regime aligned_tail")]
    assert len(lines) == 1
    line = lines[0]
    assert "- regime aligned_tail cases=[1, 2] CasePhysicsRatios: " in line
    payload = json.loads(line.split("CasePhysicsRatios: ", 1)[1])
    assert payload == _RATIO_CASE_2.to_dict()
    assert "- regime unknown_shape cases=[3] CasePhysicsRatios: unavailable" in text


def test_regime_ratio_line_falls_back_to_median_without_representative() -> None:
    """A serialized regime without representative_case_id renders its median case ratios."""
    regime = RegimeSlice(
        regime_tag="median_only",
        case_nums=(1, 2),
        representative_dims={"m": 128},
        has_unaligned_tail=False,
        case_ratios=(_RATIO_CASE_1, _RATIO_CASE_2),
    ).to_dict()
    line = _format_regime_ratio_line(regime)
    payload = json.loads(line.split("CasePhysicsRatios: ", 1)[1])
    assert payload == _RATIO_CASE_2.to_dict()


def test_regime_ratio_line_reports_unavailable_when_representative_ratio_is_missing() -> None:
    """A missing representative ratio matches RegimeSlice.representative_ratios() semantics."""
    regime = RegimeSlice(
        regime_tag="missing_representative",
        case_nums=(1, 2),
        representative_dims={"m": 128},
        has_unaligned_tail=False,
        case_ratios=(_RATIO_CASE_1,),
        representative_case_id=2,
    ).to_dict()
    line = _format_regime_ratio_line(regime)
    assert line == (
        "- regime missing_representative cases=[1, 2] CasePhysicsRatios: unavailable"
    )


def _semantic_features() -> dict[str, object]:
    return {
        "static_affine_bounds": True,
        "cube_to_vector_ratio": 1.25,
        "quant_granularity": "per_tensor",
        "indirect_sparse_gather": False,
        "cross_iter_recurrence": "none",
        "dynamic_cross_core_handoff": False,
        "inner_contiguous_bytes": 64,
        "persistent_wave_occupancy": 1.0,
    }


def test_project_features_with_ratios_updates_contig_bytes_and_occupancy() -> None:
    """Projected semantic features carry real contiguous bytes and wave occupancy."""
    features = _semantic_features()
    projected = _project_features_with_ratios(features, _RATIO_CASE_2)
    assert projected is not None
    assert projected["inner_contiguous_bytes"] == 256
    assert projected["persistent_wave_occupancy"] == 0.25
    assert projected["cube_to_vector_ratio"] == features["cube_to_vector_ratio"]
    assert projected["quant_granularity"] == features["quant_granularity"]
    assert features["inner_contiguous_bytes"] == 64
    assert _project_features_with_ratios(None, _RATIO_CASE_2) is None
    assert _project_features_with_ratios(features, None) is features


def test_derive_active_topology_axes_applies_case_ratio_phase_transition() -> None:
    """Real ratios shift GEMV-like cube cases onto the AIV/stream/irregular axes."""
    base_axes = TopologyAxes(
        compute_engine="aic_only",
        memory_ring="gm_l1_l0_fixpipe",
        iteration_regularity="affine_simd",
    )
    assert _derive_active_topology_axes(None, topology_axes=base_axes) == base_axes
    axes = _derive_active_topology_axes(
        None,
        topology_axes=base_axes,
        case_ratios=_RATIO_CASE_2,
    )
    assert axes == TopologyAxes(
        compute_engine="aiv_only",
        memory_ring="gm_ub_stream",
        iteration_regularity="irregular_simt",
    )


def test_check_round_escalation_forwards_projected_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Escalation receives the ratio-projected semantic features."""
    captured: dict[str, object] = {}

    def _spy(
        *,
        current_tier: str,
        roofline_eta: float,
        target_eta: float,
        stagnation_rounds: int,
        dominant_bound: str = "",
        compiler_summary: object | None = None,
        structural_features: Mapping[str, object] | None = None,
    ) -> EscalationDecision:
        captured["current_tier"] = current_tier
        captured["roofline_eta"] = roofline_eta
        captured["target_eta"] = target_eta
        captured["stagnation_rounds"] = stagnation_rounds
        captured["dominant_bound"] = dominant_bound
        captured["compiler_summary"] = compiler_summary
        captured["structural_features"] = structural_features
        return EscalationDecision(
            should_escalate=False,
            from_tier="tilelang_dsl",
            to_tier="tilelang_dsl",
            trigger_reasons=("spy",),
        )

    monkeypatch.setattr(
        "tilefoundry_ascend.agent.orchestrator.portfolio_runner.evaluate_track_escalation",
        _spy,
    )

    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 100},
        waterlines={
            "UB": build_scope_waterline(
                scope="UB",
                capacity_bytes=262144,
                explicit_buffers={"buf": 65536},
                implicit_buffers={},
            )
        },
    )
    residual = compute_static_dynamic_residual(ledger, {"dur_us": 20.0})
    candidate = BeamCandidate(
        candidate_id="r1_rule_0",
        round_idx=1,
        worker_kind="rule_exploit",
        rule_id=None,
        dsl_source="def make_kernel():\n    pass\n",
        track="unconstrained",
        ledger=ledger,
        residual=residual,
        latency_us=20.0,
        accuracy_passed=True,
        features=_semantic_features(),
    )
    runner = PortfolioRunner(
        backend=MockLLMBackend(),
        funnel=GateFunnel(skip_stage_b=True),
        tracker=BayesianRuleTracker(),
    )
    report = runner.check_round_escalation(
        candidate,
        target_eta=0.85,
        case_ratios=_RATIO_CASE_2,
    )
    assert report.escalation.trigger_reasons == ("spy",)
    forwarded = captured["structural_features"]
    assert isinstance(forwarded, dict)
    assert forwarded["inner_contiguous_bytes"] == 256
    assert forwarded["persistent_wave_occupancy"] == 0.25
    assert candidate.features is not None
    assert candidate.features["inner_contiguous_bytes"] == 64
