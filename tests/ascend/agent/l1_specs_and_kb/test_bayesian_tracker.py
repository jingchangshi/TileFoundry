"""Unit tests for BayesianRuleTracker posterior updates, axis pruning, decay, and persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from tilefoundry_ascend.agent.kb import (
    AscendCausalRule,
    BayesianRuleTracker,
    RulePosterior,
    SymptomFamily,
    append_kb_event_shard,
    build_predicate_context,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes


def _make_rule(
    rule_id: str,
    *,
    scope: str = "cross_op",
    origin_operator: str = "",
    priority: int = 100,
    alpha_prior: float = 2.0,
    beta_prior: float = 1.0,
    symptom_family: str = SymptomFamily.COMPUTE_1C2V_SUBCORE_IMBALANCE.value,
    compatible_backends: tuple[str, ...] = ("tilelang_dsl", "ascendc_trackb", "hybrid_trackb"),
    cpp20_transform_template: str = "",
) -> AscendCausalRule:
    return AscendCausalRule(
        rule_id=rule_id,
        regime_guard="is_legal",
        precondition="PIPE_V > 0",
        transform_template=f"Apply {rule_id}.",
        expected_postcondition="PIPE_V < prev.PIPE_V",
        source_provenance="test",
        alpha_prior=alpha_prior,
        beta_prior=beta_prior,
        priority=priority,
        scope=scope,
        origin_operator=origin_operator,
        compatible_backends=compatible_backends,
        cpp20_transform_template=cpp20_transform_template,
        symptom_family=symptom_family,
    )


def test_rule_posterior_properties_and_json_roundtrip() -> None:
    """Verify RulePosterior Beta expectations, mean speedup, and JsonRecordMixin round-trip."""
    rp = RulePosterior(
        "R_TEST", alpha=4.0, beta=2.0, successes=2, failures=1, cumulative_speedup=0.5
    )
    assert rp.trials == 3
    assert rp.expected_rate == pytest.approx(4.0 / 6.0)
    assert rp.mean_speedup == pytest.approx(0.25)
    assert RulePosterior.from_json(rp.to_json()) == rp

    zero_rp = RulePosterior("R_ZERO", alpha=0.0, beta=0.0)
    assert zero_rp.expected_rate == pytest.approx(0.0)
    assert zero_rp.mean_speedup == pytest.approx(0.0)


def test_posterior_updates_axis_isolation_and_admit_preservation() -> None:
    """Verify global and 3-axis posterior updates, error guards, and rule re-admission."""
    rule = _make_rule("R_A", alpha_prior=3.0, beta_prior=1.0)
    tracker = BayesianRuleTracker.from_rules([rule])

    vec_axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
    cube_axes = TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd")

    st = tracker.record_outcome(
        "R_A", postcondition_realized=True, speed_gain=0.40, topology_axes=vec_axes
    )
    assert st.alpha == pytest.approx(4.0)
    assert st.successes == 1
    assert st.mean_speedup == pytest.approx(0.40)
    assert tracker.axis_posteriors["ce:aiv_only"]["R_A"].successes == 1
    assert tracker.axis_posteriors["mr:gm_ub_stream"]["R_A"].successes == 1
    assert tracker.axis_posteriors["ir:affine_simd"]["R_A"].successes == 1

    tracker.record_outcome("R_A", postcondition_realized=False, topology_axes=cube_axes)
    assert tracker.posteriors["R_A"].failures == 1
    assert tracker.axis_posteriors["ce:coupled_1c2v"]["R_A"].failures == 1
    assert tracker.axis_posteriors["ir:affine_simd"]["R_A"].trials == 2

    tracker.admit_rule(_make_rule("R_A", priority=50))
    assert tracker.rules["R_A"].priority == 50
    assert tracker.posteriors["R_A"].trials == 2

    with pytest.raises(KeyError):
        tracker.record_outcome("R_MISSING", postcondition_realized=True)
    with pytest.raises(ValueError, match="Invalid topology_axes"):
        tracker.record_outcome(
            "R_A",
            postcondition_realized=True,
            topology_axes={
                "compute_engine": "",
                "memory_ring": "gm_ub_stream",
                "iteration_regularity": "affine_simd",
            },
        )


def test_bucket_pruning_unprune_decay_revive_and_local_promotion() -> None:
    """Verify global/axis-scoped pruning, unprune, Beta decay/revive, and local->cross_op."""
    r_bad = _make_rule("R_BAD", alpha_prior=2.0, beta_prior=1.0)
    r_good = _make_rule("R_GOOD", alpha_prior=2.0, beta_prior=1.0)
    r_local = _make_rule("R_LOCAL", scope="operator_local", origin_operator="rms_norm")
    tracker = BayesianRuleTracker.from_rules([r_bad, r_good, r_local])

    vec_axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
    cube_axes = TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd")

    for _ in range(10):
        tracker.record_outcome("R_BAD", postcondition_realized=False, topology_axes=vec_axes)
    tracker.record_outcome(
        "R_BAD", postcondition_realized=True, speed_gain=0.30, topology_axes=cube_axes
    )

    pruned_vec = tracker.prune_underperforming(
        min_trials=5, prune_threshold=0.25, topology_axes=vec_axes, round_idx=2
    )
    assert pruned_vec == ("R_BAD",)
    assert tracker.is_pruned("R_BAD", topology_axes=vec_axes) is True
    assert tracker.is_pruned("R_BAD", topology_axes=cube_axes) is False
    assert tracker.is_pruned("R_BAD") is False

    tracker.unprune("R_BAD", topology_axes=vec_axes)
    assert tracker.is_pruned("R_BAD", topology_axes=vec_axes) is False

    for _ in range(12):
        tracker.record_outcome("R_BAD", postcondition_realized=False)
    for _ in range(3):
        tracker.record_outcome("R_GOOD", postcondition_realized=True, speed_gain=0.25)

    pruned_global = tracker.prune_underperforming(min_trials=5, prune_threshold=0.25, round_idx=3)
    assert "R_BAD" in pruned_global
    assert tracker.is_pruned("R_BAD") is True
    assert tracker.is_pruned("R_GOOD") is False

    alpha_before = tracker.posteriors["R_GOOD"].alpha
    revived_early = tracker.decay_and_revive(round_idx=8, min_cooldown_rounds=10, decay_factor=0.80)
    assert revived_early == ()
    assert tracker.posteriors["R_GOOD"].alpha < alpha_before
    assert tracker.is_pruned("R_BAD") is True

    revived = tracker.decay_and_revive(round_idx=13, min_cooldown_rounds=10, decay_factor=0.90)
    assert revived == ("R_BAD",)
    assert tracker.is_pruned("R_BAD") is False
    assert tracker.posteriors["R_BAD"].alpha == pytest.approx(3.0)
    assert tracker.posteriors["R_BAD"].beta == pytest.approx(2.0)

    assert tracker.promote_verified_local_rules(min_successes=2, min_rate=0.6) == ()
    tracker.record_outcome("R_LOCAL", postcondition_realized=True, speed_gain=0.2)
    tracker.record_outcome("R_LOCAL", postcondition_realized=True, speed_gain=0.2)
    assert tracker.promote_verified_local_rules(min_successes=2, min_rate=0.6) == ("R_LOCAL",)
    assert tracker.rules["R_LOCAL"].scope == "cross_op"


def test_active_rules_ranking_symptom_boost_and_persistence_roundtrip(tmp_path: Path) -> None:
    """Verify visibility filtering, symptom boosting, JSON persistence, and shard sync."""
    r_vec = _make_rule(
        "R_VEC",
        priority=10,
        symptom_family=SymptomFamily.COMPUTE_VECTOR_LOOP_DEGRADED.value,
        cpp20_transform_template="Use C++20 SIMD.",
    )
    r_dma = _make_rule(
        "R_DMA",
        priority=5,
        symptom_family=SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP.value,
    )
    r_local = _make_rule("R_OP_ONLY", scope="operator_local", origin_operator="mla")
    tracker = BayesianRuleTracker.from_rules([r_vec, r_dma, r_local])

    assert [r.rule_id for r in tracker.active_rules(operator_name="rms_norm")] == [
        "R_DMA",
        "R_VEC",
    ]
    assert [r.rule_id for r in tracker.active_rules(operator_name="mla")] == [
        "R_DMA",
        "R_OP_ONLY",
        "R_VEC",
    ]

    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 80},
        waterlines={},
    )
    ctx = build_predicate_context(ledger)
    ranked = tracker.rank_matching_rules(
        ctx,
        operator_name="rms_norm",
        runtime_backend="ascendc_trackb",
        symptom_ranking=[(SymptomFamily.COMPUTE_VECTOR_LOOP_DEGRADED, 0.80)],
        top_k=1,
    )
    assert len(ranked) == 1
    assert ranked[0].rule_id == "R_VEC"
    assert ranked[0].transform_template == "Use C++20 SIMD."

    axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
    tracker.record_outcome("R_DMA", postcondition_realized=True, speed_gain=0.5, topology_axes=axes)
    lake_dir = tmp_path / "lake"
    append_kb_event_shard(
        lake_dir,
        run_id="r_own",
        rule_id="R_VEC",
        postcondition_realized=True,
        speed_gain=0.25,
        topology_axes=axes,
        round_idx=1,
    )
    append_kb_event_shard(
        lake_dir,
        run_id="r_peer",
        rule_id="R_VEC",
        postcondition_realized=True,
        speed_gain=0.35,
        topology_axes=axes,
        round_idx=1,
    )
    ingested = tracker.sync_from_experience_lake(lake_dir, exclude_run_id="r_own")
    assert ingested == 1
    assert tracker.posteriors["R_VEC"].successes == 1

    save_path = tmp_path / "tracker_state.json"
    tracker.save_json(save_path)
    restored = BayesianRuleTracker.from_dict(
        tracker.to_dict(), rules=[r_vec, r_dma, r_local]
    )
    assert restored.posteriors["R_DMA"].mean_speedup == pytest.approx(0.5)
    assert restored.sync_from_experience_lake(lake_dir, exclude_run_id="r_own") == 0

    disk_restored = BayesianRuleTracker.from_rules([r_vec, r_dma, r_local])
    disk_restored.load_posteriors_json(save_path)
    assert disk_restored.posteriors["R_VEC"].successes == 1
    assert disk_restored.axis_posteriors["ce:aiv_only"]["R_DMA"].successes == 1
