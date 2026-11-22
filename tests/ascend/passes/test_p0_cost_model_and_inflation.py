"""Tests for ExperienceLake dedup, per-regime snapshots, and transfer inflation ratio."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    BeamCandidate,
    compute_static_dynamic_residual,
)
from tilefoundry_ascend.ascend.cost_model.experience_lake import (
    CandidateArtifactRecord,
    CaseObservationRecord,
    ExperienceLake,
    project_case_workload,
)
from tilefoundry_ascend.ascend.cost_model.hybrid_cost_model import HybridAscendCostModel
from tilefoundry_ascend.ascend.cost_model.microarch_profile import MicroarchProfile
from tilefoundry_ascend.ascend.passes.observation_space import (
    ObservationVector5D,
    compute_transfer_inflation_ratio,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    ObjectivePhysicalLedger,
    build_objective_physical_ledger,
)
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes
from tilefoundry_ascend.ascend.target import AscendTarget


def _sample_ledger(
    *,
    gm_roundtrip_bytes: int = 0,
    grid_blocks: int = 16,
    gm_traffic_scope: str = "per_tile",
) -> ObjectivePhysicalLedger:
    return build_objective_physical_ledger(
        analysis_mode="exact_modulo_z3",
        is_legal=True,
        steady_state_ii=2,
        pipe_cycles={"AIV0.PIPE_V": 200, "AIV1.PIPE_V": 200},
        waterlines={},
        grid_blocks=grid_blocks,
        gm_roundtrip_bytes=gm_roundtrip_bytes,
        gm_traffic_scope=gm_traffic_scope,
        microarch_profile=MicroarchProfile(
            fp_alu_ops=1024,
            gm_to_ub_bytes=65536,
            ub_to_gm_bytes=65536,
            pipeline_depth=2,
            inner_loop_trips=4,
            source_kind="tirx",
        ).to_dict(),
    )


def test_online_dedup_keys_and_exclude_run_id_prevent_double_counting(tmp_path: Path) -> None:
    """Online mark helpers and exclude_run_id preserve 1:1 counts on lake replay."""
    lake = ExperienceLake(tmp_path / "lake")
    axes = TopologyAxes(
        compute_engine="aiv_only",
        memory_ring="gm_ub_stream",
        iteration_regularity="affine_simd",
    )
    parent_ledger = _sample_ledger(gm_roundtrip_bytes=4096)
    child_ledger = _sample_ledger(gm_roundtrip_bytes=0)

    parent_sha = "parent_sha_001"
    child_sha = "child_sha_002"
    other_sha = "other_sha_003"
    for sha, parent, ledger in (
        (parent_sha, "", parent_ledger),
        (child_sha, parent_sha, child_ledger),
        (other_sha, parent_sha, child_ledger),
    ):
        lake.register_candidate(
            CandidateArtifactRecord(
                cand_sha=sha,
                operator="vec/norm",
                regime_tag="small",
                parent_sha=parent,
                topology_axes=axes.to_dict(),
                base_ledger=ledger.to_dict(),
                microarch_profile=dict(ledger.microarch_profile),
            )
        )

    dims = {"M": 512, "N": 1024}
    pmu_parent = {"dur_us": 30.0, "aiv_vec": 0.60, "aiv_mte2": 0.30}
    pmu_child = {"dur_us": 60.0, "aiv_vec": 0.80, "aiv_mte2": 0.20}

    lake.append_observation(
        CaseObservationRecord(
            run_id="self_run",
            operator="vec/norm",
            regime_tag="small",
            cand_sha=parent_sha,
            eval_tier="baseline",
            case_num=1,
            measured_us=30.0,
            accuracy_passed=True,
            case_dims=dims,
            t_hw_us=30.0,
            msprof_pmu=pmu_parent,
        )
    )
    lake.append_observation(
        CaseObservationRecord(
            run_id="self_run",
            operator="vec/norm",
            regime_tag="small",
            cand_sha=child_sha,
            eval_tier="l2_pass",
            case_num=1,
            measured_us=60.0,
            accuracy_passed=True,
            parent_sha=parent_sha,
            case_dims=dims,
            t_hw_us=30.0,
            msprof_pmu=pmu_child,
        )
    )
    lake.append_observation(
        CaseObservationRecord(
            run_id="external_run",
            operator="vec/norm",
            regime_tag="small",
            cand_sha=other_sha,
            eval_tier="l2_pass",
            case_num=1,
            measured_us=55.0,
            accuracy_passed=True,
            parent_sha=parent_sha,
            case_dims=dims,
            t_hw_us=30.0,
            msprof_pmu={"dur_us": 55.0, "aiv_vec": 0.82, "aiv_mte2": 0.18},
        )
    )

    cm_no_exclude = HybridAscendCostModel(AscendTarget("ascend.950pr"))
    wl_p = project_case_workload(parent_ledger, case_dims=dims, axes=axes)
    wl_c = project_case_workload(child_ledger, case_dims=dims, axes=axes)

    cm_no_exclude.observe_msprof(wl_p, pmu_parent, measured_us=30.0, persist=False)
    cm_no_exclude.observe(wl_p, 30.0, persist=False, refit=False)
    cm_no_exclude.mark_online_observation_key(
        cand_sha=parent_sha,
        operator="vec/norm",
        regime_tag="small",
        case_num=1,
        eval_tier="baseline",
        measured_us=30.0,
    )

    cm_no_exclude.observe_msprof(wl_c, pmu_child, measured_us=60.0, persist=False)
    cm_no_exclude.observe(wl_c, 60.0, persist=False, refit=False)
    cm_no_exclude.mark_online_observation_key(
        cand_sha=child_sha,
        operator="vec/norm",
        regime_tag="small",
        case_num=1,
        eval_tier="l2_pass",
        measured_us=60.0,
    )
    cm_no_exclude.observe_pair_delta(wl_c, 60.0, wl_p, 30.0, refit=False)
    cm_no_exclude.mark_online_pair_key(
        cand_sha=child_sha,
        parent_sha=parent_sha,
        operator="vec/norm",
        regime_tag="small",
        case_num=1,
    )

    stats = cm_no_exclude.replay_experience_lake(lake)
    assert stats == {"observations": 1, "msprof_events": 1, "pair_deltas": 1}
    assert cm_no_exclude.sample_count == 3
    assert cm_no_exclude.msprof_event_count == 3
    assert cm_no_exclude.pair_delta_count == 2

    cm_excluded = HybridAscendCostModel(AscendTarget("ascend.950pr"))
    stats_excl = cm_excluded.replay_experience_lake(lake, exclude_run_id="self_run")
    assert stats_excl == {"observations": 1, "msprof_events": 1, "pair_deltas": 0}
    assert cm_excluded.sample_count == 1
    assert cm_excluded.msprof_event_count == 1
    assert cm_excluded.pair_delta_count == 0


def test_honest_static_and_measured_transfer_inflation_ratio() -> None:
    """Clean streaming kernels yield 1.0, roundtrip spills exceed 1.0, and HBM overrides."""
    floor_bytes = 150 * 1024 * 1024
    spill_bytes = 32 * 1024 * 1024
    roofline = {"min_gm_bytes": floor_bytes, "compute_floor_us": 12.0}

    clean_ledger = _sample_ledger(gm_roundtrip_bytes=0, gm_traffic_scope="full_grid")
    clean_ratio = compute_transfer_inflation_ratio(clean_ledger, roofline=roofline)
    clean_obs = ObservationVector5D.from_tiers(clean_ledger, telemetry={"dur_us": 20.0}, roofline=roofline)
    clean_res = compute_static_dynamic_residual(clean_ledger, {"dur_us": 20.0}, roofline=roofline)
    assert clean_ratio == pytest.approx(1.0)
    assert clean_obs.transfer_inflation_ratio == pytest.approx(1.0)
    assert clean_res.transfer_inflation_ratio == pytest.approx(1.0)

    spill_ledger = _sample_ledger(gm_roundtrip_bytes=spill_bytes, gm_traffic_scope="full_grid")
    expected_spill_ratio = (floor_bytes + spill_bytes) / float(floor_bytes)
    spill_obs = ObservationVector5D.from_tiers(spill_ledger, telemetry={"dur_us": 25.0}, roofline=roofline)
    spill_res = compute_static_dynamic_residual(spill_ledger, {"dur_us": 25.0}, roofline=roofline)
    assert expected_spill_ratio == pytest.approx(1.2133333, rel=1e-5)
    assert spill_obs.transfer_inflation_ratio == pytest.approx(expected_spill_ratio)
    assert spill_res.transfer_inflation_ratio == pytest.approx(expected_spill_ratio)

    measured_tel = {"dur_us": 30.0, "hbm_read_bytes": 120 * 1024 * 1024, "hbm_write_bytes": 60 * 1024 * 1024}
    measured_obs = ObservationVector5D.from_tiers(clean_ledger, telemetry=measured_tel, roofline=roofline)
    measured_res = compute_static_dynamic_residual(clean_ledger, measured_tel, roofline=roofline)
    assert measured_obs.transfer_inflation_ratio == pytest.approx(1.2)
    assert measured_res.transfer_inflation_ratio == pytest.approx(1.2)

    fallback_tel = {"dur_us": 10.0, "transfer_inflation_ratio": 1.45}
    fallback_obs = ObservationVector5D.from_tiers(clean_ledger, telemetry=fallback_tel)
    fallback_res = compute_static_dynamic_residual(clean_ledger, fallback_tel)
    assert fallback_obs.transfer_inflation_ratio == pytest.approx(1.45)
    assert fallback_res.transfer_inflation_ratio == pytest.approx(1.45)


def test_per_regime_snapshots_upgrade_legacy_load_and_beam_roundtrip(tmp_path: Path) -> None:
    """Per-regime snapshots, atomic upgrades, legacy sha files, and BeamCandidate round-trip."""
    lake = ExperienceLake(tmp_path / "lake")
    sha = "deadbeef01"
    parent_sha = "cafebabe00"
    ledger = _sample_ledger(gm_roundtrip_bytes=1024)

    p_small = lake.register_candidate(
        CandidateArtifactRecord(
            cand_sha=sha,
            operator="op/a",
            regime_tag="small",
            parent_sha=parent_sha,
            base_ledger={},
            bisheng_summary={},
        )
    )
    p_large = lake.register_candidate(
        CandidateArtifactRecord(
            cand_sha=sha,
            operator="op/a",
            regime_tag="large",
            parent_sha=parent_sha,
            base_ledger=ledger.to_dict(),
            bisheng_summary={"spill_bytes": 64},
        )
    )
    assert p_small.name == f"{sha}_small.json"
    assert p_large.name == f"{sha}_large.json"
    assert p_small != p_large

    p_upgraded = lake.register_candidate(
        CandidateArtifactRecord(
            cand_sha=sha,
            operator="op/a",
            regime_tag="small",
            parent_sha=parent_sha,
            base_ledger=ledger.to_dict(),
            bisheng_summary={"spill_bytes": 256, "vloopv2_loops": 2},
        )
    )
    assert p_upgraded == p_small

    p_legacy = lake.register_candidate(
        CandidateArtifactRecord(
            cand_sha=parent_sha,
            operator="op/a",
            regime_tag="",
            base_ledger=ledger.to_dict(),
        )
    )
    assert p_legacy.name == f"{parent_sha}.json"

    by_regime, _obs = lake.load_records_by_regime()
    assert by_regime[(sha, "small")].bisheng_summary["spill_bytes"] == 256
    assert by_regime[(sha, "large")].bisheng_summary["spill_bytes"] == 64
    assert (parent_sha, "") in by_regime

    flat_map, _obs_flat = lake.load_records()
    assert sha in flat_map
    assert parent_sha in flat_map

    lake.append_observation(
        CaseObservationRecord(
            run_id="r1",
            operator="op/a",
            regime_tag="small",
            cand_sha=parent_sha,
            eval_tier="baseline",
            case_num=1,
            measured_us=90.0,
            accuracy_passed=True,
        )
    )
    lake.append_observation(
        CaseObservationRecord(
            run_id="r1",
            operator="op/a",
            regime_tag="small",
            cand_sha=sha,
            eval_tier="l2_pass",
            case_num=1,
            measured_us=45.0,
            accuracy_passed=True,
            parent_sha=parent_sha,
        )
    )
    pairs = lake.extract_pairwise_deltas()
    assert len(pairs) == 1
    assert pairs[0][0].regime_tag == "small"
    assert pairs[0][0].bisheng_summary["spill_bytes"] == 256
    assert pairs[0][2].cand_sha == parent_sha

    res = compute_static_dynamic_residual(ledger, {"dur_us": 45.0})
    cand = BeamCandidate(
        candidate_id="c1",
        round_idx=2,
        worker_kind="dsl_mutate",
        rule_id="rule_x",
        dsl_source="def k(): pass",
        track="tilelang_dsl",
        ledger=ledger,
        residual=res,
        latency_us=45.0,
        accuracy_passed=True,
        case_latencies_us={1: 42.0, 2: 48.0},
        policy={"bm": 64, "num_stages": 2},
    )
    serialized = json.loads(json.dumps(cand.to_dict()))
    restored = BeamCandidate.from_dict(serialized)
    assert restored.case_latencies_us == {1: 42.0, 2: 48.0}
    assert restored.policy == {"bm": 64, "num_stages": 2}

    legacy_payload = cand.to_dict()
    legacy_payload.pop("case_latencies_us")
    legacy_payload.pop("policy")
    restored_legacy = BeamCandidate.from_dict(legacy_payload)
    assert restored_legacy.case_latencies_us == {}
    assert restored_legacy.policy == {}


def test_case_observation_record_roofline_fields_and_legacy_compat() -> None:
    """Persist roofline bounds on CaseObservationRecord and default legacy dicts to 0.0."""
    rec = CaseObservationRecord(
        run_id="r_roof",
        operator="op/a",
        regime_tag="small",
        cand_sha="abc123",
        eval_tier="l2_pass",
        case_num=1,
        measured_us=25.0,
        accuracy_passed=True,
        min_gm_bytes=65536.0,
        compute_floor_us=10.5,
        transfer_inflation_ratio=1.35,
    )
    payload = rec.to_dict()
    restored = CaseObservationRecord.from_dict(payload)
    assert restored.min_gm_bytes == pytest.approx(65536.0)
    assert restored.compute_floor_us == pytest.approx(10.5)
    assert restored.transfer_inflation_ratio == pytest.approx(1.35)

    legacy_payload = dict(payload)
    legacy_payload.pop("min_gm_bytes")
    legacy_payload.pop("compute_floor_us")
    legacy_payload.pop("transfer_inflation_ratio")
    restored_legacy = CaseObservationRecord.from_dict(legacy_payload)
    assert restored_legacy.min_gm_bytes == 0.0
    assert restored_legacy.compute_floor_us == 0.0
    assert restored_legacy.transfer_inflation_ratio == 0.0
