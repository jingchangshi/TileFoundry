"""Unit tests for distiller ideal env, ExperienceLake schema, funnel telemetry, and anchors."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import yaml

from tilefoundry_ascend.agent.specs.anchor_specs import (
    AnchorCaseSpec,
    RegimeTargetSpec,
    inject_lake_counterexample_anchors,
)
from tilefoundry_ascend.agent.specs.fault_types import FaultPlane, TypedFault
from tilefoundry_ascend.agent.campaign.funnel_telemetry import (
    MAX_CLEAN_CALIBRATION_BURST_CV,
    SPEARMAN_QUARANTINE_FLOOR,
    DeterministicSearchBoundaryLedger,
    RoundFunnelEvent,
    append_round_funnel_event,
    assert_unmodeled_fault_share_bounded,
    compute_spearman_rho,
    is_clean_calibration_sample,
)
from tilefoundry_ascend.agent.flywheel.ablation_distiller import (
    AscendCausalRule,
    SymptomFamily,
    _build_zero_ideal_env,
    _validate_distilled_rule,
)
from tilefoundry_ascend.ascend.cost_model.experience_lake import (
    CaseObservationRecord,
    append_counterexample_boundary,
    load_counterexample_boundaries,
)


def _make_rule(*, precondition: str, expected_postcondition: str) -> AscendCausalRule:
    return AscendCausalRule(
        rule_id="R_V51_TEST",
        regime_guard="is_legal and any_overflow_bytes == 0",
        precondition=precondition,
        transform_template="Increase burst coalescing factor and tile occupancy.",
        expected_postcondition=expected_postcondition,
        source_provenance="flywheel:ablation",
        compatible_backends=("tilelang_dsl",),
        symptom_family=SymptomFamily.DMA_UNALIGNED_SHORT_BURST.value,
    )


def test_distiller_accepts_efficiency_ratios_and_rejects_tautologies() -> None:
    """Efficiency ratio predicates pass validation while tautologies stay rejected."""
    dma_rule = _make_rule(
        precondition="dma_burst_efficiency_ratio < 0.6 and PIPE_MTE2 > 0",
        expected_postcondition="dma_burst_efficiency_ratio > prev.dma_burst_efficiency_ratio",
    )
    assert _validate_distilled_rule(dma_rule) == (True, "")

    eta_rule = _make_rule(
        precondition="effective_compute_eta < 0.4",
        expected_postcondition="effective_compute_eta > prev.effective_compute_eta",
    )
    assert _validate_distilled_rule(eta_rule) == (True, "")

    contig_rule = _make_rule(
        precondition="dominant_contig_bytes < 4096",
        expected_postcondition="dominant_contig_bytes > prev.dominant_contig_bytes",
    )
    assert _validate_distilled_rule(contig_rule) == (True, "")

    env = _build_zero_ideal_env(
        ast.parse(
            "dma_burst_efficiency_ratio < 0.6 and effective_compute_eta < 0.4 "
            "and dominant_contig_bytes < 4096 and is_atom_aligned",
            mode="eval",
        )
    )
    assert env["dma_burst_efficiency_ratio"] == 1.0
    assert env["effective_compute_eta"] == 1.0
    assert env["dominant_contig_bytes"] == 32768
    assert env["is_atom_aligned"] is True

    taut_rule = _make_rule(
        precondition="ub_spill_bytes >= 0",
        expected_postcondition="ub_spill_bytes < prev.ub_spill_bytes",
    )
    ok_taut, reason_taut = _validate_distilled_rule(taut_rule)
    assert ok_taut is False
    assert "tautological" in reason_taut

    nonstrict_rule = _make_rule(
        precondition="dma_burst_efficiency_ratio < 0.6",
        expected_postcondition="x <= prev.x",
    )
    ok_post, reason_post = _validate_distilled_rule(nonstrict_rule)
    assert ok_post is False
    assert "non-strict" in reason_post


def test_case_observation_record_v3_roundtrip_and_v2_compat() -> None:
    """CaseObservationRecord defaults to v3 and backfills v2 payloads cleanly."""
    rec = CaseObservationRecord(
        run_id="run_v51",
        operator="grouped_matmul",
        regime_tag="unaligned_tail",
        cand_sha="deadbeef",
        eval_tier="l2_pass",
        case_num=14,
        measured_us=42.5,
        accuracy_passed=True,
        analytical_pred_us=38.0,
        ring_burst_k=5,
        burst_batch_cv=0.04,
        spike_retries=1,
        card_ext_hbm_mb=2048,
    )
    assert rec.schema_version == 3
    payload = rec.to_dict()
    restored = CaseObservationRecord.from_dict(payload)
    assert restored == rec

    legacy = dict(payload)
    legacy["schema_version"] = 2
    for key in (
        "analytical_pred_us",
        "ring_burst_k",
        "burst_batch_cv",
        "spike_retries",
        "card_ext_hbm_mb",
    ):
        legacy.pop(key)
    restored_legacy = CaseObservationRecord.from_dict(legacy)
    assert restored_legacy.schema_version == 2
    assert restored_legacy.analytical_pred_us == 0.0
    assert restored_legacy.ring_burst_k == 1
    assert restored_legacy.burst_batch_cv == 0.0
    assert restored_legacy.spike_retries == 0
    assert restored_legacy.card_ext_hbm_mb == 0


def test_counterexample_boundary_append_and_load(tmp_path: Path) -> None:
    """Counterexample boundaries persist to JSONL and filter by operator."""
    lake_dir = tmp_path / "lake"
    p1 = append_counterexample_boundary(
        lake_dir,
        {
            "operator": "grouped_matmul",
            "regime_tag": "unaligned_tail",
            "case_num": 14,
            "failure_stage": "l3_accuracy",
            "has_unaligned_dim": True,
        },
    )
    p2 = append_counterexample_boundary(
        lake_dir,
        {
            "operator": "level4/mla",
            "regime_tag": "split_kv_decode",
            "case_num": 7,
            "failure_stage": "l3_timeout",
        },
    )
    assert p1 == p2
    assert p1.name == "counterexample_anchors.jsonl"

    all_recs = load_counterexample_boundaries(lake_dir)
    assert len(all_recs) == 2

    gmm_recs = load_counterexample_boundaries(lake_dir, operator="grouped_matmul")
    assert len(gmm_recs) == 1
    assert gmm_recs[0]["case_num"] == 14

    mla_recs = load_counterexample_boundaries(lake_dir, operator="mla")
    assert len(mla_recs) == 1
    assert mla_recs[0]["case_num"] == 7


def test_funnel_telemetry_events_and_typed_fault_contract(tmp_path: Path) -> None:
    """Funnel telemetry preserves TypedFault planes, gates noise, and persists JSONL."""
    evt_ok = RoundFunnelEvent.from_typed_fault(
        run_id="r1",
        operator="grouped_matmul",
        round_idx=1,
        regime_tag="unaligned_tail",
        topology_axes={"compute_engine": "aic_only", "memory_ring": "gm_l1_l0_fixpipe"},
        candidate_id="cand_ok",
        cand_sha="abc",
        worker_kind="rule_codegen",
        fault=None,
    )
    assert evt_ok.error_code_norm == "OK"
    assert evt_ok.fault_plane == FaultPlane.NONE.value
    assert evt_ok.passed_l0_l1 is True

    evt_noop = evt_ok.apply_typed_fault(
        TypedFault(
            plane=FaultPlane.NOOP_IDENTICAL,
            code="NOOP_IDENTICAL",
            failed_tier="noop_identical",
            detail="identical",
        )
    )
    assert evt_noop.fault_plane == FaultPlane.NOOP_IDENTICAL.value
    assert evt_noop.error_code_norm == "NOOP_IDENTICAL"
    assert evt_noop.noop_identical is True
    assert_unmodeled_fault_share_bounded([evt_ok, evt_noop])

    assert is_clean_calibration_sample(MAX_CLEAN_CALIBRATION_BURST_CV) is True
    assert is_clean_calibration_sample(MAX_CLEAN_CALIBRATION_BURST_CV + 0.01) is False
    assert is_clean_calibration_sample(float("nan")) is False

    evt = RoundFunnelEvent(
        run_id="r1",
        operator="grouped_matmul",
        round_idx=2,
        regime_tag="unaligned_tail",
        topology_axes={"compute_engine": "cube", "memory_ring": "ring"},
        candidate_id="cand_0",
        cand_sha="abc",
        worker_kind="rule_codegen",
        rule_id="R01",
        passed_l0_l1=True,
        error_code_norm="OK",
        analytical_pred_us=20.0,
        measured_l2_us=22.0,
    )
    out = append_round_funnel_event(tmp_path / "telemetry", evt)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    decoded = json.loads(lines[0])
    assert decoded["candidate_id"] == "cand_0"
    assert decoded["measured_l2_us"] == 22.0


def test_boundary_ledger_roundtrip_precheck_prompt_and_quarantine() -> None:
    """Boundary ledger tracks compile walls, objective prompts, and Spearman quarantine."""
    ledger = DeterministicSearchBoundaryLedger()
    assert (
        ledger.record_compile_wall(
            regime_tag="prefill",
            cand_sha="sha_ub",
            knob_signature="M=256,N=256,K=128",
            error_code_norm="UB_SRAM_OVERFLOW",
            ub_bytes=262144,
        )
        is True
    )
    assert (
        ledger.record_compile_wall(
            regime_tag="prefill",
            cand_sha="sha_ub2",
            knob_signature="M=256,N=128,K=128",
            error_code_norm="UB_SRAM_OVERFLOW",
            ub_bytes=196608,
        )
        is True
    )
    assert ledger.min_overflow_ub_bytes["prefill"] == 196608
    assert (
        ledger.record_compile_wall(
            regime_tag="prefill",
            cand_sha="sha_stack",
            knob_signature="M=128,N=128,K=256",
            error_code_norm="VF_STACK_OVERFLOW",
            stack_bytes=16384,
        )
        is True
    )

    assert (
        ledger.precheck_known_infeasible(regime_tag="prefill", cand_sha="sha_ub")
        == "deterministic_boundary_sha:UB_SRAM_OVERFLOW"
    )
    assert (
        ledger.precheck_known_infeasible(
            regime_tag="prefill",
            cand_sha="fresh_sha",
            knob_signature="M=256,N=256,K=128",
        )
        == "deterministic_boundary_knobs:UB_SRAM_OVERFLOW"
    )
    assert (
        ledger.precheck_known_infeasible(
            regime_tag="prefill",
            cand_sha="fresh_sha",
            knob_signature="M=64,N=64,K=64",
        )
        is None
    )

    prompt = ledger.format_regime_boundary_prompt("prefill")
    assert "ub_bytes >= 196608" in prompt
    assert "stack_bytes=16384" in prompt
    assert "M=256,N=256,K=128->UB_SRAM_OVERFLOW" in prompt
    assert ledger.format_regime_boundary_prompt("unseen_regime") == ""

    assert compute_spearman_rho({"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 2.0}) is None
    assert compute_spearman_rho(
        {"a": 10.0, "b": 20.0, "c": 30.0},
        {"a": 15.0, "b": 25.0, "c": 35.0},
    ) == pytest.approx(1.0)
    rho_inv = compute_spearman_rho(
        {"a": 10.0, "b": 20.0, "c": 30.0},
        {"a": 35.0, "b": 25.0, "c": 15.0},
    )
    assert rho_inv == pytest.approx(-1.0)

    ledger.record_spearman_rho("prefill", 0.15)
    assert ledger.is_cost_prescreen_quarantined("prefill") is False
    ledger.record_spearman_rho("prefill", 0.50)
    assert ledger.is_cost_prescreen_quarantined("prefill") is False
    ledger.record_spearman_rho("prefill", 0.10)
    assert ledger.is_cost_prescreen_quarantined("prefill") is False
    ledger.record_spearman_rho("prefill", SPEARMAN_QUARANTINE_FLOOR - 0.05)
    assert ledger.is_cost_prescreen_quarantined("prefill") is True

    restored = DeterministicSearchBoundaryLedger.from_dict(ledger.to_dict())
    assert restored == ledger


def _write_synthetic_cannbench(
    root: Path,
    *,
    task: str,
    cases: list[dict[str, object]],
    metadata: dict[str, dict[str, float]],
) -> Path:
    level, op_key = task.split("/", 1)
    task_dir = root / "tasks" / level / op_key
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "cases.yaml").write_text(yaml.safe_dump({"cases": cases}), encoding="utf-8")
    (task_dir / "proto.yaml").write_text(
        yaml.safe_dump({"operator": {"name": op_key, "inputs": [{"name": "x", "shape": ["M", "K"]}]}}),
        encoding="utf-8",
    )
    meta_dir = root / "tasks" / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "950pr.json").write_text(
        json.dumps({level: {op_key: metadata}}),
        encoding="utf-8",
    )
    return root


def test_inject_lake_counterexample_anchors_same_op_case_14(tmp_path: Path) -> None:
    """Same-operator counterexample injection prioritizes grouped_matmul case 14."""
    lake_dir = tmp_path / "lake"
    repo_dir = _write_synthetic_cannbench(
        tmp_path / "cannbench",
        task="level3/grouped_matmul",
        cases=[
            {"case_id": 1, "input_shape": [[64, 64]]},
            {"case_id": 11, "input_shape": [[128, 128]]},
            {"case_id": 14, "input_shape": [[128, 77]]},
        ],
        metadata={
            "1": {"baseline_perf_us": 50.0, "t_hw_us": 10.0},
            "11": {"baseline_perf_us": 80.0, "t_hw_us": 20.0},
            "14": {"baseline_perf_us": 95.0, "t_hw_us": 25.0},
        },
    )

    initial = [
        AnchorCaseSpec("level3/grouped_matmul", 1, "small_latency", 50.0, 10.0),
    ]
    regimes = [
        RegimeTargetSpec("small_latency", "small_latency", "Small cases"),
        RegimeTargetSpec("unaligned_tail", "unaligned_tail", "Unaligned tail cases"),
    ]

    for _ in range(3):
        append_counterexample_boundary(
            lake_dir,
            {
                "operator": "grouped_matmul",
                "regime_tag": "unaligned_tail",
                "case_num": 14,
                "failure_stage": "l3_accuracy",
                "has_unaligned_dim": True,
            },
        )
    append_counterexample_boundary(
        lake_dir,
        {
            "operator": "grouped_matmul",
            "regime_tag": "",
            "case_num": 11,
            "case_dims": {"total_elements": 2097152},
            "failure_stage": "l3_timeout",
        },
    )
    append_counterexample_boundary(
        lake_dir,
        {
            "operator": "grouped_matmul",
            "regime_tag": "small_latency",
            "case_num": 1,
            "failure_stage": "l3_accuracy",
        },
    )

    injected = inject_lake_counterexample_anchors(
        initial,
        operator_name="grouped_matmul",
        regime_specs=regimes,
        experience_lake_dir=lake_dir,
        cannbench_repo=repo_dir,
        max_injected=1,
    )
    assert [a.case_num for a in injected] == [1, 14]
    assert injected[1].regime_tag == "unaligned_tail"
    assert injected[1].ref_perf_us == 95.0
    assert injected[1].t_hw_us == 25.0


def test_inject_lake_counterexample_anchors_cross_op_tail_fallback(tmp_path: Path) -> None:
    """Cross-operator odd/prime tail failure triggers scanning operator cases.yaml."""
    lake_dir = tmp_path / "lake"
    repo_dir = _write_synthetic_cannbench(
        tmp_path / "cannbench",
        task="level4/attention_residual",
        cases=[
            {"case_id": 2, "input_shape": [[1, 64, 128]]},
            {"case_id": 9, "input_shape": [[1, 32, 77]]},
        ],
        metadata={
            "2": {"baseline_perf_us": 20.0, "t_hw_us": 4.0},
            "9": {"baseline_perf_us": 45.0, "t_hw_us": 9.0},
        },
    )
    append_counterexample_boundary(
        lake_dir,
        {
            "operator": "grouped_matmul",
            "regime_tag": "unaligned_tail",
            "topology_axes": {
                "compute_engine": "cube",
                "memory_ring": "ring",
                "iteration_regularity": "irregular",
            },
            "case_num": 14,
            "case_dims": {"K": 77},
            "odd_or_prime_tail": True,
        },
    )

    initial = [
        AnchorCaseSpec("level4/attention_residual", 2, "small_decode", 20.0, 4.0),
    ]
    regimes = [
        RegimeTargetSpec("small_decode", "small_decode", "Small decode"),
        RegimeTargetSpec("unaligned_tail", "unaligned_tail", "Unaligned tail"),
    ]

    injected = inject_lake_counterexample_anchors(
        initial,
        operator_name="attention_residual",
        regime_specs=regimes,
        experience_lake_dir=lake_dir,
        cannbench_repo=repo_dir,
        topology_axes={"compute_engine": "cube", "memory_ring": "ring"},
    )
    assert [a.case_num for a in injected] == [2, 9]
    assert injected[1].regime_tag == "unaligned_tail"
    assert injected[1].ref_perf_us == 45.0

    unchanged = inject_lake_counterexample_anchors(
        initial,
        operator_name="attention_residual",
        regime_specs=regimes,
        experience_lake_dir=lake_dir,
        cannbench_repo=repo_dir,
        topology_axes={"compute_engine": "vector", "memory_ring": "direct"},
    )
    assert [a.case_num for a in unchanged] == [2]

    assert (
        inject_lake_counterexample_anchors(
            initial,
            operator_name="attention_residual",
            regime_specs=regimes,
            experience_lake_dir=None,
        )
        == initial
    )
    empty_repo = tmp_path / "empty_cannbench"
    (empty_repo / "tasks" / "level4").mkdir(parents=True, exist_ok=True)
    assert (
        inject_lake_counterexample_anchors(
            initial,
            operator_name="attention_residual",
            regime_specs=regimes,
            experience_lake_dir=lake_dir,
            cannbench_repo=empty_repo,
        )
        == initial
    )
    assert DeterministicSearchBoundaryLedger.from_dict(None) == DeterministicSearchBoundaryLedger()
