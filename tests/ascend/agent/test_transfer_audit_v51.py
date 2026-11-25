"""Unit and CLI integration tests for v5.1 cross-topology transfer audit."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tilefoundry_ascend.agent.campaign.transfer_audit import (
    format_topology_quadrant,
    main,
    render_transfer_audit_markdown,
    summarize_funnel_and_transfer,
)

Q1_AXES: dict[str, str] = {
    "compute_engine": "aiv_only",
    "memory_ring": "gm_ub_stream",
    "iteration_regularity": "dense_static",
}
Q2_AXES: dict[str, str] = {
    "compute_engine": "aic_only",
    "memory_ring": "gm_l1_l0_fixpipe",
    "iteration_regularity": "ragged_group",
}
Q3_AXES: dict[str, str] = {
    "compute_engine": "coupled_1c2v",
    "memory_ring": "bidirectional_cv_ring",
    "iteration_regularity": "irregular_simt",
}
Q4_AXES: dict[str, str] = {
    "compute_engine": "coupled_1c2v",
    "memory_ring": "persistent_onchip_state",
    "iteration_regularity": "dense_static",
}


def _make_event(
    *,
    operator: str,
    axes: dict[str, str],
    rule_id: str = "",
    round_idx: int = 1,
    passed_l0_l1: bool = True,
    error_code_norm: str = "OK",
    noop_identical: bool = False,
    promoted_l3: bool = False,
    delta_speedup: float = 0.0,
    wall_sec: float = 10.0,
    analytical_pred_us: float = 0.0,
    measured_l2_us: float = 0.0,
    burst_batch_cv: float = 0.0,
    l3_failure_stage: str = "",
) -> dict[str, object]:
    return {
        "operator": operator,
        "topology_axes": dict(axes),
        "rule_id": rule_id,
        "round_idx": round_idx,
        "passed_l0_l1": passed_l0_l1,
        "error_code_norm": error_code_norm,
        "noop_identical": noop_identical,
        "promoted_l3": promoted_l3,
        "delta_speedup": delta_speedup,
        "wall_sec": wall_sec,
        "analytical_pred_us": analytical_pred_us,
        "measured_l2_us": measured_l2_us,
        "burst_batch_cv": burst_batch_cv,
        "l3_failure_stage": l3_failure_stage,
    }


def _populate_experience_lake(lake_dir: Path) -> None:
    kb_rules_dir = lake_dir / "kb" / "distilled_rules"
    kb_rules_dir.mkdir(parents=True, exist_ok=True)
    (kb_rules_dir / "D01_CV_RING_HANDOFF.json").write_text(
        json.dumps(
            {
                "rule_id": "D01_CV_RING_HANDOFF",
                "origin_operator": "flash_attn",
                "source_provenance": "ablation_distiller:flash_attn",
            }
        ),
        encoding="utf-8",
    )

    flat_rules_dir = lake_dir / "distilled_rules"
    flat_rules_dir.mkdir(parents=True, exist_ok=True)
    (flat_rules_dir / "D02_PERSISTENT_CACHE.json").write_text(
        json.dumps(
            {
                "rule_id": "D02_PERSISTENT_CACHE",
                "origin_operator": "rmsnorm",
                "topology_axes": Q4_AXES,
                "source_provenance": "ablation_distiller:rmsnorm",
            }
        ),
        encoding="utf-8",
    )

    kb_evidence_dir = lake_dir / "kb" / "evidence"
    kb_evidence_dir.mkdir(parents=True, exist_ok=True)
    kb_edges = [
        {
            "operator": "flash_attn",
            "round_idx": 1,
            "rule_id": "R01_DUAL_COPY_1C2V_SPLIT",
            "staged_rule_ids": ["D01_CV_RING_HANDOFF"],
            "topology_axes": Q1_AXES,
            "relative_gain": 0.32,
        },
        {
            "operator": "matmul_dense",
            "round_idx": 2,
            "rule_id": "R08_PERSISTENT_SWIZZLE",
            "staged_rule_ids": [],
            "topology_axes": Q1_AXES,
            "relative_gain": 0.18,
        },
    ]
    (kb_evidence_dir / "winning_edges.jsonl").write_text(
        "\n".join(json.dumps(edge) for edge in kb_edges) + "\n",
        encoding="utf-8",
    )

    flat_edge = {
        "operator": "rmsnorm",
        "round_idx": 1,
        "rule_id": "D02_PERSISTENT_CACHE",
        "staged_rule_ids": ["D02_PERSISTENT_CACHE"],
        "topology_axes": Q4_AXES,
        "relative_gain": 0.21,
    }
    (lake_dir / "winning_edges.jsonl").write_text(
        json.dumps(flat_edge) + "\n",
        encoding="utf-8",
    )


def _build_four_quadrant_events() -> list[dict[str, object]]:
    events: list[dict[str, object]] = [
        _make_event(
            operator="vec_add",
            axes=Q1_AXES,
            rule_id="R08_PERSISTENT_SWIZZLE",
            round_idx=1,
            promoted_l3=True,
            delta_speedup=0.14,
            analytical_pred_us=100.0,
            measured_l2_us=110.0,
            burst_batch_cv=0.03,
        ),
        _make_event(
            operator="moe_gmm",
            axes=Q2_AXES,
            rule_id="R08_PERSISTENT_SWIZZLE",
            round_idx=2,
            promoted_l3=True,
            delta_speedup=0.22,
            analytical_pred_us=50.0,
            measured_l2_us=220.0,
            burst_batch_cv=0.04,
        ),
        _make_event(
            operator="gqa_decode",
            axes=Q3_AXES,
            rule_id="R08_PERSISTENT_SWIZZLE",
            round_idx=3,
            promoted_l3=False,
            delta_speedup=0.0,
            l3_failure_stage="numerical_accuracy",
        ),
        _make_event(
            operator="layernorm",
            axes=Q4_AXES,
            rule_id="R08_PERSISTENT_SWIZZLE",
            round_idx=4,
            promoted_l3=True,
            delta_speedup=0.12,
        ),
        _make_event(
            operator="gqa_decode",
            axes=Q3_AXES,
            rule_id="D01_CV_RING_HANDOFF",
            round_idx=1,
            noop_identical=True,
            promoted_l3=False,
            delta_speedup=0.0,
        ),
        _make_event(
            operator="gqa_decode",
            axes=Q3_AXES,
            rule_id="D01_CV_RING_HANDOFF",
            round_idx=2,
            promoted_l3=True,
            delta_speedup=0.28,
        ),
        _make_event(
            operator="gqa_decode",
            axes=Q3_AXES,
            rule_id="D01_CV_RING_HANDOFF",
            round_idx=3,
            promoted_l3=True,
            delta_speedup=0.16,
        ),
        _make_event(
            operator="layernorm",
            axes=Q4_AXES,
            rule_id="D01_CV_RING_HANDOFF",
            round_idx=4,
            promoted_l3=False,
            delta_speedup=0.0,
            l3_failure_stage="tail_guard_regression",
        ),
        _make_event(
            operator="layernorm",
            axes=Q4_AXES,
            rule_id="D02_PERSISTENT_CACHE",
            round_idx=2,
            promoted_l3=True,
            delta_speedup=0.25,
        ),
        _make_event(
            operator="vec_add",
            axes=Q1_AXES,
            rule_id="R19_UNROLL_NO_ALIAS",
            round_idx=1,
            noop_identical=True,
        ),
        _make_event(
            operator="vec_add",
            axes=Q1_AXES,
            rule_id="R19_UNROLL_NO_ALIAS",
            round_idx=2,
            noop_identical=True,
        ),
        _make_event(
            operator="vec_add",
            axes=Q1_AXES,
            rule_id="R19_UNROLL_NO_ALIAS",
            round_idx=3,
            noop_identical=False,
        ),
        _make_event(
            operator="moe_gmm",
            axes=Q2_AXES,
            rule_id="R19_UNROLL_NO_ALIAS",
            round_idx=1,
            noop_identical=False,
            promoted_l3=False,
        ),
    ]

    for idx in range(5):
        events.append(
            _make_event(
                operator="vec_add",
                axes=Q1_AXES,
                rule_id="R23_BOUNDARY_NOOP",
                round_idx=idx + 1,
                noop_identical=(idx < 2),
                promoted_l3=(idx == 4),
                delta_speedup=0.08 if idx == 4 else 0.0,
            )
        )

    compile_error_specs = [
        (False, "CCE_FIXPIPE_LAYOUT"),
        (False, "CCE_FIXPIPE_LAYOUT"),
        (False, "CCE_FIXPIPE_LAYOUT"),
        (False, "SYNC_BARRIER_MISSING"),
        (False, "COST_COARSE_REJECT"),
        (False, "COST_COARSE_REJECT"),
        (True, "OK"),
    ]
    for idx, (passed, err_code) in enumerate(compile_error_specs):
        events.append(
            _make_event(
                operator="moe_gmm",
                axes=Q2_AXES,
                round_idx=10 + idx,
                passed_l0_l1=passed,
                error_code_norm=err_code,
            )
        )

    events.extend(
        [
            _make_event(
                operator="moe_gmm",
                axes=Q2_AXES,
                round_idx=20,
                analytical_pred_us=50.0,
                measured_l2_us=52.0,
                burst_batch_cv=0.15,
            ),
            _make_event(
                operator="moe_gmm",
                axes=Q2_AXES,
                round_idx=21,
                analytical_pred_us=40.0,
                measured_l2_us=180.0,
                burst_batch_cv=0.02,
            ),
        ]
    )
    return events


def test_format_topology_quadrant_canonical_keys() -> None:
    """Verify canonical 3-axis TopologyAxes quadrant keys and unknown fallback."""
    assert format_topology_quadrant(Q1_AXES) == "aiv_only.gm_ub_stream.dense_static"
    assert format_topology_quadrant(Q2_AXES) == "aic_only.gm_l1_l0_fixpipe.ragged_group"
    assert (
        format_topology_quadrant(Q3_AXES)
        == "coupled_1c2v.bidirectional_cv_ring.irregular_simt"
    )
    assert (
        format_topology_quadrant(Q4_AXES)
        == "coupled_1c2v.persistent_onchip_state.dense_static"
    )
    assert format_topology_quadrant(None) == "unknown_topology"
    assert format_topology_quadrant({}) == "unknown_topology"


def test_transfer_audit_classifications_alerts_and_calibration(tmp_path: Path) -> None:
    """Verify rule classifications, >30% compile wall alerts, contention stats, and refit."""
    lake_dir = tmp_path / "experience_lake"
    _populate_experience_lake(lake_dir)
    events = _build_four_quadrant_events()

    report = summarize_funnel_and_transfer(events, experience_lake_dir=lake_dir)

    cross_transfer = report["cross_topology_transfer"]
    assert isinstance(cross_transfer, list)
    transfer_by_rule = {
        str(row["rule_id"]): row
        for row in cross_transfer
        if isinstance(row, dict)
    }
    assert transfer_by_rule["R08_PERSISTENT_SWIZZLE"]["classification"] == "universal_transferable"
    assert transfer_by_rule["R08_PERSISTENT_SWIZZLE"]["quadrant_transfer_rate"] == 0.75
    assert transfer_by_rule["R08_PERSISTENT_SWIZZLE"]["rule_origin"] == "seed"

    assert transfer_by_rule["D01_CV_RING_HANDOFF"]["classification"] == "topology_specialized"
    assert transfer_by_rule["D01_CV_RING_HANDOFF"]["quadrant_transfer_rate"] == 0.5
    assert transfer_by_rule["D01_CV_RING_HANDOFF"]["rule_origin"] == "distilled"

    assert (
        transfer_by_rule["R19_UNROLL_NO_ALIAS"]["classification"]
        == "needs_precondition_tightening"
    )
    assert transfer_by_rule["R19_UNROLL_NO_ALIAS"]["quadrant_transfer_rate"] == 0.0

    low_eff = report["low_efficiency_rules"]
    assert isinstance(low_eff, list)
    low_eff_ids = {(str(item["rule_id"]), str(item["quadrant"])) for item in low_eff}
    assert ("R19_UNROLL_NO_ALIAS", format_topology_quadrant(Q1_AXES)) in low_eff_ids
    assert ("R23_BOUNDARY_NOOP", format_topology_quadrant(Q1_AXES)) not in low_eff_ids
    r19_entry = next(item for item in low_eff if item["rule_id"] == "R19_UNROLL_NO_ALIAS")
    assert r19_entry["noop_rate"] == pytest.approx(0.6667, abs=1e-4)
    assert r19_entry["needs_precondition_tightening"] is True

    alerts = report["compile_wall_alerts"]
    assert isinstance(alerts, list)
    assert len(alerts) == 1
    assert alerts[0]["operator"] == "moe_gmm"
    assert alerts[0]["error_code_norm"] == "CCE_FIXPIPE_LAYOUT"
    assert alerts[0]["count"] == 3
    assert alerts[0]["share"] == 0.75

    noise = report["contention_noise_summary"]
    assert isinstance(noise, dict)
    assert noise["total_l2_events"] == 4
    assert noise["clean_l2_events"] == 3
    assert noise["noisy_l2_events"] == 1
    assert noise["noisy_share"] == 0.25

    calibration = report["cost_calibration_by_quadrant"]
    assert isinstance(calibration, dict)
    q1_cal = calibration[format_topology_quadrant(Q1_AXES)]
    q2_cal = calibration[format_topology_quadrant(Q2_AXES)]
    assert q1_cal["clean_samples"] == 1
    assert q1_cal["median_abs_log_error"] == pytest.approx(abs(math.log(1.1)), abs=1e-4)
    assert q1_cal["needs_tier2_refit"] is False
    assert q2_cal["clean_samples"] == 2
    assert q2_cal["needs_tier2_refit"] is True

    assert report["l3_failure_stage_counts"] == {
        "numerical_accuracy": 1,
        "tail_guard_regression": 1,
    }


def test_distilled_rule_directed_transfer_matrix_and_compat(tmp_path: Path) -> None:
    """Verify directed distilled-rule transfer matrix and optional lake compatibility."""
    events = _build_four_quadrant_events()
    compat_report = summarize_funnel_and_transfer(events)
    assert compat_report["distilled_rule_transfer_matrix"] == []
    compat_cross = compat_report["cross_topology_transfer"]
    assert isinstance(compat_cross, list)
    assert len(compat_cross) >= 3

    lake_dir = tmp_path / "lake"
    _populate_experience_lake(lake_dir)
    report = summarize_funnel_and_transfer(events, experience_lake_dir=lake_dir)
    matrix = report["distilled_rule_transfer_matrix"]
    assert isinstance(matrix, list)

    by_edge = {
        (
            str(row["rule_id"]),
            str(row["source_operator"]),
            str(row["target_operator"]),
            str(row["target_quadrant"]),
        ): row
        for row in matrix
    }

    q1_key = format_topology_quadrant(Q1_AXES)
    q2_key = format_topology_quadrant(Q2_AXES)
    q3_key = format_topology_quadrant(Q3_AXES)
    q4_key = format_topology_quadrant(Q4_AXES)

    d01_edge = by_edge[("D01_CV_RING_HANDOFF", "flash_attn", "gqa_decode", q3_key)]
    assert d01_edge["source_quadrant"] == q1_key
    assert d01_edge["rule_origin"] == "distilled"
    assert d01_edge["dispatches"] == 3
    assert d01_edge["l3_promoted_count"] == 2
    assert d01_edge["first_hit_delta_speedup"] == 0.28
    assert d01_edge["mean_delta_speedup"] == 0.22

    d02_edge = by_edge[("D02_PERSISTENT_CACHE", "rmsnorm", "layernorm", q4_key)]
    assert d02_edge["source_quadrant"] == q4_key
    assert d02_edge["rule_origin"] == "distilled"
    assert d02_edge["first_hit_delta_speedup"] == 0.25

    assert ("R08_PERSISTENT_SWIZZLE", "matmul_dense", "moe_gmm", q2_key) not in by_edge


def test_cli_end_to_end_json_and_markdown_outputs(tmp_path: Path) -> None:
    """Verify CLI executes end-to-end with base-dir, lake-dir, JSON, and Markdown outputs."""
    base_dir = tmp_path / "marathon"
    op1_dir = base_dir / "gqa_decode" / "telemetry"
    op2_dir = base_dir / "moe_gmm" / "telemetry"
    op1_dir.mkdir(parents=True, exist_ok=True)
    op2_dir.mkdir(parents=True, exist_ok=True)

    events = _build_four_quadrant_events()
    half = len(events) // 2
    (op1_dir / "round_funnel_events.jsonl").write_text(
        "\n".join(json.dumps(ev) for ev in events[:half]) + "\n",
        encoding="utf-8",
    )
    (op2_dir / "round_funnel_events.jsonl").write_text(
        "\n".join(json.dumps(ev) for ev in events[half:]) + "\n",
        encoding="utf-8",
    )

    lake_dir = tmp_path / "lake"
    _populate_experience_lake(lake_dir)

    out_json = tmp_path / "reports" / "audit.json"
    out_md = tmp_path / "reports" / "audit.md"

    rc = main(
        [
            "--base-dir",
            str(base_dir),
            "--experience-lake-dir",
            str(lake_dir),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ]
    )
    assert rc == 0
    assert out_json.is_file()
    assert out_md.is_file()

    parsed = json.loads(out_json.read_text(encoding="utf-8"))
    assert "distilled_rule_transfer_matrix" in parsed
    assert "low_efficiency_rules" in parsed
    assert len(parsed["compile_wall_alerts"]) == 1

    md_text = out_md.read_text(encoding="utf-8")
    assert md_text == render_transfer_audit_markdown(parsed)
    assert "# Cross-Topology Causal Rule Transfer Audit" in md_text
    assert "## Distilled Rule Directed Transfer Matrix" in md_text
    assert "## Low-Efficiency Rules (Precondition Tightening Needed)" in md_text
    assert "D01_CV_RING_HANDOFF" in md_text
    assert "D02_PERSISTENT_CACHE" in md_text
    assert "CCE_FIXPIPE_LAYOUT" in md_text
