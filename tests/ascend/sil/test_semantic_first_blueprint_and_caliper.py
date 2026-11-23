"""Tests for semantic-first blueprints, explicit ledger traffic scope, and caliper solving."""

from __future__ import annotations

import dataclasses
import json
import math
import shutil
from pathlib import Path
from textwrap import dedent

import pytest
import torch

from tilefoundry_ascend.agent import verify_reference_cpu_parity as agent_verify_parity
from tilefoundry_ascend.agent.orchestrator import (
    verify_reference_cpu_parity as orchestrator_verify_parity,
)
from tilefoundry_ascend.agent.orchestrator.e2e_evolution import (
    _compile_blueprint_safe,
    _summarize_blueprint,
)
from tilefoundry_ascend.agent.orchestrator.twin_bridge import verify_reference_cpu_parity
from tilefoundry_ascend.ascend.cost_model.calibration_backtest import (
    load_official_reports,
    run_cost_model_backtest,
)
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import (
    BlueprintProvenance,
    BlueprintScheduleDelta,
    HirBlueprintPacket,
    HirRooflineSummary,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    ObjectivePhysicalLedger,
    build_objective_physical_ledger,
    build_scope_waterline,
)
from tilefoundry_ascend.ascend.schedule.regime_solver import (
    SOLVER_MODE_ALGEBRAIC,
    SOLVER_MODE_Z3,
    RegimePiecewiseSolver,
    RegimePlanResult,
    normalize_solver_mode,
    recompute_regime_for_live_buffers,
)
from tilefoundry_ascend.ascend.sil import DEFAULT_OPS_ROOT, TopologyAxes
from tilefoundry_ascend.ascend.sil.pytorch_frontend import (
    PyTorchSemanticAnalyzer,
    RepresentativeCase,
)
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MLA_DIR = _REPO_ROOT / "examples" / "ascend950pr_ops" / "mla"
_AR_DIR = _REPO_ROOT / "examples" / "ascend950pr_ops" / "attention_residual"


def _write_modelless_task_repo(root: Path) -> Path:
    tasks_root = root / "tasks"
    (tasks_root / "metadata").mkdir(parents=True)
    (tasks_root / "metadata" / "950pr.json").write_text(
        json.dumps({"level1": {}, "level2": {}, "level3": {}}),
        encoding="utf-8",
    )
    (tasks_root / "suite_manifest.json").write_text(
        json.dumps(
            {
                "operators": [
                    {"id": "exp", "level": "L1", "path": "tasks/level1/exp"},
                    {"id": "softmax", "level": "L2", "path": "tasks/level2/softmax"},
                    {"id": "matmul_op", "level": "L3", "path": "tasks/level3/matmul_op"},
                ]
            }
        ),
        encoding="utf-8",
    )

    exp_dir = tasks_root / "level1" / "exp"
    exp_dir.mkdir(parents=True)
    (exp_dir / "golden.py").write_text(
        dedent(
            """
            import torch

            def exp(x):
                return torch.exp(x)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (exp_dir / "proto.yaml").write_text(
        dedent(
            """
            operator:
              name: exp
              inputs:
                - name: x
                  dtype: [bfloat16]
              outputs:
                - name: y
                  dtype: [bfloat16]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (exp_dir / "cases.yaml").write_text(
        dedent(
            """
            cases:
              - case_id: 1
                input_shape: [[64, 1024]]
                dtype: [bfloat16]
              - case_id: 2
                input_shape: [[256, 4096]]
                dtype: [bfloat16]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    sm_dir = tasks_root / "level2" / "softmax"
    sm_dir.mkdir(parents=True)
    (sm_dir / "golden.py").write_text(
        dedent(
            """
            import torch

            def softmax(x):
                return torch.softmax(x, dim=-1)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (sm_dir / "proto.yaml").write_text(
        dedent(
            """
            operator:
              name: softmax
              inputs:
                - name: x
                  dtype: [bfloat16]
              outputs:
                - name: y
                  dtype: [bfloat16]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (sm_dir / "cases.yaml").write_text(
        dedent(
            """
            cases:
              - case_id: 1
                input_shape: [[128, 1024]]
                dtype: [bfloat16]
              - case_id: 2
                input_shape: [[8192, 8192]]
                dtype: [bfloat16]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    mm_dir = tasks_root / "level3" / "matmul_op"
    mm_dir.mkdir(parents=True)
    (mm_dir / "golden.py").write_text(
        dedent(
            """
            import torch

            def matmul_op(x, w):
                return torch.matmul(x, w)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (mm_dir / "proto.yaml").write_text(
        dedent(
            """
            operator:
              name: matmul_op
              inputs:
                - name: x
                  dtype: [float16]
                - name: w
                  dtype: [float16]
              outputs:
                - name: y
                  dtype: [float16]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (mm_dir / "cases.yaml").write_text(
        dedent(
            """
            cases:
              - case_id: 1
                input_shape: [[64, 128], [128, 256]]
                dtype: [float16, float16]
              - case_id: 2
                input_shape: [[256, 512], [512, 1024]]
                dtype: [float16, float16]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return tasks_root


def test_modelless_semantic_first_blueprints_softmax_parity_and_backtest(
    tmp_path: Path,
) -> None:
    """Verify golden-only workflows without model.py across blueprint, parity, and backtest."""
    tasks_root = _write_modelless_task_repo(tmp_path)
    assert not list(tasks_root.rglob("model.py"))
    assert not list(tasks_root.rglob("scheduled.py"))

    analyzer = PyTorchSemanticAnalyzer()
    sm_source = (tasks_root / "level2" / "softmax" / "golden.py").read_text(encoding="utf-8")
    sm_semantics = analyzer.analyze_source(sm_source, entry_name="softmax")
    sm_case = RepresentativeCase(
        case_id=2,
        input_shapes=((8192, 8192),),
        input_dtypes=("bfloat16",),
        output_dtype="bfloat16",
    )
    sm_packet = analyzer.build_blueprint(
        sm_semantics,
        {"M": 8192, "D": 8192, "total_elements": 8192 * 8192},
        representative_case=sm_case,
    )
    assert sm_packet.roofline.min_gm_bytes == 268_435_456
    assert sm_packet.provenance == BlueprintProvenance(
        semantic_source="pytorch_golden",
        traffic_basis="declared_io_exact",
        coordinate_scope="representative_anchor",
        has_authored_hir=False,
        has_scheduled_hir=False,
    )
    assert sm_packet.schedule_delta is None
    assert sm_packet.z3_pipeline_summary == {}
    assert "traffic_basis" not in sm_packet.z3_pipeline_summary
    rebuilt_sm = HirBlueprintPacket.from_dict(json.loads(json.dumps(sm_packet.to_dict())))
    assert rebuilt_sm == sm_packet

    e2e_pkt = _compile_blueprint_safe(
        sm_source,
        entry_name="softmax",
        representative_case=sm_case,
    )
    assert e2e_pkt is not None
    assert e2e_pkt.roofline.min_gm_bytes == 268_435_456
    assert e2e_pkt.provenance is not None
    assert e2e_pkt.provenance.semantic_source == "pytorch_golden"
    assert e2e_pkt.z3_pipeline_summary == {}
    summary = _summarize_blueprint(e2e_pkt)
    assert summary["provenance"] == e2e_pkt.provenance.to_dict()
    assert summary["schedule_delta"] is None

    assert agent_verify_parity is verify_reference_cpu_parity
    assert orchestrator_verify_parity is verify_reference_cpu_parity
    ok, msg = verify_reference_cpu_parity(
        lambda x: torch.softmax(x.to(torch.float32), dim=-1).to(x.dtype),
        golden_source=sm_source,
        entry_name="softmax",
        input_specs=[((4, 32), "bfloat16")],
        dual_profile=True,
    )
    assert ok is True, msg
    assert "reference_cpu_parity_passed" in msg
    assert "stress_profile=ok" in msg

    bad_ok, bad_msg = verify_reference_cpu_parity(
        lambda x: torch.softmax(x, dim=-1) + 0.1,
        golden_source=sm_source,
        entry_name="softmax",
        input_specs=[((4, 32), "float32")],
        dual_profile=True,
    )
    assert bad_ok is False
    assert "reference_cpu_parity_mismatch(contract)" in bad_msg

    stress_bad_ok, stress_bad_msg = verify_reference_cpu_parity(
        lambda x: torch.where(x.abs() > 1.2, torch.zeros_like(x), torch.exp(x)),
        golden_fn=lambda x: torch.exp(x),
        input_specs=[((8, 16), "float32")],
        dual_profile=True,
    )
    assert stress_bad_ok is False
    assert "reference_cpu_parity_mismatch" in stress_bad_msg

    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True)
    report_payload = {
        "operators": [
            {
                "rel_path": "level1/exp",
                "operator": "exp",
                "score": 85.0,
                "cases": [
                    {
                        "case_id": "level1/exp_1",
                        "elapsed_us": 4.0,
                        "t_hw_us": 0.0,
                        "baseline_perf_us": 8.0,
                        "speedup": 2.0,
                        "perf_score": 85.0,
                        "accuracy": {"passed": True},
                    },
                    {
                        "case_id": "level1/exp_2",
                        "elapsed_us": 20.0,
                        "t_hw_us": 0.0,
                        "baseline_perf_us": 40.0,
                        "speedup": 2.0,
                        "perf_score": 85.0,
                        "accuracy": {"passed": True},
                    },
                ],
            },
            {
                "rel_path": "level2/softmax",
                "operator": "softmax",
                "score": 80.0,
                "cases": [
                    {
                        "case_id": "level2/softmax_1",
                        "elapsed_us": 8.0,
                        "t_hw_us": 0.0,
                        "baseline_perf_us": 14.0,
                        "speedup": 1.75,
                        "perf_score": 80.0,
                        "accuracy": {"passed": True},
                    },
                    {
                        "case_id": "level2/softmax_2",
                        "elapsed_us": 240.0,
                        "t_hw_us": 0.0,
                        "baseline_perf_us": 420.0,
                        "speedup": 1.75,
                        "perf_score": 80.0,
                        "accuracy": {"passed": True},
                    },
                ],
            },
            {
                "rel_path": "level3/matmul_op",
                "operator": "matmul_op",
                "score": 90.0,
                "cases": [
                    {
                        "case_id": "level3/matmul_op_1",
                        "elapsed_us": 6.0,
                        "t_hw_us": 0.0,
                        "baseline_perf_us": 12.0,
                        "speedup": 2.0,
                        "perf_score": 90.0,
                        "accuracy": {"passed": True},
                    },
                    {
                        "case_id": "level3/matmul_op_2",
                        "elapsed_us": 32.0,
                        "t_hw_us": 0.0,
                        "baseline_perf_us": 64.0,
                        "speedup": 2.0,
                        "perf_score": 90.0,
                        "accuracy": {"passed": True},
                    },
                ],
            },
        ]
    }
    (report_dir / "official_final.json").write_text(
        json.dumps(report_payload),
        encoding="utf-8",
    )
    records, scanned, dedup = load_official_reports([report_dir], ops_root=tasks_root)
    assert scanned == 1
    assert dedup == 3
    assert len(records) == 6
    sm_case2 = next(r for r in records if r.case_id == "level2/softmax_2")
    assert sm_case2.features.gm_bytes == pytest.approx(268_435_456.0)
    assert sm_case2.t_hw_us > 100.0
    bt = run_cost_model_backtest([report_dir], ops_root=tasks_root)
    assert bt.unique_operator_cases == 6
    assert bt.p0_p1_loco_cv.mean_intra_group_spearman_rho == pytest.approx(1.0)


def test_authored_hir_enriches_schedule_delta_without_overwriting_anchor_roofline() -> None:
    """Supplying model.py and scheduled.py never overwrites real anchor Roofline dimensions."""
    mla_model_src = (_MLA_DIR / "model.py").read_text(encoding="utf-8")
    mla_sched_src = (_MLA_DIR / "scheduled.py").read_text(encoding="utf-8")
    analyzer = PyTorchSemanticAnalyzer()
    mla_golden = dedent(
        """
        import torch

        def mla(q, k, v):
            scores = torch.matmul(q, k.transpose(-2, -1))
            probs = torch.softmax(scores, dim=-1)
            return torch.matmul(probs, v)
        """
    )
    semantics = analyzer.analyze_source(mla_golden, entry_name="mla")
    real_anchor_dims = {
        "B": 4,
        "S_q": 256,
        "S_kv": 4096,
        "N_q": 32,
        "N_kv": 8,
        "D_qk": 576,
        "D_v": 512,
    }
    rep_case = RepresentativeCase(
        case_id=1,
        input_shapes=((4, 32, 256, 576), (4, 8, 4096, 576), (4, 8, 4096, 512)),
        input_dtypes=("bfloat16", "bfloat16", "bfloat16"),
        output_dtype="bfloat16",
    )

    golden_only = analyzer.build_universal_blueprint(
        semantics,
        real_anchor_dims,
        representative_case=rep_case,
    )
    enriched = analyzer.build_universal_blueprint(
        semantics,
        real_anchor_dims,
        authored_hir_source=mla_model_src,
        scheduled_hir_source=mla_sched_src,
        representative_case=rep_case,
    )

    assert enriched.roofline == golden_only.roofline
    assert enriched.roofline.anchor_dims["S_q"] == 256
    assert enriched.roofline.anchor_dims["S_kv"] == 4096
    assert enriched.roofline.ideal_roofline_us > 10.0
    assert enriched.provenance == BlueprintProvenance(
        semantic_source="pytorch_golden_plus_hir",
        traffic_basis="declared_io_exact",
        coordinate_scope="representative_anchor",
        has_authored_hir=True,
        has_scheduled_hir=True,
    )
    assert enriched.schedule_delta is not None
    assert enriched.schedule_delta.scheduled_ledger is not None
    assert enriched.reference_ledger == enriched.schedule_delta.scheduled_ledger
    assert enriched.z3_pipeline_summary == enriched.schedule_delta.z3_stage_summary
    assert enriched.z3_pipeline_summary
    assert "traffic_basis" not in enriched.z3_pipeline_summary
    assert "source" not in enriched.z3_pipeline_summary

    roundtrip = HirBlueprintPacket.from_dict(json.loads(json.dumps(enriched.to_dict())))
    assert roundtrip == enriched
    replaced = dataclasses.replace(enriched, fallback_reason="test_note")
    assert replaced.fallback_reason == "test_note"
    assert replaced.z3_pipeline_summary == enriched.z3_pipeline_summary

    ar_model_src = (_AR_DIR / "model.py").read_text(encoding="utf-8")
    ar_sched_src = (_AR_DIR / "scheduled.py").read_text(encoding="utf-8")
    ar_unaligned_dims = {
        "B": 2,
        "S_q": 16,
        "S_kv": 16,
        "N_q": 8,
        "N_kv": 8,
        "D": 1000,
    }
    ar_enriched = analyzer.build_universal_blueprint(
        semantics,
        ar_unaligned_dims,
        authored_hir_source=ar_model_src,
        scheduled_hir_source=ar_sched_src,
    )
    assert ar_enriched.roofline.anchor_dims["D_qk"] == 1000
    assert ar_enriched.schedule_delta is not None
    assert ar_enriched.schedule_delta.scheduled_ledger is not None
    assert any("hir_default_dims_fallback" in d for d in ar_enriched.lowering_diagnostics)


def test_hir_blueprint_packet_strips_legacy_non_z3_keys() -> None:
    """HirBlueprintPacket.z3_pipeline_summary never leaks non-Z3 metadata keys."""
    roofline = HirRooflineSummary(
        anchor_dims={"M": 64, "D": 128},
        compute_flops=1024,
        min_gm_bytes=4096,
        arithmetic_intensity=0.25,
        compute_floor_us=0.1,
        memory_floor_us=0.2,
        ideal_roofline_us=0.2,
        dominant_bound="memory",
    )
    legacy_pkt = HirBlueprintPacket(
        entry_name="demo",
        roofline=roofline,
        z3_pipeline_summary={
            "traffic_basis": "estimated",
            "source": "pytorch_semantic_frontend",
            "k_loop": {"ii": 8, "stages": {"load": 0}, "phases": {}, "versions": {}},
        },
        reference_ledger=None,
        lowering_diagnostics=(),
    )
    assert "traffic_basis" not in legacy_pkt.z3_pipeline_summary
    assert "source" not in legacy_pkt.z3_pipeline_summary
    assert legacy_pkt.z3_pipeline_summary == {
        "k_loop": {"ii": 8, "stages": {"load": 0}, "phases": {}, "versions": {}}
    }
    delta = BlueprintScheduleDelta(
        z3_stage_summary={
            "traffic_basis": "estimated",
            "main_loop": {"ii": 16, "stages": {}, "phases": {}, "versions": {}},
        },
        recommended_facets=("dma_and_cache_hierarchy", "pipeline_and_sync_graph"),
    )
    assert "traffic_basis" not in delta.z3_stage_summary
    assert BlueprintScheduleDelta.from_dict(delta.to_dict()) == delta


def test_blueprint_schedule_delta_validates_and_derives_canonical_7_facets() -> None:
    """Reject legacy facet names and emit canonical 7-facet names."""
    from tilefoundry_ascend.ascend.passes.hir_tir_bridge import (  # noqa: PLC0415
        _derive_recommended_facets,
    )
    from tilefoundry_ascend.ascend.sil.schema import VALID_FREEDOM_FACETS  # noqa: PLC0415

    with pytest.raises(ValueError, match="Unknown recommended_facets"):
        BlueprintScheduleDelta(recommended_facets=("sync_skeleton",))

    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=16,
        pipe_cycles={"AIV0.PIPE_V": 120},
        waterlines={},
        gm_roundtrip_bytes=65536,
        unaligned_dma_padding_bytes=64,
        dominant_contig_bytes=4096,
        in_loop_pipe_all_barriers=1,
        has_online_rescale_loop=True,
        ub_bank_conflict_risk=0.25,
    )
    derived = _derive_recommended_facets(ledger, {"main_loop": {"ii": 16}})
    assert derived
    assert all(facet in VALID_FREEDOM_FACETS for facet in derived)
    assert "alg_compute_and_monoid" in derived
    assert "dma_and_cache_hierarchy" in derived
    assert "sram_layout_and_aliasing" in derived
    assert "pipeline_and_sync_graph" in derived
    assert "tile_and_stage_knobs" in derived
    delta = BlueprintScheduleDelta(recommended_facets=derived)
    assert delta.recommended_facets == derived


def test_objective_physical_ledger_gm_traffic_scope_and_total_gm_bytes() -> None:
    """ObjectivePhysicalLedger distinguishes per_tile vs full_grid GM traffic explicitly."""
    per_tile = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=16,
        pipe_cycles={"AIV0.PIPE_MTE2": 64},
        waterlines={},
        grid_blocks=28,
        active_grid_blocks=14,
        gm_roundtrip_bytes=65536,
        gm_traffic_scope="per_tile",
    )
    assert per_tile.gm_traffic_scope == "per_tile"
    assert per_tile.total_gm_bytes() == 65536 * 14
    assert per_tile.total_gm_bytes(28) == 65536 * 28
    assert ObjectivePhysicalLedger.from_dict(per_tile.to_dict()) == per_tile

    full_grid = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=16,
        pipe_cycles={"AIV0.PIPE_MTE2": 64},
        waterlines={},
        grid_blocks=28,
        gm_roundtrip_bytes=1_048_576,
        gm_traffic_scope="full_grid",
    )
    assert full_grid.gm_traffic_scope == "full_grid"
    assert full_grid.total_gm_bytes() == 1_048_576
    assert full_grid.total_gm_bytes(56) == 1_048_576
    assert ObjectivePhysicalLedger.from_dict(full_grid.to_dict()) == full_grid

    legacy_dict = full_grid.to_dict()
    legacy_dict.pop("gm_traffic_scope")
    rebuilt_legacy = ObjectivePhysicalLedger.from_dict(legacy_dict)
    assert rebuilt_legacy.gm_traffic_scope == "per_tile"
    assert rebuilt_legacy.total_gm_bytes() == 1_048_576 * 28

    with pytest.raises(ValueError, match="gm_traffic_scope"):
        ObjectivePhysicalLedger.from_dict(
            dict(full_grid.to_dict(), gm_traffic_scope="invalid_scope")
        )


def test_regime_solver_algebraic_mode_roofline_twopass_and_monotone_caliper() -> None:
    """Verify algebraic_piecewise mode, Roofline UB twopass algebra, and <=3-step caliper."""
    assert SOLVER_MODE_ALGEBRAIC == "algebraic_piecewise"
    assert SOLVER_MODE_Z3 == "z3_smt"
    assert normalize_solver_mode("analytical_piecewise") == "algebraic_piecewise"
    assert normalize_solver_mode("algebraic_piecewise") == "algebraic_piecewise"

    target = AscendTarget("ascend.950pr")
    facts = target.get_facts(AscendSpaceFacts)
    assert isinstance(facts, AscendSpaceFacts)
    solver = RegimePiecewiseSolver(target)
    aiv_axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")

    wide_roofline = HirRooflineSummary(
        anchor_dims={"M": 64, "D": 65536, "elements": 64 * 65536},
        compute_flops=64 * 65536 * 11,
        min_gm_bytes=64 * 65536 * 4,
        arithmetic_intensity=2.75,
        compute_floor_us=1.0,
        memory_floor_us=10.0,
        ideal_roofline_us=10.0,
        dominant_bound="memory",
    )
    wide_plan = solver.solve_regime(
        {"regime_tag": "opaque_regime_0", "representative_dims": {"M": 64}},
        op_name="anon_reduce",
        topology_axes=aiv_axes,
        roofline=wide_roofline,
        input_dtypes=("bfloat16",),
    )
    assert wide_plan.solver_mode == SOLVER_MODE_ALGEBRAIC
    assert wide_plan.caliper_steps == 0
    assert wide_plan.caliper_action == "none"
    assert wide_plan.policy["requires_twopass_reduction"] is True
    ub_tile_elems = wide_plan.policy["ub_tile_elems"]
    twopass_switch = wide_plan.policy["twopass_switch_elems"]
    assert isinstance(ub_tile_elems, int) and isinstance(twopass_switch, int)
    assert 64 <= ub_tile_elems < 65536
    assert ub_tile_elems & (ub_tile_elems - 1) == 0
    assert twopass_switch <= facts.ub_simdvf_usable_bytes // 8
    assert wide_plan.policy["tile_mnk"][1] == ub_tile_elems

    legacy_dict = dict(wide_plan.to_summary_dict(), solver_mode="analytical_piecewise")
    restored = RegimePlanResult.from_dict(legacy_dict)
    assert restored.solver_mode == SOLVER_MODE_ALGEBRAIC
    assert restored.caliper_steps == 0
    assert restored.caliper_action == "none"

    cube_axes = TopologyAxes("coupled_1c2v", "l0c_ub_epilogue", "affine_simd")
    cube_base = solver.solve_regime(
        {"regime_tag": "cube_main", "representative_dims": {"M": 256, "N": 256, "K": 128}},
        op_name="anon_mm",
        topology_axes=cube_axes,
        uses_cube=True,
    )
    ub_overflow = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=False,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 120},
        waterlines={
            "UB": build_scope_waterline(
                scope="UB",
                capacity_bytes=facts.ub_simdvf_usable_bytes,
                explicit_buffers={"buf": int(facts.ub_simdvf_usable_bytes * 3.2)},
                implicit_buffers={},
            )
        },
    )
    ub_caliper = recompute_regime_for_live_buffers(solver, cube_base, ub_overflow)
    assert 1 <= ub_caliper.caliper_steps <= 3
    assert ub_caliper.caliper_action == "ub_halving"
    assert ub_caliper.policy["ub_estimated_bytes"] <= facts.ub_simdvf_usable_bytes
    assert ub_caliper.to_summary_dict()["caliper_steps"] == ub_caliper.caliper_steps
    assert ub_caliper.to_summary_dict()["caliper_action"] == "ub_halving"

    staged_policy = dict(cube_base.policy, l1_stages=3, l1_half_bank_split=True)
    staged_plan = dataclasses.replace(cube_base, policy=staged_policy)
    l1_overflow = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=False,
        steady_state_ii=None,
        pipe_cycles={"AIC.PIPE_M": 120},
        waterlines={
            "L1": build_scope_waterline(
                scope="L1",
                capacity_bytes=facts.l1_bytes,
                explicit_buffers={"a_l1": int(facts.l1_bytes * 1.8)},
                implicit_buffers={},
            )
        },
    )
    l1_caliper = recompute_regime_for_live_buffers(solver, staged_plan, l1_overflow)
    assert 1 <= l1_caliper.caliper_steps <= 3
    assert l1_caliper.caliper_action == "l1_stage_backoff"
    assert l1_caliper.policy["l1_stages"] < 3
    assert l1_caliper.policy["l1_estimated_bytes"] <= facts.l1_bytes


def test_verify_reference_cpu_parity_optional_hir_crosscheck() -> None:
    """Verify PyTorch golden parity can optionally cross-check HIR Twin when supplied."""
    ar_model_src = (_AR_DIR / "model.py").read_text(encoding="utf-8")

    def _ar_ref(
        v_stack: torch.Tensor,
        x: torch.Tensor,
        w: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
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
        return ((1.0 - 0.5) * xf + 0.5 * agg).to(v_stack.dtype)

    ok, msg = verify_reference_cpu_parity(
        _ar_ref,
        golden_fn=_ar_ref,
        input_specs=[
            ((4, 1, 4, 64), "bfloat16"),
            ((1, 4, 64), "bfloat16"),
            ((64,), "bfloat16"),
            ((64,), "bfloat16"),
        ],
        authored_hir_source=ar_model_src,
        test_dims={"N": 4, "B": 1, "S": 4, "D": 64},
        dual_profile=True,
    )
    assert ok is True, msg
    assert "hir_twin=ok" in msg


@pytest.mark.skipif(
    not DEFAULT_OPS_ROOT.is_dir(),
    reason="cann-bench tasks root unavailable",
)
def test_real_stock_operators_build_blueprints_without_model_py(tmp_path: Path) -> None:
    """Copying golden.py + proto.yaml + cases.yaml without model.py builds valid blueprints."""
    analyzer = PyTorchSemanticAnalyzer()
    selected = (
        ("level1", "swi_glu"),
        ("level2", "rms_norm"),
        ("level2", "softmax"),
        ("level3", "grouped_matmul"),
        ("level4", "gqa"),
    )
    for level, op_name in selected:
        src_dir = DEFAULT_OPS_ROOT / level / op_name
        dst_dir = tmp_path / level / op_name
        dst_dir.mkdir(parents=True)
        for fname in ("golden.py", "proto.yaml", "cases.yaml"):
            shutil.copy2(src_dir / fname, dst_dir / fname)
        assert not (dst_dir / "model.py").exists()
        golden_src = (dst_dir / "golden.py").read_text(encoding="utf-8")
        semantics = analyzer.analyze_source(golden_src, entry_name=op_name)
        assert semantics.entry_name == op_name
        dims = (
            {"B": 1, "S_q": 64, "S_kv": 128, "N_q": 8, "N_kv": 2, "D_qk": 128, "D_v": 128}
            if op_name == "gqa"
            else (
                {"M": 64, "N": 128, "K": 64}
                if op_name == "grouped_matmul"
                else {"M": 64, "D": 512, "total_elements": 64 * 512}
            )
        )
        pkt = analyzer.build_blueprint(semantics, dims)
        assert pkt.roofline.ideal_roofline_us > 0.0
        assert pkt.provenance is not None
        assert pkt.provenance.semantic_source == "pytorch_golden"
        assert pkt.provenance.has_authored_hir is False
        assert pkt.z3_pipeline_summary == {}
