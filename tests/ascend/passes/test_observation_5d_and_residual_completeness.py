"""Completeness tests for 5D observation fields, residual stack_bytes, and 7-facet unlocks."""

from __future__ import annotations

import pytest

from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    StaticDynamicResidual,
    compiler_summary_from_residual,
    compute_static_dynamic_residual,
    extract_cpp_trackb_signature_tags,
    extract_structural_signature,
)
from tilefoundry_ascend.ascend.cost_model.bisheng_res_parser import BishengResourceSummary
from tilefoundry_ascend.ascend.passes.observation_space import (
    AxisResidual5D,
    ObservationVector5D,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger
from tilefoundry_ascend.ascend.sil.schema import (
    VALID_FREEDOM_FACETS,
    ConstraintRelaxationMask,
)


def test_5d_stage6_fixpipe_and_stack_bytes_roundtrip() -> None:
    """Roundtrip new 5D fields and stack_bytes through observation, residual, and summary."""
    ledger = build_objective_physical_ledger(
        analysis_mode="exact_modulo_z3",
        is_legal=True,
        steady_state_ii=2,
        pipe_cycles={"AIC.PIPE_M": 64, "AIC.PIPE_FIX": 48, "AIV0.PIPE_V": 32},
        waterlines={},
        ub_scalar_compute_loops=2,
        min_gm_dma_bytes=16,
    )
    summary = BishengResourceSummary(
        vec_regs=16,
        scalar_regs=8,
        spill_bytes=0,
        stack_bytes=512,
        vloopv2_loops=1,
        extra_signals={
            "scalar_compute_loop_fallback": 3.0,
            "dcache_bypass_narrow_store": 4.0,
        },
    )
    telemetry: dict[str, object] = {
        "dur_us": 20.0,
        "scalar_compute_loop_fallback": 1.5,
        "dcache_bypass_narrow_store": 1.0,
    }

    obs = ObservationVector5D.from_tiers(
        ledger,
        compiler_summary=summary,
        telemetry=telemetry,
    )
    assert obs.compute_issue.l0_ub_scalar_compute_loops == 2
    assert obs.compute_issue.l1_scalar_compute_loop_fallback == 3.0
    assert obs.offchip_dma.l0_min_gm_dma_bytes == 16
    assert obs.offchip_dma.l0_fixpipe_cycles == 48
    assert obs.offchip_dma.l1_dcache_bypass_narrow_store == 4.0
    assert obs.onchip_sram_reg.l1_stack_bytes == 512

    obs_restored = ObservationVector5D.from_dict(obs.to_dict())
    assert obs_restored == obs

    residuals = obs_restored.compute_axis_residuals()
    assert residuals.compute_issue_gap == 1.0
    assert residuals.offchip_dma_gap == 1.0
    assert residuals.onchip_sram_reg_gap == pytest.approx(0.5)

    residual = compute_static_dynamic_residual(
        ledger,
        telemetry,
        compiler_summary=summary,
    )
    assert residual.stack_bytes == 512
    assert residual.observation == obs

    res_restored = StaticDynamicResidual.from_dict(residual.to_dict())
    assert res_restored == residual
    assert res_restored.stack_bytes == 512

    recovered = compiler_summary_from_residual(res_restored)
    assert recovered is not None
    assert recovered.stack_bytes == 512
    assert recovered.extra_signals["scalar_compute_loop_fallback"] == 3.0
    assert recovered.extra_signals["dcache_bypass_narrow_store"] == 4.0


def test_narrow_store_and_stack_fold_into_axis_residuals() -> None:
    """Fold min_gm_dma_bytes < 32, ub_scalar_compute_loops, and stack_bytes into axis gaps."""
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 32, "AIV1.PIPE_V": 32},
        waterlines={},
        ub_scalar_compute_loops=1,
        min_gm_dma_bytes=16,
    )
    summary = BishengResourceSummary(vec_regs=8, stack_bytes=512, vloopv2_loops=1)
    obs = ObservationVector5D.from_tiers(ledger, compiler_summary=summary)
    assert obs.compute_issue.l0_ub_scalar_compute_loops == 1
    assert obs.compute_issue.l1_scalar_compute_loop_fallback == 1.0
    assert obs.offchip_dma.l0_min_gm_dma_bytes == 16
    assert obs.offchip_dma.l1_dcache_bypass_narrow_store == pytest.approx(2.0)

    gaps = obs.compute_axis_residuals()
    assert gaps.compute_issue_gap == pytest.approx(0.5)
    assert gaps.offchip_dma_gap == pytest.approx(0.5)
    assert gaps.onchip_sram_reg_gap == pytest.approx(0.5)


def test_unlock_causal_facets_covers_all_seven_valid_facets() -> None:
    """Verify AxisResidual5D.unlock_causal_facets can unlock all 7 VALID_FREEDOM_FACETS."""
    all_locked = ConstraintRelaxationMask(
        alg_compute_and_monoid=False,
        tile_and_stage_knobs=False,
        exec_mode_and_spr=False,
        sram_layout_and_aliasing=False,
        dma_and_cache_hierarchy=False,
        pipeline_and_sync_graph=False,
        grid_and_workspace_topology=False,
    )
    assert all_locked.unlocked_facets() == ()

    sram_knobs = AxisResidual5D(0.0, 0.0, 0.45, 0.0, 0.0).unlock_causal_facets(all_locked)
    assert sram_knobs.is_unlocked("tile_and_stage_knobs") is True
    assert sram_knobs.is_unlocked("sram_layout_and_aliasing") is True
    assert sram_knobs.is_unlocked("alg_compute_and_monoid") is False

    grid_knobs = AxisResidual5D(0.0, 0.0, 0.0, 0.0, 0.40).unlock_causal_facets(all_locked)
    assert grid_knobs.is_unlocked("tile_and_stage_knobs") is True
    assert grid_knobs.is_unlocked("grid_and_workspace_topology") is True

    compute_alg = AxisResidual5D(0.45, 0.0, 0.0, 0.0, 0.0).unlock_causal_facets(all_locked)
    assert compute_alg.is_unlocked("alg_compute_and_monoid") is True
    assert compute_alg.is_unlocked("exec_mode_and_spr") is True

    full = AxisResidual5D(1.0, 1.0, 1.0, 1.0, 1.0).unlock_causal_facets(all_locked)
    assert full == ConstraintRelaxationMask.top()
    assert full.unlocked_facets() == VALID_FREEDOM_FACETS


def test_extract_cpp_trackb_signature_tags_detects_ascend_gemm_l1_template() -> None:
    """Verify templated ascend_gemm_l1 and asc_mmad variants emit cpp:cube_mmad."""
    cpp_src = (
        "__aicore__ void kernel() {\n"
        "  ascend_gemm_l1<64, 64, 64, 32, true, half>(l0c, a_l1, b_l1);\n"
        "}\n"
    )
    tags = extract_cpp_trackb_signature_tags(cpp_src)
    assert "cpp:cube_mmad" in tags
    assert "cpp:cube_mmad" in extract_cpp_trackb_signature_tags("asc_mmad_mx(c, a, b);")
    assert "cpp:cube_mmad" in extract_cpp_trackb_signature_tags("asc_mmad_sparse(c, a, b);")

    py_mod = (
        '_TRACKB_SRC = r"""\n'
        "#include <tl_templates/ascend/common.h>\n"
        "__aicore__ void k() {\n"
        "  ascend_gemm_l1<64, 64, 64, 32, true, half>(c, a, b);\n"
        "}\n"
        '"""\n'
    )
    sig = extract_structural_signature(py_mod, track="manual_ssbuf")
    assert "cpp:cube_mmad" in sig
