"""Tests for ObjectivePhysicalLedger -> CostFeatures 4-signal bridge and prescreen wiring."""

from __future__ import annotations

import pytest

from tilefoundry_ascend.agent.orchestrator import gate_funnel as gf
from tilefoundry_ascend.agent.orchestrator import tilesize_prescreen as tp
from tilefoundry_ascend.ascend.cost_model import (
    HybridAscendCostModel,
    build_cost_features_from_ledger,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    ObjectivePhysicalLedger,
    build_objective_physical_ledger,
)
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes
from tilefoundry_ascend.ascend.target import AscendTarget


def _make_vector_ledger(
    *,
    grid_blocks: int = 56,
    active_grid_blocks: int | None = None,
    min_gm_dma_bytes: int = 4096,
    ub_scalar_compute_loops: int = 0,
    ub_bank_conflict_risk: float = 0.0,
    unaligned_dma_padding_bytes: int = 0,
) -> ObjectivePhysicalLedger:
    return build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=100,
        pipe_cycles={"AIV0.PIPE_V": 4000, "AIV0.PIPE_MTE2": 800},
        waterlines={},
        grid_blocks=grid_blocks,
        active_grid_blocks=active_grid_blocks,
        gm_roundtrip_bytes=65536,
        dominant_contig_bytes=32768,
        min_gm_dma_bytes=min_gm_dma_bytes,
        ub_scalar_compute_loops=ub_scalar_compute_loops,
        ub_bank_conflict_risk=ub_bank_conflict_risk,
        unaligned_dma_padding_bytes=unaligned_dma_padding_bytes,
    )


def _make_cube_ledger(
    *,
    grid_blocks: int = 28,
    active_grid_blocks: int | None = None,
    l1_bank_conflict_risk: float = 0.0,
) -> ObjectivePhysicalLedger:
    return build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=100,
        pipe_cycles={"AIC.PIPE_M": 4000, "AIV0.PIPE_V": 1000, "AIV0.PIPE_MTE2": 800},
        waterlines={},
        grid_blocks=grid_blocks,
        active_grid_blocks=active_grid_blocks,
        gm_roundtrip_bytes=65536,
        dominant_contig_bytes=32768,
        min_gm_dma_bytes=4096,
        l1_bank_conflict_risk=l1_bank_conflict_risk,
    )


def test_ledger_four_signals_increase_static_stage_a_predicted_latency() -> None:
    """Verify all L0 ledger risk signals raise deterministic Stage-A cost over clean baseline."""
    target = AscendTarget("ascend.950pr")
    model = HybridAscendCostModel(target)
    vec_axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")

    clean_ledger = _make_vector_ledger()
    clean_cf = build_cost_features_from_ledger(clean_ledger, target=target, axes=vec_axes)
    clean_pred = model.predict(clean_cf, bisheng_res=None)
    clean_pred_repeat = model.predict(clean_cf, bisheng_res=None)
    assert clean_pred.pred_us == clean_pred_repeat.pred_us
    assert "dcache_bypass_narrow_store" not in clean_cf.extra_signals
    assert clean_cf.ub_bank_conflict_ratio == 0.0
    assert clean_cf.has_unaligned_tail is False
    assert clean_cf.active_aiv_cores == 56

    narrow_cf = build_cost_features_from_ledger(
        _make_vector_ledger(min_gm_dma_bytes=8),
        target=target,
        axes=vec_axes,
    )
    narrow_pred = model.predict(narrow_cf, bisheng_res=None)
    assert narrow_cf.extra_signals.get("dcache_bypass_narrow_store") == pytest.approx(4.0)
    assert narrow_pred.pred_us == model.predict(narrow_cf, bisheng_res=None).pred_us
    assert narrow_pred.pred_us > clean_pred.pred_us

    scalar_cf = build_cost_features_from_ledger(
        _make_vector_ledger(ub_scalar_compute_loops=1),
        target=target,
        axes=vec_axes,
    )
    scalar_pred = model.predict(scalar_cf, bisheng_res=None)
    assert scalar_cf.extra_signals.get("scalar_compute_loop_fallback") == pytest.approx(1.0)
    assert scalar_pred.pred_us == model.predict(scalar_cf, bisheng_res=None).pred_us
    assert scalar_pred.pred_us > clean_pred.pred_us

    ub_conflict_cf = build_cost_features_from_ledger(
        _make_vector_ledger(ub_bank_conflict_risk=0.5),
        target=target,
        axes=vec_axes,
    )
    ub_conflict_pred = model.predict(ub_conflict_cf, bisheng_res=None)
    assert ub_conflict_cf.ub_bank_conflict_ratio == pytest.approx(0.5)
    assert ub_conflict_pred.bank_conflict_us > 0.0
    assert ub_conflict_pred.pred_us == model.predict(ub_conflict_cf, bisheng_res=None).pred_us
    assert ub_conflict_pred.pred_us > clean_pred.pred_us

    unaligned_cf = build_cost_features_from_ledger(
        _make_vector_ledger(unaligned_dma_padding_bytes=16),
        target=target,
        axes=vec_axes,
    )
    unaligned_pred = model.predict(unaligned_cf, bisheng_res=None)
    assert unaligned_cf.has_unaligned_tail is True
    assert unaligned_pred.tier1_penalty_us > clean_pred.tier1_penalty_us
    assert unaligned_pred.pred_us == model.predict(unaligned_cf, bisheng_res=None).pred_us
    assert unaligned_pred.pred_us > clean_pred.pred_us

    low_grid_cf = build_cost_features_from_ledger(
        _make_vector_ledger(grid_blocks=56, active_grid_blocks=4),
        target=target,
        axes=vec_axes,
    )
    low_grid_pred = model.predict(low_grid_cf, bisheng_res=None)
    assert low_grid_cf.active_aic_cores == 4
    assert low_grid_cf.active_aiv_cores == 4
    assert low_grid_pred.pred_us == model.predict(low_grid_cf, bisheng_res=None).pred_us
    assert low_grid_pred.pred_us > clean_pred.pred_us


def test_l1_bank_conflict_and_coupled_1c2v_active_cores_bridge() -> None:
    """Verify L1 bank conflict trigger and coupled_1c2v active core scaling."""
    target = AscendTarget("ascend.950pr")
    model = HybridAscendCostModel(target)
    coupled_axes = TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd")

    clean_cube_cf = build_cost_features_from_ledger(
        _make_cube_ledger(grid_blocks=28, active_grid_blocks=28),
        target=target,
        axes=coupled_axes,
    )
    clean_cube_pred = model.predict(clean_cube_cf, bisheng_res=None)
    assert clean_cube_cf.active_aic_cores == 28
    assert clean_cube_cf.active_aiv_cores == 56

    l1_conflict_cf = build_cost_features_from_ledger(
        _make_cube_ledger(l1_bank_conflict_risk=0.6),
        target=target,
        axes=coupled_axes,
    )
    l1_conflict_pred = model.predict(l1_conflict_cf, bisheng_res=None)
    assert l1_conflict_cf.l1_half_bank_split is False
    assert l1_conflict_cf.l1_stages >= 2
    assert l1_conflict_pred.bank_conflict_us > clean_cube_pred.bank_conflict_us
    assert l1_conflict_pred.pred_us > clean_cube_pred.pred_us

    explicit_split_cf = build_cost_features_from_ledger(
        _make_cube_ledger(l1_bank_conflict_risk=0.6),
        target=target,
        axes=coupled_axes,
        live_plan={"policy": {"l1_half_bank_split": True, "l1_stages": 2}},
    )
    assert explicit_split_cf.l1_half_bank_split is True
    assert model.predict(explicit_split_cf, bisheng_res=None).bank_conflict_us == pytest.approx(
        clean_cube_pred.bank_conflict_us
    )

    low_coupled_cf = build_cost_features_from_ledger(
        _make_cube_ledger(grid_blocks=28, active_grid_blocks=4),
        target=target,
        axes=coupled_axes,
    )
    low_coupled_pred = model.predict(low_coupled_cf, bisheng_res=None)
    assert low_coupled_cf.active_aic_cores == 4
    assert low_coupled_cf.active_aiv_cores == 8
    assert low_coupled_pred.pred_us > clean_cube_pred.pred_us

    override_cores_cf = build_cost_features_from_ledger(
        _make_cube_ledger(grid_blocks=28, active_grid_blocks=4),
        target=target,
        axes=coupled_axes,
        live_plan={"policy": {"active_aic_cores": 14, "active_aiv_cores": 28}},
    )
    assert override_cores_cf.active_aic_cores == 14
    assert override_cores_cf.active_aiv_cores == 28


def test_prescreen_live_ir_neighborhood_forwards_ordered_tensor_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify prescreen_live_ir_neighborhood forwards ordered_tensor_shapes to Stage A."""
    seen_shapes: list[object] = []

    def fake_stage_a(*_args: object, **kwargs: object) -> gf.StageAResult:
        seen_shapes.append(kwargs.get("ordered_tensor_shapes"))
        return gf.StageAResult(passed=True, prim_func=None, ledger=_make_vector_ledger())

    monkeypatch.setattr(gf, "run_stage_a_static", fake_stage_a)
    shapes = [[16, 128], [16, 128]]
    dsl = "def build_kernel(bm=64, bn=64, bk=64, stages=2):\n    return None\n"
    summary = tp.prescreen_live_ir_neighborhood(
        dsl,
        factory_name="build_kernel",
        center=tp.TileConfigCandidate(bm=64, bn=64, bk=64, stages=2),
        max_variants=2,
        ordered_tensor_shapes=shapes,
    )
    assert "Live-IR Tile Prescreen Summary:" in summary
    assert seen_shapes == [shapes, shapes]
