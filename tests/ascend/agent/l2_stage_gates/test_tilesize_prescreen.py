"""Tests for zero-magic hard physics prescreen, lattice sweeper, and delta gating."""

from __future__ import annotations

from pathlib import Path

import pytest

from tilefoundry_ascend.agent.gates import gate_funnel as gf
from tilefoundry_ascend.agent.gates.gate_funnel import (
    StageAResult,
    prescreen_dsl_tile_kwargs,
)
from tilefoundry_ascend.agent.orchestrator import tilesize_prescreen as tp
from tilefoundry_ascend.agent.orchestrator.tilesize_prescreen import (
    LatticeSweepCandidate,
    PrescreenScore,
    TileConfigCandidate,
    count_dsl_gemm_calls,
    evaluate_tile_candidate,
    format_prescreen_summary,
    generate_aligned_neighborhood,
    inspect_builder_tile_params,
    prescreen_live_ir_neighborhood,
    prescreen_tile_configs,
    rank_stage_a_candidates_with_cost_model,
    sweep_executable_lattice_zero_llm,
)
from tilefoundry_ascend.ascend.cost_model.cost_gate_policy import (
    CostModelMode,
    evaluate_cost_gate_policy,
    is_paradigm_shift_exempt,
)
from tilefoundry_ascend.ascend.cost_model.hybrid_cost_model import (
    CostFeatures,
    CostPrediction,
    HybridAscendCostModel,
)
from tilefoundry_ascend.ascend.passes import build_objective_physical_ledger, build_scope_waterline
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget

_SYNTHETIC_VECTOR_SOURCE = """
import tilelang.ascend.language as T


def _select_vec_config(d_dim: int = 128) -> dict[str, int]:
    return {"tile_m": 4, "num_stages": 1, "d_dim": int(d_dim)}


def _build_vec_prim_func(
    d_dim: int = 128,
    tile_m: int = 4,
    num_stages: int = 1,
    **_extra,
):
    rows = max(1, int(tile_m))
    stages = max(1, int(num_stages))

    @T.prim_func
    def main(
        X: T.Tensor((2048, d_dim), "float16"),
        Y: T.Tensor((2048, d_dim), "float16"),
    ):
        with T.Kernel(56) as cid:
            x_ub = T.alloc_shared((rows, d_dim), "float16")
            y_ub = T.alloc_shared((rows, d_dim), "float16")
            for it in T.Pipelined(4, num_stages=stages):
                row_off = cid * rows + it * 56 * rows
                T.copy(X[row_off : row_off + rows, :], x_ub)
                with T.SimdVF():
                    for r, c in T.Parallel(rows, d_dim):
                        y_ub[r, c] = x_ub[r, c] + T.float16(1.0)
                T.copy(y_ub, Y[row_off : row_off + rows, :])

    return main
"""

_SYNTHETIC_CUBE_SOURCE = """
import tilelang.ascend.language as T


def _select_cube_config(m_dim: int = 256, n_dim: int = 256, k_dim: int = 128) -> dict[str, int]:
    return {
        "bm": 64,
        "bn": 64,
        "bk": 32,
        "num_stages": 1,
        "m_dim": int(m_dim),
        "n_dim": int(n_dim),
        "k_dim": int(k_dim),
    }


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


def _prediction(pred_us: float) -> CostPrediction:
    return CostPrediction(
        pred_us=pred_us,
        tier0_physics_us=pred_us,
        tier1_penalty_us=0.0,
        tier2_residual_factor=1.0,
        cube_floor_us=0.0,
        vec_floor_us=0.0,
        hbm_floor_us=pred_us,
        l2_hit_rate=0.0,
        wave_quantization_ratio=1.0,
        bank_conflict_us=0.0,
        effective_hbm_bw_bps=1.0,
        dominant_bound="hbm",
    )


class _StubCostModel:
    def __init__(self, predictions: list[float], ratios: list[float] | None = None) -> None:
        self.target = AscendTarget("ascend.950pr")
        self.facts = self.target.get_facts(AscendSpaceFacts)
        self._predictions = list(predictions)
        self._ratios = list(ratios or [])
        self.ratio_parents: list[CostFeatures] = []

    def predict(self, _features: CostFeatures) -> CostPrediction:
        return _prediction(self._predictions.pop(0))

    def predict_delta_ratio(self, _candidate: CostFeatures, parent: CostFeatures) -> float:
        self.ratio_parents.append(parent)
        return self._ratios.pop(0)


class _ForbidPredictCostModel:
    def __init__(self) -> None:
        self.target = AscendTarget("ascend.950pr")
        self.facts = self.target.get_facts(AscendSpaceFacts)

    def predict(self, _features: CostFeatures) -> CostPrediction:
        raise AssertionError("CostModel.predict must not run when cost_model_mode == 'off'")

    def predict_delta_ratio(self, _cand: CostFeatures, _parent: CostFeatures) -> float:
        raise AssertionError("CostModel.predict_delta_ratio must not run in 'off' mode")


def test_tilesize_prescreen_zero_magic_hard_physics_in_off_mode() -> None:
    """Verify hard feasibility checks, Roofline/tail-waste ranking, and zero CostModel calls."""
    target = AscendTarget("ascend.950pr")
    facts = target.get_facts(AscendSpaceFacts)

    zero_dim = evaluate_tile_candidate(TileConfigCandidate(bm=0, bn=64, bk=64), facts)
    assert zero_dim.is_feasible is False
    assert zero_dim.rejection_reason == "non_positive_tile_dimension"

    aligned = evaluate_tile_candidate(TileConfigCandidate(bm=48, bn=64, bk=64), facts)
    assert aligned.is_feasible is True
    bad_align = evaluate_tile_candidate(TileConfigCandidate(bm=15, bn=64, bk=64), facts)
    assert bad_align.is_feasible is False
    assert bad_align.rejection_reason == "unaligned_mmad_16x16x16_atom"

    stack_bad = evaluate_tile_candidate(
        TileConfigCandidate(bm=64, bn=64, bk=64, stack_bytes=128), facts
    )
    assert stack_bad.is_feasible is False
    assert stack_bad.rejection_reason is not None and "stack_overflow" in stack_bad.rejection_reason

    cap_bad = evaluate_tile_candidate(TileConfigCandidate(bm=256, bn=512, bk=128), facts)
    assert cap_bad.is_feasible is False
    assert cap_bad.rejection_reason is not None

    no_split = evaluate_tile_candidate(
        TileConfigCandidate(bm=128, bn=128, bk=64, stages=2, l1_half_bank_split=False),
        facts,
    )
    assert no_split.half_l1_penalty == 0.0

    fp16 = evaluate_tile_candidate(TileConfigCandidate(bm=64, bn=64, bk=64, dtype_bytes=2), facts)
    fp8 = evaluate_tile_candidate(
        TileConfigCandidate(bm=64, bn=64, bk=64, dtype_bytes=1, dtype="fp8e4m3"), facts
    )
    fp32 = evaluate_tile_candidate(TileConfigCandidate(bm=64, bn=64, bk=64, dtype_bytes=4), facts)
    assert fp8.cube_macs_per_cycle > fp16.cube_macs_per_cycle > fp32.cube_macs_per_cycle

    exact_fit = TileConfigCandidate(bm=64, bn=64, bk=64, stages=2)
    tail_waste = TileConfigCandidate(bm=112, bn=112, bk=64, stages=2)
    ranked = prescreen_tile_configs(
        [tail_waste, exact_fit, bad_align.candidate, cap_bad.candidate],
        target=target,
        m_dim=256,
        n_dim=256,
        k_dim=128,
        cost_model_mode="off",
    )
    assert len(ranked) == 2
    assert all(isinstance(r, PrescreenScore) for r in ranked)
    assert ranked[0].candidate == exact_fit
    assert ranked[0].tail_waste_ratio == pytest.approx(0.0)
    assert ranked[1].tail_waste_ratio > 0.0
    assert ranked[0].t_roof_us <= ranked[1].t_roof_us

    limited = prescreen_tile_configs([tail_waste, exact_fit], target=target, top_k=1)
    assert len(limited) == 1
    assert "Tile Prescreen Summary" in format_prescreen_summary(ranked)
    assert format_prescreen_summary([]) == "No feasible tile configurations found."

    neighbors = generate_aligned_neighborhood(
        TileConfigCandidate(bm=64, bn=64, bk=64, stages=2, aiv_split=2),
        m_max=96,
        n_max=96,
        k_max=96,
    )
    assert neighbors
    assert all(
        (c.bm // 16) % 2 == 0 and c.bm <= 96 and c.bn <= 96 and c.bk <= 96
        for c in neighbors
    )

    forbid_model = _ForbidPredictCostModel()
    summary_off = prescreen_live_ir_neighborhood(
        _SYNTHETIC_CUBE_SOURCE,
        factory_name="_build_cube_prim_func",
        center=TileConfigCandidate(bm=64, bn=64, bk=32, stages=1),
        base_kwargs={"m_dim": 256, "n_dim": 256, "k_dim": 128},
        dims={"M": 256, "N": 256, "K": 128},
        max_variants=2,
        cost_model=forbid_model,
        cost_model_mode="off",
    )
    assert "Live-IR Tile Prescreen Summary" in summary_off
    assert "pred_us=" not in summary_off


def test_lattice_sweeper_vector_cube_and_off_mode() -> None:
    """Verify Contract-D exports, Vector/Cube sweeps, patched selectors, and off-mode ranking."""
    assert "LatticeSweepCandidate" in tp.__all__
    assert "sweep_executable_lattice_zero_llm" in tp.__all__
    assert count_dsl_gemm_calls(_SYNTHETIC_CUBE_SOURCE) == 1
    assert count_dsl_gemm_calls(_SYNTHETIC_VECTOR_SOURCE) == 0
    params, has_varkw = inspect_builder_tile_params(_SYNTHETIC_CUBE_SOURCE, "_build_cube_prim_func")
    assert {"bm", "bn", "bk", "num_stages"} <= params and has_varkw is True

    target = AscendTarget("ascend.950pr")
    cost_model = HybridAscendCostModel(target)

    vec_results = sweep_executable_lattice_zero_llm(
        _SYNTHETIC_VECTOR_SOURCE,
        selector_fn_name="_select_vec_config",
        builder_fn_name="_build_vec_prim_func",
        sample_builder_kwargs={"d_dim": 128},
        dims={"D": 128},
        cost_model=cost_model,
        max_points=6,
        top_k=3,
    )
    assert 1 <= len(vec_results) <= 3
    assert all(isinstance(item, LatticeSweepCandidate) for item in vec_results)
    assert all(
        "tile_m" in item.knob_overrides and "num_stages" in item.knob_overrides
        for item in vec_results
    )
    assert vec_results == sorted(vec_results, key=lambda c: (c.pred_us, c.ub_bytes))

    ns_vec: dict[str, object] = {}
    exec(compile(vec_results[0].patched_source, "<vec_patched>", "exec"), ns_vec)
    sel_vec = ns_vec["_select_vec_config"]
    assert callable(sel_vec)
    patched_vec_cfg = sel_vec(d_dim=128)
    assert patched_vec_cfg["tile_m"] == vec_results[0].knob_overrides["tile_m"]
    assert patched_vec_cfg["num_stages"] == vec_results[0].knob_overrides["num_stages"]

    cube_results = sweep_executable_lattice_zero_llm(
        _SYNTHETIC_CUBE_SOURCE,
        selector_fn_name="_select_cube_config",
        builder_fn_name="_build_cube_prim_func",
        sample_builder_kwargs={"m_dim": 256, "n_dim": 256, "k_dim": 128},
        dims={"M": 256, "N": 256, "K": 128},
        cost_model=cost_model,
        max_points=6,
        top_k=3,
    )
    assert 1 <= len(cube_results) <= 3
    assert cube_results == sorted(cube_results, key=lambda c: (c.pred_us, c.ub_bytes))
    for item in cube_results:
        assert item.knob_overrides["bm"] % 16 == 0
        assert item.knob_overrides["bn"] % 16 == 0
        assert item.knob_overrides["bk"] % 16 == 0
        assert item.pred_us > 0.0
        assert item.l1_bytes > 0

    forbid_model = _ForbidPredictCostModel()
    off_results = sweep_executable_lattice_zero_llm(
        _SYNTHETIC_CUBE_SOURCE,
        selector_fn_name="_select_cube_config",
        builder_fn_name="_build_cube_prim_func",
        sample_builder_kwargs={"m_dim": 256, "n_dim": 256, "k_dim": 128},
        dims={"M": 256, "N": 256, "K": 128},
        cost_model=forbid_model,
        cost_model_mode="off",
        max_points=4,
        top_k=3,
    )
    assert len(off_results) >= 1
    assert all(item.pred_us == 0.0 for item in off_results)
    assert off_results == sorted(
        off_results, key=lambda c: (round(c.t_roof_us, 12), round(c.tail_waste_ratio, 12))
    )
    assert (
        sweep_executable_lattice_zero_llm(
            _SYNTHETIC_CUBE_SOURCE,
            selector_fn_name="_select_cube_config",
            builder_fn_name="_build_cube_prim_func",
            sample_builder_kwargs={},
            dims={},
            top_k=0,
        )
        == []
    )

    chosen_kw, chosen_sa = prescreen_dsl_tile_kwargs(
        _SYNTHETIC_VECTOR_SOURCE,
        factory_name="_build_vec_prim_func",
        candidate_kwargs_list=[
            {"d_dim": 128, "tile_m": 4, "num_stages": 1},
            {"d_dim": 128, "tile_m": 8, "num_stages": 1},
        ],
        target=target,
    )
    assert chosen_kw["d_dim"] == 128 and chosen_sa.passed is True
    with pytest.raises(ValueError, match="must not be empty"):
        prescreen_dsl_tile_kwargs(
            _SYNTHETIC_VECTOR_SOURCE, factory_name="_build_vec_prim_func", candidate_kwargs_list=[]
        )


def test_prescreen_delta_gating_parent_reuse_and_paradigm_exemption() -> None:
    """Verify parent-relative delta ranking, live-IR parent reuse, and cost gate policy."""
    ub_wl = build_scope_waterline(
        scope="UB",
        capacity_bytes=262144,
        explicit_buffers={"a": 1024},
        implicit_buffers={},
    )
    parent_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=4,
        pipe_cycles={"AIV0.PIPE_V": 32},
        waterlines={"UB": ub_wl},
    )
    shifted_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=4,
        pipe_cycles={"AIC.PIPE_M": 32, "AIV0.PIPE_V": 32},
        waterlines={"UB": ub_wl},
    )
    sa_a = StageAResult(passed=True, prim_func=None, ledger=parent_ledger)
    sa_b = StageAResult(passed=True, prim_func=None, ledger=shifted_ledger)

    model = _StubCostModel([100.0, 10.0, 20.0], [1.5, 0.8])
    ranked = rank_stage_a_candidates_with_cost_model(
        [("alpha", "", sa_a), ("beta", "", sa_b)],
        cost_model=model,
        parent_ledger=parent_ledger,
        creative_candidate_ids={"beta"},
    )
    assert [item.candidate_id for item in ranked] == ["beta", "alpha"]
    assert [item.pred_us for item in ranked] == [80.0, 150.0]
    assert ranked[0].coarse_reject_exempt is True
    assert ranked[1].coarse_reject_exempt is False
    assert len(model.ratio_parents) == 2
    assert model.ratio_parents[0] is model.ratio_parents[1]

    no_parent_model = _StubCostModel([7.0, 7.0, 3.0])
    no_parent_ranked = rank_stage_a_candidates_with_cost_model(
        [("first", "", sa_a), ("second", "", sa_a), ("third", "", sa_a)],
        cost_model=no_parent_model,
        creative_candidate_ids={"second"},
    )
    assert [item.candidate_id for item in no_parent_ranked] == ["third", "first", "second"]
    assert no_parent_ranked[2].coarse_reject_exempt is True
    assert no_parent_model.ratio_parents == []

    live_model = _StubCostModel([100.0, 100.0], [0.4])
    summary = prescreen_live_ir_neighborhood(
        _SYNTHETIC_CUBE_SOURCE,
        factory_name="_build_cube_prim_func",
        center=TileConfigCandidate(bm=64, bn=64, bk=32, stages=1),
        base_kwargs={"m_dim": 256, "n_dim": 256, "k_dim": 128},
        dims={"M": 256, "N": 256, "K": 128},
        max_variants=2,
        cost_model=live_model,
    )
    assert "pred_us=40.00us" in summary
    assert "pred_us=100.00us" in summary
    assert len(live_model.ratio_parents) == 1

    assert is_paradigm_shift_exempt(
        TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd"),
        TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd"),
        cand_workspace_stages=1,
        parent_workspace_stages=0,
    )
    assert not evaluate_cost_gate_policy(mode=CostModelMode.OFF, bucket_clean_samples=20).engaged
    assert not evaluate_cost_gate_policy(
        mode=CostModelMode.SHADOW, bucket_clean_samples=20, out_of_sample_spearman_rho=0.95
    ).engaged
    assert not evaluate_cost_gate_policy(
        mode=CostModelMode.ADVISORY, bucket_clean_samples=4, out_of_sample_spearman_rho=0.95
    ).engaged
    engaged = evaluate_cost_gate_policy(
        mode=CostModelMode.ADVISORY, bucket_clean_samples=12, out_of_sample_spearman_rho=0.85
    )
    assert engaged.engaged is True and engaged.allow_prompt_guidance is True
    assert gf is not None
