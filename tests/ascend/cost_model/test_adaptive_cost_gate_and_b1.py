"""Tests for adaptive cost gate policy, B1 scalar penalty de-tox, and symptom ranking."""

from __future__ import annotations

import math
from textwrap import dedent

import pytest

from tilefoundry_ascend.agent.kb import SymptomFamily, rank_symptom_families
from tilefoundry_ascend.agent.orchestrator.gate_funnel import StageAResult
from tilefoundry_ascend.agent.orchestrator.tilesize_prescreen import (
    rank_stage_a_candidates_with_cost_model,
)
from tilefoundry_ascend.ascend.cost_model import (
    BishengResourceSummary,
    CostFeatures,
    CostGateDecision,
    CostModelMode,
    HybridAscendCostModel,
    MicroarchProfile,
    count_reachable_ub_scalar_compute_loops,
    evaluate_cost_gate_policy,
    is_paradigm_shift_exempt,
    parse_bisheng_res_usage,
)
from tilefoundry_ascend.ascend.cost_model.bisheng_res_parser import (
    _penalty_scalar_compute_loop_fallback,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes
from tilefoundry_ascend.ascend.target import AscendTarget


def test_evaluate_cost_gate_policy_state_transitions() -> None:
    """Verify CostGateDecision V5.4 modes (off, shadow, advisory) and t-stat gating."""
    default_off = evaluate_cost_gate_policy()
    assert isinstance(default_off, CostGateDecision)
    assert default_off.mode == CostModelMode.OFF
    assert default_off.engaged is False

    shadow = evaluate_cost_gate_policy(
        mode=CostModelMode.SHADOW,
        bucket_clean_samples=30,
        out_of_sample_spearman_rho=0.65,
    )
    assert shadow.mode == CostModelMode.SHADOW
    assert shadow.engaged is False
    assert shadow.allow_prompt_guidance is False
    assert shadow.allow_candidate_pruning is False
    assert shadow.reason == "shadow_observation_only"

    insufficient = evaluate_cost_gate_policy(
        mode=CostModelMode.ADVISORY,
        bucket_clean_samples=2,
        out_of_sample_spearman_rho=0.8,
    )
    assert insufficient.mode == CostModelMode.ADVISORY
    assert insufficient.engaged is False
    assert insufficient.reason == "insufficient_bucket_samples"

    advisory_ok = evaluate_cost_gate_policy(
        mode=CostModelMode.ADVISORY,
        bucket_clean_samples=30,
        out_of_sample_spearman_rho=0.65,
    )
    assert advisory_ok.mode == CostModelMode.ADVISORY
    assert advisory_ok.engaged is True
    assert advisory_ok.allow_prompt_guidance is True
    assert advisory_ok.allow_candidate_pruning is False
    assert advisory_ok.reason == "advisory_t_stat_validated"

    disabled_flag = evaluate_cost_gate_policy(enable_prescreen_flag=False, mode="advisory")
    assert disabled_flag.mode == CostModelMode.OFF
    assert disabled_flag.engaged is False
    assert disabled_flag.reason == "disabled_by_config"

    disabled_mode = evaluate_cost_gate_policy(mode="off")
    assert disabled_mode.mode == CostModelMode.OFF
    assert disabled_mode.engaged is False
    assert disabled_mode.reason == "disabled_by_config"


def test_is_paradigm_shift_exempt_and_prescreen_ratio() -> None:
    """Exempt creative, workspace-stage, and cross-topology candidates from pruning."""
    axes_a = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
    axes_b = TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd")

    assert is_paradigm_shift_exempt(axes_a, axes_a, is_creative=False) is False
    assert is_paradigm_shift_exempt(axes_a, axes_a, is_creative=True) is True
    assert (
        is_paradigm_shift_exempt(
            axes_a,
            axes_a,
            cand_workspace_stages=2,
            parent_workspace_stages=0,
        )
        is True
    )
    assert is_paradigm_shift_exempt(axes_a, axes_b) is True

    model = HybridAscendCostModel(AscendTarget("ascend.950pr"))
    parent_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=100,
        pipe_cycles={"AIC.PIPE_M": 400},
        gm_roundtrip_bytes=16384,
        grid_blocks=28,
        waterlines={},
    )
    slow_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=2000,
        pipe_cycles={"AIC.PIPE_M": 20000},
        gm_roundtrip_bytes=1_048_576,
        grid_blocks=28,
        waterlines={},
    )
    ranked = rank_stage_a_candidates_with_cost_model(
        [
            ("exploit_slow", "", StageAResult(passed=True, prim_func=None, ledger=slow_ledger)),
            ("creative_slow", "", StageAResult(passed=True, prim_func=None, ledger=slow_ledger)),
        ],
        cost_model=model,
        axes=axes_b,
        parent_ledger=parent_ledger,
        creative_candidate_ids={"creative_slow"},
    )
    by_id = {item.candidate_id: item for item in ranked}
    assert by_id["exploit_slow"].coarse_reject_exempt is False
    assert by_id["creative_slow"].coarse_reject_exempt is True


def test_outer_gm_metadata_loop_decoupled_and_no_cost_avalanche() -> None:
    """Ignore outer __gm__ group metadata loops while catching true __ubuf__ scalar loops."""
    mixed_with_group_meta = dedent(
        """
        #include <tl_templates/ascend/common.h>
        extern "C" __global__ __aicore__ void grouped_kernel(
            __gm__ half* a, __gm__ half* b, __gm__ float* out, __gm__ int32_t* group_list, int num_groups
        ) {
            __ubuf__ float* ub_zero = (__ubuf__ float*)0;
            for (int z = 0; z < 8; ++z) {
                ub_zero[z] = 0.0f;
            }
            int m_offset = 0;
            for (int g = 0; g < num_groups; ++g) {
                int m_g = group_list[g];
                int k_blocks = (m_g + 15) / 16;
                for (int kb = 0; kb < k_blocks; ++kb) {
                    asc_mmad(a + m_offset, b, out + m_offset, 16, 16, 16,
                             asc_unit_flag_mode::DISABLE, true, false, true);
                }
                simd_inst::vadd(ub_zero, ub_zero, ub_zero);
                m_offset += m_g;
            }
        }
        extern "C" int launch(void* stream, void* a, void* b, void* out, void* g_list, int ng) {
            grouped_kernel<<<28, nullptr, stream>>>(
                (__gm__ half*)a, (__gm__ half*)b, (__gm__ float*)out, (__gm__ int32_t*)g_list, ng
            );
            return 0;
        }
        """
    )
    assert count_reachable_ub_scalar_compute_loops(mixed_with_group_meta) == 0
    meta_summary = parse_bisheng_res_usage(asm_or_dump_text=mixed_with_group_meta)
    assert meta_summary.extra_signals.get("scalar_compute_loop_fallback", 0.0) == 0.0

    true_ub_scalar = dedent(
        """
        #include <tl_templates/ascend/common.h>
        extern "C" __global__ __vector__ void ub_scalar_kernel(__gm__ float* x, __gm__ float* y) {
            __ubuf__ float* ub_x = (__ubuf__ float*)0;
            asc_copy_gm2ub_align(ub_x, x, 256);
            for (int i = 0; i < 64; ++i) {
                ub_x[i] = ub_x[i] * 0.5f + 1.0f;
            }
            asc_copy_ub2gm_align(y, ub_x, 256);
        }
        extern "C" int launch(void* stream, void* x, void* y) {
            ub_scalar_kernel<<<28, nullptr, stream>>>((__gm__ float*)x, (__gm__ float*)y);
            return 0;
        }
        """
    )
    assert count_reachable_ub_scalar_compute_loops(true_ub_scalar) == 1

    model = HybridAscendCostModel(AscendTarget("ascend.950pr"))
    axes = TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd")
    no_meta_src = dedent(
        """
        #include <tl_templates/ascend/common.h>
        extern "C" __global__ __aicore__ void clean_kernel(
            __gm__ half* a, __gm__ half* b, __gm__ float* out
        ) {
            __ubuf__ float* ub_zero = (__ubuf__ float*)0;
            asc_mmad(a, b, out, 16, 16, 16, asc_unit_flag_mode::DISABLE, true, false, true);
            simd_inst::vadd(ub_zero, ub_zero, ub_zero);
        }
        extern "C" int launch(void* stream, void* a, void* b, void* out) {
            clean_kernel<<<28, nullptr, stream>>>(
                (__gm__ half*)a, (__gm__ half*)b, (__gm__ float*)out
            );
            return 0;
        }
        """
    )
    clean_summary = parse_bisheng_res_usage(asm_or_dump_text=no_meta_src)
    prof = MicroarchProfile(
        cube_mmad_count=16,
        cube_macs_total=16 * 128 * 128 * 64,
        fp_alu_ops=256,
        gm_to_l1_bytes=65536,
        l0c_to_gm_bytes=65536,
        pipeline_depth=2,
        inner_loop_trips=8,
        source_kind="cce",
    ).to_dict()
    base_cf = CostFeatures(
        cube_flops=2.0e12,
        vec_flops=2.0e9,
        gm_bytes=32.0e6,
        total_tiles=28,
        contig_burst_bytes=32768,
        topology_axes=axes,
        microarch_profile=prof,
    )
    base_pred = model.predict(base_cf, bisheng_res=clean_summary)
    meta_pred = model.predict(base_cf, bisheng_res=meta_summary)
    meta_rel_diff = abs(meta_pred.pred_us - base_pred.pred_us) / max(1e-6, base_pred.pred_us)
    assert meta_rel_diff < 0.20

    spurious_scalar_cf = CostFeatures(
        cube_flops=2.0e12,
        vec_flops=2.0e9,
        gm_bytes=32.0e6,
        total_tiles=28,
        contig_burst_bytes=32768,
        topology_axes=axes,
        microarch_profile=prof,
        extra_signals={"scalar_compute_loop_fallback": 2.0},
    )
    spurious_pred = model.predict(spurious_scalar_cf)
    rel_diff = abs(spurious_pred.pred_us - base_pred.pred_us) / max(1e-6, base_pred.pred_us)
    assert rel_diff < 0.20

    bounded = _penalty_scalar_compute_loop_fallback(
        BishengResourceSummary(extra_signals={"scalar_compute_loop_fallback": 4.0}),
        {"base_ref": 20.0, "vec_floor_us": 5.0},
    )
    assert bounded == pytest.approx(7.0)
    assert bounded < 10.0


def test_rank_symptom_families_caps_scalar_trap_and_drops_tie_privilege() -> None:
    """Cap SCALAR_LOOP_TRAP when vector/cube work runs and break ties alphabetically."""
    ranked = rank_symptom_families(
        {
            "measured_aiv_vec_ratio": 0.25,
            "measured_aic_mac_ratio": 0.0,
            "scalar_compute_loop_fallback": 2.0,
            "measured_mte2_wait_ratio": 0.70,
            "l2_hit_rate": 0.20,
        }
    )
    ranked_map = dict(ranked)
    assert ranked_map[SymptomFamily.SCALAR_LOOP_TRAP] == pytest.approx(0.30 / 1.10, rel=1e-4)
    assert sum(ranked_map.values()) == pytest.approx(1.0, abs=1e-6)
    assert ranked[0][0] == SymptomFamily.DMA_L2_THRASH_REUSE_MISS

    static_ranked = dict(
        rank_symptom_families(
            {
                "l0_cube_cycles": 1000,
                "l0_vec_cycles": 200,
                "scalar_compute_loop_fallback": 3.0,
            }
        )
    )
    assert static_ranked[SymptomFamily.SCALAR_LOOP_TRAP] <= 0.55

    tie_ranked = rank_symptom_families(
        {
            "measured_aiv_vec_ratio": 0.0,
            "measured_aic_mac_ratio": 0.0,
            "l0_cube_cycles": 0,
            "l0_vec_cycles": 0,
            "transfer_inflation_ratio": 3.5,
        }
    )
    tie_scores = dict(tie_ranked)
    assert sum(tie_scores.values()) == pytest.approx(1.0, abs=1e-6)
    assert tie_scores[SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP] == pytest.approx(0.425, rel=1e-4)
    assert tie_scores[SymptomFamily.SCALAR_LOOP_TRAP] == pytest.approx(0.425, rel=1e-4)
    assert tie_ranked[0][0] == SymptomFamily.SCALAR_LOOP_TRAP
