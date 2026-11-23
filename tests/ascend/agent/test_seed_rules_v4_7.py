"""Tests for v4.8 MECE seed rules (R01..R41), UB scalar loop detection, and L2 PMU gating."""

from __future__ import annotations

from textwrap import dedent

import pytest

from tilefoundry_ascend.agent.kb import (
    AscendCausalRule,
    SymptomFamily,
    build_predicate_context,
    load_seed_rules,
    validate_rule_predicate,
)
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    StaticDynamicResidual,
    compute_static_dynamic_residual,
)
from tilefoundry_ascend.ascend.cost_model import (
    BishengResourceSummary,
    count_reachable_ub_scalar_compute_loops,
    detect_scalar_compute_loop_fallback,
    parse_bisheng_res_usage,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    ObjectivePhysicalLedger,
    build_objective_physical_ledger,
    build_scope_waterline,
)
from tilefoundry_ascend.ascend.sil.schema import CasePhysicsRatios


def test_seed_rules_exact_41_mece_provenance_and_ast_validity() -> None:
    """Verify 41 atomic rules R01..R41, CE over 13 families, unique mechanism tags, and ASTs."""
    rules = load_seed_rules()
    assert len(rules) == 41
    rule_ids = [r.rule_id for r in rules]
    assert len(set(rule_ids)) == 41
    for idx, rule_id in enumerate(rule_ids, start=1):
        assert rule_id.startswith(f"R{idx:02d}_")

    by_id = {r.rule_id: r for r in rules}
    assert "R17_RMEM_ACCUMULATOR_HOIST" in by_id
    assert "R26_UB_256KB_SPR_UNLOCK" in by_id
    assert "R29_W4_AIV_SUBBYTE_UNPACK" in by_id
    assert "R32_GROUPED_ROLLING_CORE_DISPATCH" in by_id
    assert "R34_SCALAR_LOOP_TO_VECTOR_DMA_PIPELINE" in by_id
    assert "R38_COALESCE_SUB32B_TAIL_DMA_STORE" in by_id
    assert "R39_FIXPIPE_DEQ_SCALE_FOLD" in by_id
    assert "R40_SIMTVF_FP32_WIDEN_BEFORE_SUBBYTE_QUANT" in by_id
    assert "R41_EMPTY_GROUP_ZERO_TOKEN_SKIP" in by_id

    r02 = by_id["R02_ONCHIP_L0C_TO_UB_DIRECT"]
    assert "asc_copy_l0c2ub" in r02.cpp20_transform_template
    assert "asc_set_l0c_copy_nz_para" in r02.cpp20_transform_template

    covered_families = {SymptomFamily(r.symptom_family) for r in rules}
    assert covered_families == set(SymptomFamily)
    mechanism_tags = {r.mechanism_tag for r in rules}
    assert len(mechanism_tags) == 41
    assert all(tag.strip() for tag in mechanism_tags)
    assert all(r.concept_description.strip() for r in rules)

    default_rule = AscendCausalRule(
        rule_id="R_CUSTOM_TEST",
        regime_guard="is_legal",
        precondition="PIPE_V > 0",
        transform_template="Optimize vector pass.",
        expected_postcondition="PIPE_V <= prev.PIPE_V",
        source_provenance="ref:test",
    )
    assert default_rule.mechanism_tag == "r_custom_test"
    reloaded = AscendCausalRule.from_dict(r02.to_dict())
    assert reloaded.mechanism_tag == r02.mechanism_tag
    assert reloaded.concept_description == r02.concept_description

    for rule in rules:
        assert rule.source_provenance != "ref:asc_devkit"
        assert "asc_devkit" not in rule.source_provenance
        combined_text = "\n".join(
            [
                rule.rule_id,
                rule.mechanism_tag,
                rule.concept_description,
                rule.regime_guard,
                rule.precondition,
                rule.anti_precondition,
                rule.transform_template,
                rule.cpp20_transform_template,
                rule.expected_postcondition,
                rule.rationale,
                rule.source_provenance,
            ]
        )
        assert "910b" not in combined_text.lower()
        assert "measured_latency_us" not in rule.expected_postcondition
        validate_rule_predicate(rule.regime_guard)
        validate_rule_predicate(rule.precondition)
        validate_rule_predicate(rule.anti_precondition)
        validate_rule_predicate(rule.expected_postcondition)


def test_all_41_rules_default_context_and_pairwise_me_matrix() -> None:
    """Verify default eval safety and 0% collision across all 7 ME rule families/pairs."""
    rules = {r.rule_id: r for r in load_seed_rules()}
    minimal_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 16},
        waterlines={},
    )
    default_ctx = build_predicate_context(minimal_ledger, prev_ledger=minimal_ledger)
    assert "dcache_bypass_narrow_store" in default_ctx
    assert "has_empty_group" in default_ctx
    for rule in rules.values():
        assert isinstance(rule.matches(default_ctx), bool)
        assert isinstance(rule.postcondition_realized(default_ctx), bool)

    def _ctx(
        *,
        pipes: dict[str, int] | None = None,
        ii: int | None = 2,
        waterlines: dict[str, object] | None = None,
        extra: dict[str, object] | None = None,
        **ledger_kw: object,
    ) -> dict[str, object]:
        ledger = build_objective_physical_ledger(
            analysis_mode="readonly_envelope",
            is_legal=True,
            steady_state_ii=ii,
            pipe_cycles=pipes or {"AIC.PIPE_M": 64, "AIV0.PIPE_V": 64},
            waterlines=waterlines or {},
            **ledger_kw,
        )
        return build_predicate_context(ledger, extra=extra)

    ub_90 = build_scope_waterline(
        scope="UB",
        capacity_bytes=253952,
        explicit_buffers={"x": 230000},
        implicit_buffers={},
    )
    l1_ok = build_scope_waterline(
        scope="L1",
        capacity_bytes=524288,
        explicit_buffers={"x": 160000},
        implicit_buffers={},
    )

    me_groups: list[tuple[tuple[str, ...], list[dict[str, object]]]] = [
        (
            ("R10_BACKWARD_SYNC_CV_PINGPONG", "R12_UNIT_FLAG_FIXPIPE_CHAIN_FUSION"),
            [
                _ctx(pipes={"AIC.PIPE_M": 64, "AIC.PIPE_FIX": 64, "AIV0.PIPE_V": 40}, ii=4),
                _ctx(pipes={"AIC.PIPE_M": 64, "AIC.PIPE_FIX": 64}, ii=4),
            ],
        ),
        (
            ("R13_SPLIT_KV_WAVE_RECOVERY", "R21_SUBROW_DEPTH_OR_WIDTH_CORE_SHARDING"),
            [
                _ctx(pipes={"AIC.PIPE_M": 64}, grid_blocks=4, wave_occupancy=0.14),
                _ctx(pipes={"AIV0.PIPE_V": 64}, grid_blocks=2, wave_occupancy=2.0 / 56.0),
                _ctx(pipes={"AIV0.PIPE_V": 64}, grid_blocks=1, wave_occupancy=1.0 / 56.0),
            ],
        ),
        (
            ("R17_RMEM_ACCUMULATOR_HOIST", "R26_UB_256KB_SPR_UNLOCK"),
            [
                _ctx(pipes={"AIV0.PIPE_V": 64}, waterlines={"UB": ub_90}, in_loop_ub_reload_stores=2),
                _ctx(pipes={"AIV0.PIPE_V": 64}, waterlines={"UB": ub_90}, in_loop_ub_reload_stores=0),
            ],
        ),
        (
            (
                "R24_VLOOPV2_LEGALIZATION_AND_DIRECT_VF",
                "R30_REGBASE_MX_QUANT_DATABLOCK",
                "R40_SIMTVF_FP32_WIDEN_BEFORE_SUBBYTE_QUANT",
            ),
            [
                _ctx(
                    pipes={"AIV0.PIPE_V": 64},
                    has_lowbit_quant=False,
                    in_loop_ub_reload_stores=2,
                ),
                _ctx(
                    pipes={"AIV0.PIPE_V": 64},
                    has_lowbit_quant=True,
                    in_loop_ub_reload_stores=2,
                ),
                _ctx(
                    pipes={"AIV0.PIPE_V": 64},
                    has_lowbit_quant=True,
                    opaque_dynamic_regions=("simt_vf",),
                ),
            ],
        ),
        (
            ("R29_W4_AIV_SUBBYTE_UNPACK", "R39_FIXPIPE_DEQ_SCALE_FOLD"),
            [
                _ctx(
                    pipes={"AIC.PIPE_M": 64, "AIC.PIPE_FIX": 64, "AIV0.PIPE_V": 64},
                    has_lowbit_quant=True,
                ),
                _ctx(
                    pipes={"AIC.PIPE_M": 64, "AIC.PIPE_FIX": 64, "AIV0.PIPE_V": 16},
                    has_lowbit_quant=True,
                ),
            ],
        ),
        (
            ("R32_GROUPED_ROLLING_CORE_DISPATCH", "R41_EMPTY_GROUP_ZERO_TOKEN_SKIP"),
            [
                _ctx(
                    pipes={"AIC.PIPE_M": 64, "AIV0.PIPE_V": 32},
                    wave_occupancy=0.50,
                    extra={"has_empty_group": False, "iteration_regularity": "ragged_group"},
                ),
                _ctx(
                    pipes={"AIC.PIPE_M": 64, "AIV0.PIPE_V": 32},
                    wave_occupancy=0.50,
                    extra={"has_empty_group": True, "iteration_regularity": "ragged_group"},
                ),
            ],
        ),
        (
            (
                "R16_PADDED_PITCH_TAIL_MASKING",
                "R36_SUBATOM_CONTIGUOUS_ALIGNED_CORE_SLICE",
                "R38_COALESCE_SUB32B_TAIL_DMA_STORE",
            ),
            [
                _ctx(
                    pipes={"AIV0.PIPE_V": 64, "AIV0.PIPE_MTE3": 32},
                    unaligned_dma_padding_bytes=16,
                    min_gm_dma_bytes=64,
                ),
                _ctx(
                    pipes={"AIV0.PIPE_V": 64, "AIV0.PIPE_MTE2": 32},
                    unaligned_dma_padding_bytes=16,
                    min_gm_dma_bytes=64,
                    dma_burst_efficiency_ratio=0.60,
                    extra={"contig_atom_ratio": 0.5},
                ),
                _ctx(
                    pipes={"AIV0.PIPE_V": 64, "AIV0.PIPE_MTE3": 32},
                    min_gm_dma_bytes=16,
                ),
                _ctx(
                    pipes={"AIV0.PIPE_V": 64, "AIV0.PIPE_MTE3": 32},
                    unaligned_dma_padding_bytes=16,
                    extra={"dcache_bypass_narrow_store": 4.0},
                ),
            ],
        ),
    ]

    for group_ids, probe_ctxs in me_groups:
        fired: set[str] = set()
        for probe_ctx in probe_ctxs:
            matched = [rid for rid in group_ids if rules[rid].matches(probe_ctx)]
            assert len(matched) <= 1, f"ME collision in {group_ids}: {matched}"
            fired.update(matched)
        assert fired == set(group_ids), f"Not all rules fired in {group_ids}: {fired}"

    ctx_r28_ub_only = _ctx(
        pipes={"AIC.PIPE_M": 64, "AIV0.PIPE_V": 32},
        waterlines={"L1": l1_ok, "UB": ub_90},
        has_online_rescale_loop=True,
    )
    assert rules["R28_INPLACE_L1_SLOT_ALIASING"].matches(ctx_r28_ub_only) is False


def test_reachable_ub_scalar_compute_loop_detector_and_dead_helper_isolation() -> None:
    """Detect reachable __ubuf__ scalar compute loops while ignoring dead helpers and SIMD/SIMT."""
    scalar_ub_with_dead_simd = dedent(
        """
        #include <tl_templates/ascend/common.h>
        __aicore__ inline void dead_vec_helper(__ubuf__ float* p) {
            for (int i = 0; i < 64; ++i) {
                simd_inst::vadd(p, p, p);
            }
        }
        extern "C" __global__ __vector__ void norm_kernel(__gm__ float* x, __gm__ float* y) {
            __ubuf__ float* ub_x = (__ubuf__ float*)0;
            asc_copy_gm2ub_align(ub_x, x, 256);
            for (int i = 0; i < 64; ++i) {
                ub_x[i] = ub_x[i] * 0.5f + 1.0f;
            }
            asc_copy_ub2gm_align(y, ub_x, 256);
        }
        extern "C" int launch(void* stream, void* x, void* y) {
            norm_kernel<<<28, nullptr, stream>>>((__gm__ float*)x, (__gm__ float*)y);
            return 0;
        }
        """
    )
    loop_count = count_reachable_ub_scalar_compute_loops(scalar_ub_with_dead_simd)
    assert loop_count == 1
    assert detect_scalar_compute_loop_fallback(cpp_source=scalar_ub_with_dead_simd) >= 1.0
    summary = parse_bisheng_res_usage(asm_or_dump_text=scalar_ub_with_dead_simd)
    assert summary.extra_signals.get("scalar_compute_loop_fallback", 0.0) >= 1.0

    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=120,
        pipe_cycles={"AIV0.PIPE_V": 60, "AIV0.PIPE_MTE2": 120},
        waterlines={},
        ub_scalar_compute_loops=loop_count,
    )
    roundtrip = ObjectivePhysicalLedger.from_dict(ledger.to_dict())
    assert roundtrip.ub_scalar_compute_loops == 1

    ctx = build_predicate_context(roundtrip)
    assert ctx["ub_scalar_compute_loops"] == 1
    assert ctx["scalar_compute_loop_fallback"] >= 1.0
    assert ctx["l2_pmu_available"] == 0

    rules = {r.rule_id: r for r in load_seed_rules()}
    assert rules["R34_SCALAR_LOOP_TO_VECTOR_DMA_PIPELINE"].matches(ctx) is True


def test_vectorized_simd_and_parallel_simt_loops_are_ignored() -> None:
    """Ignore loops using SIMD intrinsics or parallel __simt_vf__ kernels."""
    simd_src = dedent(
        """
        #include <tl_templates/ascend/common.h>
        extern "C" __global__ __vector__ void vec_kernel(__gm__ float* x, __gm__ float* y) {
            __ubuf__ float* ub_x = (__ubuf__ float*)0;
            for (int i = 0; i < 4; ++i) {
                simd_inst::vadd(ub_x + i * 64, ub_x + i * 64, ub_x + i * 64);
            }
        }
        extern "C" int launch(void* stream, void* x, void* y) {
            vec_kernel<<<28, nullptr, stream>>>((__gm__ float*)x, (__gm__ float*)y);
            return 0;
        }
        """
    )
    assert count_reachable_ub_scalar_compute_loops(simd_src) == 0
    assert detect_scalar_compute_loop_fallback(cpp_source=simd_src) == 0.0

    parallel_simt_src = dedent(
        """
        #include <tl_templates/ascend/common.h>
        __simt_vf__ __launch_bounds__(128) void simt_body(__ubuf__ float* ub, __gm__ float* g) {
            for (int i = 0; i < 16; ++i) {
                ub[i] = g[i] * 2.0f + 1.0f;
            }
        }
        extern "C" __global__ __vector__ void simt_kernel(__gm__ float* x) {
            __ubuf__ float* ub_x = (__ubuf__ float*)0;
            asc_vf_call<simt_body>(dim3(128, 1, 1), ub_x, x);
        }
        extern "C" int launch(void* stream, void* x) {
            simt_kernel<<<28, nullptr, stream>>>((__gm__ float*)x);
            return 0;
        }
        """
    )
    assert count_reachable_ub_scalar_compute_loops(parallel_simt_src) == 0
    assert detect_scalar_compute_loop_fallback(cpp_source=parallel_simt_src) == 0.0


def test_l2_pmu_available_gating_and_r34_anti_precondition() -> None:
    """Suppress L2 ratio false triggers when l2_pmu_available==0 and block R34 at high vec ratio."""
    rules = {r.rule_id: r for r in load_seed_rules()}
    r34 = rules["R34_SCALAR_LOOP_TO_VECTOR_DMA_PIPELINE"]

    vec_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=100,
        pipe_cycles={"AIV0.PIPE_V": 100, "AIV0.PIPE_MTE2": 100},
        waterlines={},
    )
    ctx_static = build_predicate_context(vec_ledger)
    assert ctx_static["l2_pmu_available"] == 0
    assert r34.matches(ctx_static) is False

    empty_msprof_res = compute_static_dynamic_residual(vec_ledger, {"dur_us": 15.0})
    ctx_empty_msprof = build_predicate_context(vec_ledger, residual=empty_msprof_res)
    assert ctx_empty_msprof["l2_pmu_available"] == 0
    assert r34.matches(ctx_empty_msprof) is False

    blocked_env_res = compute_static_dynamic_residual(
        vec_ledger,
        {"dur_us": 15.0, "aiv_vec_ratio": 0.0, "is_analytical_proxy": True},
    )
    ctx_blocked_env = build_predicate_context(vec_ledger, residual=blocked_env_res)
    assert ctx_blocked_env["l2_pmu_available"] == 0
    assert r34.matches(ctx_blocked_env) is False

    ctx_high_vec = build_predicate_context(
        vec_ledger,
        extra={"aiv_vec_ratio": 0.42, "l2_pmu_available": 1, "ub_scalar_compute_loops": 2},
    )
    assert ctx_high_vec["l2_pmu_available"] == 1
    assert ctx_high_vec["aiv_vec_ratio"] == pytest.approx(0.42)
    assert ctx_high_vec["l2_aiv_vec_ratio"] == pytest.approx(0.42)
    assert r34.matches(ctx_high_vec) is False

    real_pmu_res = compute_static_dynamic_residual(
        vec_ledger,
        {"dur_us": 25.0, "aiv_vec_ratio": 0.05, "aic_mac_ratio": 0.0},
    )
    ctx_low_vec = build_predicate_context(vec_ledger, residual=real_pmu_res)
    assert ctx_low_vec["l2_pmu_available"] == 1
    assert ctx_low_vec["aiv_vec_ratio"] == pytest.approx(0.05)
    assert r34.matches(ctx_low_vec) is True


def test_residual_stage6_signals_and_prev_namespace_closure() -> None:
    """Preserve Stage-6 signals on StaticDynamicResidual and L1/L2 fields in prev.*."""
    rules = {r.rule_id: r for r in load_seed_rules()}
    pre_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=8,
        pipe_cycles={"AIV0.PIPE_V": 80, "AIV0.PIPE_MTE2": 64, "AIV0.PIPE_MTE3": 32},
        waterlines={},
        min_gm_dma_bytes=16,
        dma_burst_efficiency_ratio=0.55,
    )
    post_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=4,
        pipe_cycles={"AIV0.PIPE_V": 40, "AIV0.PIPE_MTE2": 32, "AIV0.PIPE_MTE3": 16},
        waterlines={},
        min_gm_dma_bytes=64,
        dma_burst_efficiency_ratio=0.92,
    )
    pre_comp = BishengResourceSummary(
        vec_regs=32,
        scalar_regs=20,
        spill_bytes=128,
        stack_bytes=64,
        extra_signals={"ub_scalar_compute_loops": 2.0},
    )
    post_comp = BishengResourceSummary(vec_regs=48, scalar_regs=16, spill_bytes=0, stack_bytes=0)
    pre_obs = pre_ledger.to_observation_5d(
        compiler_summary=pre_comp,
        telemetry={
            "dur_us": 42.0,
            "aiv_vec_ratio": 0.12,
            "mte2_wait_ratio": 0.40,
            "l2_hit_rate": 0.45,
        },
    )
    pre_res = StaticDynamicResidual(
        measured_latency_us=42.0,
        predicted_bottleneck_util=0.75,
        measured_aic_mac_ratio=0.0,
        measured_aiv_vec_ratio=0.12,
        measured_mte2_wait_ratio=0.40,
        measured_sync_stall_ratio=0.10,
        unexplained_gap_ratio=0.25,
        spill_bytes=128,
        l2_hit_rate=0.45,
        scalar_compute_loop_fallback=2.5,
        dcache_bypass_narrow_store=4.0,
        observation=pre_obs,
    )
    post_res = StaticDynamicResidual(
        measured_latency_us=18.0,
        predicted_bottleneck_util=0.85,
        measured_aic_mac_ratio=0.0,
        measured_aiv_vec_ratio=0.72,
        measured_mte2_wait_ratio=0.15,
        measured_sync_stall_ratio=0.04,
        unexplained_gap_ratio=0.05,
        spill_bytes=0,
        l2_hit_rate=0.88,
        scalar_compute_loop_fallback=0.0,
        dcache_bypass_narrow_store=0.0,
    )
    pre_ratios = CasePhysicsRatios(
        case_id=1,
        stride_ratio=1.0,
        contig_atom_ratio=0.5,
        is_atom_aligned=False,
        working_set_sram_ratio=0.4,
        working_set_vreg_ratio=0.3,
        wave_occupancy_ratio=0.5,
        oi_ridge_ratio=0.4,
        inner_contiguous_bytes=16,
    )

    pre_ctx = build_predicate_context(
        pre_ledger,
        residual=pre_res,
        extra={"marker_flag": True},
    )
    assert pre_ctx["scalar_compute_loop_fallback"] == pytest.approx(2.5)
    assert pre_ctx["dcache_bypass_narrow_store"] == pytest.approx(4.0)
    assert pre_ctx["extra_signals"]["scalar_compute_loop_fallback"] == pytest.approx(2.5)
    assert pre_ctx["extra_signals"]["dcache_bypass_narrow_store"] == pytest.approx(4.0)
    assert rules["R34_SCALAR_LOOP_TO_VECTOR_DMA_PIPELINE"].matches(pre_ctx) is True
    assert rules["R16_PADDED_PITCH_TAIL_MASKING"].matches(pre_ctx) is False
    assert rules["R38_COALESCE_SUB32B_TAIL_DMA_STORE"].matches(pre_ctx) is True

    post_ctx = build_predicate_context(
        post_ledger,
        residual=post_res,
        compiler_summary=post_comp,
        prev_ledger=pre_ledger,
        prev_residual=pre_res,
        prev_compiler_summary=pre_comp,
        prev_case_ratios=pre_ratios,
    )
    prev = post_ctx["prev"]
    assert isinstance(prev, dict)
    assert prev["stack_bytes"] == 64
    assert prev["spill_bytes"] == 128
    assert prev["vec_regs"] == 32
    assert prev["l1_vec_regs"] == 32
    assert prev["l2_hit_rate"] == pytest.approx(0.45)
    assert prev["scalar_compute_loop_fallback"] == pytest.approx(2.5)
    assert prev["dcache_bypass_narrow_store"] == pytest.approx(4.0)
    assert prev["contig_atom_ratio"] == pytest.approx(0.5)
    assert prev["onchip_sram_reg_gap"] > 0.0
    assert rules["R38_COALESCE_SUB32B_TAIL_DMA_STORE"].postcondition_realized(post_ctx) is True
    assert rules["R36_SUBATOM_CONTIGUOUS_ALIGNED_CORE_SLICE"].postcondition_realized(post_ctx) is True
    assert rules["R25_SCALAR_BOUND_SPILL_ELIMINATION"].postcondition_realized(post_ctx) is True

    from tilefoundry_ascend.agent.kb.schema import _observation_to_signal_dict  # noqa: PLC0415

    obs = pre_ledger.to_observation_5d()
    sig_default = _observation_to_signal_dict(obs)
    assert sig_default["min_gm_dma_bytes"] in (0, 16)
    assert sig_default["AIC.PIPE_FIX"] == 0
    assert sig_default["PIPE_FIX"] == 0
    assert sig_default["dcache_bypass_narrow_store"] == pytest.approx(0.0)
    assert sig_default["ub_scalar_compute_loops"] == 0
    assert sig_default["scalar_compute_loop_fallback"] == pytest.approx(0.0)

    object.__setattr__(obs.compute_issue, "l0_fixpipe_cycles", 96)
    object.__setattr__(obs.compute_issue, "l0_ub_scalar_compute_loops", 3)
    object.__setattr__(obs.compute_issue, "l1_scalar_compute_loop_fallback", 3.5)
    object.__setattr__(obs.offchip_dma, "l1_dcache_bypass_narrow_store", 6.0)
    object.__setattr__(obs.offchip_dma, "l0_min_gm_dma_bytes", 16)
    sig_populated = _observation_to_signal_dict(obs)
    assert sig_populated["AIC.PIPE_FIX"] == 96
    assert sig_populated["PIPE_FIX"] == 96
    assert sig_populated["min_gm_dma_bytes"] == 16
    assert sig_populated["dcache_bypass_narrow_store"] == pytest.approx(6.0)
    assert sig_populated["ub_scalar_compute_loops"] == 3
    assert sig_populated["scalar_compute_loop_fallback"] == pytest.approx(3.5)
