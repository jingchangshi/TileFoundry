"""Unit tests for WP3 8-dim capability router, causal rules KB, and Bayesian tracker."""

from pathlib import Path
import re

import pytest

from tilefoundry_ascend.agent import (
    AscendCausalRule,
    BayesianRuleTracker,
    UnsafePredicateError,
    build_predicate_context,
    default_seed_rules_path,
    extract_structural_features,
    load_seed_rules,
    route_execution_track,
    safe_eval_predicate,
    validate_rule_predicate,
)
from tilefoundry_ascend.agent.orchestrator import StaticDynamicResidual
from tilefoundry_ascend.ascend.passes import (
    build_objective_physical_ledger,
    build_scope_waterline,
)
from tilefoundry_ascend.ascend.target import AscendTarget


def test_capability_router_extracts_8_features_and_routes_tracks() -> None:
    """Verify phi_1..phi_8 extraction across developer, simt_vf, and manual_ssbuf tracks."""
    import tilelang.ascend.language as T

    @T.prim_func
    def static_gemm(
        A: T.Tensor((128, 64), "bfloat16"),
        B: T.Tensor((128, 64), "bfloat16"),
        C: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(56):
            a_l1 = T.alloc_l1((128, 64), "bfloat16")
            b_l1 = T.alloc_l1((128, 64), "bfloat16")
            c_l0c = T.alloc_l0c((128, 128), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, c_l0c, transpose_B=True)
            T.copy(c_l0c, C)

    dev_ledger = build_objective_physical_ledger(
        analysis_mode="exact_modulo_z3",
        is_legal=True,
        steady_state_ii=2,
        pipe_cycles={"AIC.PIPE_M": 64, "AIV0.PIPE_V": 16},
        waterlines={},
    )
    f_dev = extract_structural_features(static_gemm, ledger=dev_ledger)
    assert len(f_dev.as_tuple()) == 8
    assert f_dev.phi_1_static_affine_bounds is True
    assert f_dev.phi_2_cube_to_vector_ratio == pytest.approx(4.0)
    assert f_dev.phi_4_indirect_sparse_gather is False
    assert f_dev.phi_6_dynamic_cross_core_handoff is False
    assert f_dev.phi_7_inner_contiguous_bytes == 128
    assert f_dev.phi_8_persistent_wave_occupancy == pytest.approx(
        56.0 / float(AscendTarget("ascend.950pr").device.aic_count)
    )
    assert route_execution_track(f_dev).track == "developer"

    @T.prim_func
    def ssbuf_indirect(
        src: T.Tensor((64, 64), "float8_e4m3fn"),
        extra: T.Tensor((64, 32), "float4_e2m1fnx2"),
        idx: T.Tensor((16,), "int32"),
        out: T.Tensor((16, 64), "float32"),
    ):
        with T.MixedKernel(1) as (_, sid):
            ss = T.alloc_ss((2,), "uint32")
            ub = T.alloc_shared((16, 64), "float32")
            ss[sid] = T.Cast("uint32", sid)
            for i in T.serial(16):
                row_idx = idx[i]
                for j in T.serial(64):
                    ub[i, j] = T.Cast("float32", src[row_idx, j])
            T.copy(ub, out)

    f_ss = extract_structural_features(ssbuf_indirect)
    assert f_ss.phi_3_quant_granularity == "mixed_fp8_fp4"
    assert f_ss.phi_4_indirect_sparse_gather is True
    assert f_ss.phi_6_dynamic_cross_core_handoff is True
    assert route_execution_track(f_ss).track == "manual_ssbuf"

    f_simt = extract_structural_features(
        ledger=build_objective_physical_ledger(
            analysis_mode="readonly_envelope",
            is_legal=True,
            steady_state_ii=None,
            pipe_cycles={"AIV0.PIPE_V": 40},
            waterlines={},
            opaque_dynamic_regions=("while_loop",),
        ),
        cross_iter_recurrence="non_associative_scan",
        inner_contiguous_bytes=16,
    )
    assert f_simt.phi_1_static_affine_bounds is False
    assert route_execution_track(f_simt).track == "simt_vf"


def test_seed_rules_yaml_completeness_and_sanitization() -> None:
    """Verify seed_rules.yaml loads R01..R17 with pure ledger predicates and sanitization."""
    yaml_text = default_seed_rules_path().read_text(encoding="utf-8")
    assert "/home/" not in yaml_text
    assert "/Users/" not in yaml_text
    assert "/data" not in yaml_text
    assert "phi_" not in yaml_text
    for forbidden_router_field in (
        "static_affine_bounds",
        "indirect_sparse_gather",
        "quant_granularity",
        "cross_iter_recurrence",
        "dynamic_cross_core_handoff",
        "inner_contiguous_bytes",
        "persistent_wave_occupancy",
    ):
        assert forbidden_router_field not in yaml_text
    for forbidden_op in (
        "sparse_flash_mla",
        "mixed_quant_sparse_flash_mla",
        "lightning_indexer",
        "flash_attention_score",
        "attention_residual",
        "flashmla",
    ):
        assert forbidden_op not in yaml_text.lower()
    assert not re.search(r"\bmla\b", yaml_text.lower())

    rules = load_seed_rules()
    assert len(rules) == 20
    ids = {rule.rule_id for rule in rules}
    for idx in range(1, 21):
        prefix = f"R{idx:02d}_"
        assert any(rid.startswith(prefix) for rid in ids)
    for rule in rules:
        validate_rule_predicate(rule.regime_guard)
        validate_rule_predicate(rule.precondition)
        validate_rule_predicate(rule.expected_postcondition)


def test_safe_ast_evaluator_and_causal_rule_matching() -> None:
    """Verify AST whitelist blocks injection and matches FixPipe, MTE2, and 1C2V rules."""
    with pytest.raises(UnsafePredicateError):
        safe_eval_predicate("__import__('os').system('echo x')", {})
    with pytest.raises(UnsafePredicateError):
        safe_eval_predicate("(lambda x: x > 1)(2)", {})
    with pytest.raises(UnsafePredicateError):
        safe_eval_predicate("AIC._secret > 0", {})

    rules = {r.rule_id: r for r in load_seed_rules()}

    before_fix = build_objective_physical_ledger(
        analysis_mode="exact_modulo_z3",
        is_legal=True,
        steady_state_ii=3,
        pipe_cycles={"AIC.PIPE_M": 64, "AIC.PIPE_FIX": 96},
        waterlines={},
    )
    ctx_fix = build_predicate_context(before_fix)
    assert rules["R03_ACCUM_SPAN_FIXPIPE_RELIEF"].matches(ctx_fix) is True

    after_fix = build_objective_physical_ledger(
        analysis_mode="exact_modulo_z3",
        is_legal=True,
        steady_state_ii=2,
        pipe_cycles={"AIC.PIPE_M": 128, "AIC.PIPE_FIX": 96},
        waterlines={},
    )
    post_ctx_fix = build_predicate_context(after_fix, prev_ledger=before_fix)
    assert rules["R03_ACCUM_SPAN_FIXPIPE_RELIEF"].postcondition_realized(post_ctx_fix) is True

    before_split = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIC.PIPE_M": 40, "AIV0.PIPE_V": 80},
        waterlines={},
    )
    ctx_split = build_predicate_context(before_split)
    assert rules["R01_DUAL_COPY_1C2V_SPLIT"].matches(ctx_split) is True

    pure_vec_split = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 80, "AIV0.PIPE_MTE2": 100},
        waterlines={},
    )
    assert rules["R01_DUAL_COPY_1C2V_SPLIT"].matches(
        build_predicate_context(pure_vec_split)
    ) is True

    after_split = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIC.PIPE_M": 40, "AIV0.PIPE_V": 40, "AIV1.PIPE_V": 40},
        waterlines={},
    )
    assert (
        rules["R01_DUAL_COPY_1C2V_SPLIT"].postcondition_realized(
            build_predicate_context(after_split, prev_ledger=before_split)
        )
        is True
    )

    bloat_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=False,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_MTE2": 64},
        waterlines={
            "UB": build_scope_waterline(
                scope="UB",
                capacity_bytes=253952,
                explicit_buffers={"x": 220000},
                implicit_buffers={"nd2nz_tmp": 65536},
                degraded_single_buffers=("x",),
            )
        },
        mte2_inflight_per_aiv=32,
        unaligned_dma_padding_bytes=12,
        has_indirect_gather=True,
    )
    ctx_bloat = build_predicate_context(bloat_ledger)
    assert rules["R04_MTE2_SPARSE_GATHER_COALESCE"].matches(ctx_bloat) is True
    assert rules["R07_IMPLICIT_BLOAT_HEADROOM_TRIM"].matches(ctx_bloat) is True


def test_all_17_seed_rules_fire_with_features_none() -> None:
    """Verify all 17 rules R01..R17 match and realize postconditions when features=None."""
    rules = {r.rule_id: r for r in load_seed_rules()}
    ub_half = build_scope_waterline(
        scope="UB",
        capacity_bytes=262144,
        explicit_buffers={"a": 131072},
        implicit_buffers={},
    )
    ub_full = build_scope_waterline(
        scope="UB",
        capacity_bytes=253952,
        explicit_buffers={"a": 230000},
        implicit_buffers={},
    )
    ub_trimmed = build_scope_waterline(
        scope="UB",
        capacity_bytes=262144,
        explicit_buffers={"a": 120000},
        implicit_buffers={},
    )

    def _res(
        lat: float,
        *,
        util: float = 0.85,
        mte2_w: float = 0.0,
        sync_s: float = 0.0,
        gap: float = 0.0,
        aiv_v: float = 0.0,
    ) -> StaticDynamicResidual:
        return StaticDynamicResidual(
            measured_latency_us=lat,
            predicted_bottleneck_util=util,
            measured_aic_mac_ratio=0.4,
            measured_aiv_vec_ratio=aiv_v,
            measured_mte2_wait_ratio=mte2_w,
            measured_sync_stall_ratio=sync_s,
            unexplained_gap_ratio=gap,
        )

    cases = [
        (
            "R01_DUAL_COPY_1C2V_SPLIT",
            build_objective_physical_ledger(
                analysis_mode="readonly_envelope",
                is_legal=True,
                steady_state_ii=None,
                pipe_cycles={"AIV0.PIPE_V": 100},
                waterlines={},
            ),
            build_objective_physical_ledger(
                analysis_mode="readonly_envelope",
                is_legal=True,
                steady_state_ii=None,
                pipe_cycles={"AIV0.PIPE_V": 50, "AIV1.PIPE_V": 50},
                waterlines={},
            ),
            _res(40.0),
            _res(22.0),
        ),
        (
            "R02_ONCHIP_L0C_TO_UB_DIRECT",
            build_objective_physical_ledger(
                analysis_mode="exact_modulo_z3",
                is_legal=True,
                steady_state_ii=4,
                pipe_cycles={"AIC.PIPE_M": 64, "AIC.PIPE_FIX": 64, "AIV0.PIPE_MTE2": 80},
                waterlines={},
                gm_roundtrip_bytes=65536,
            ),
            build_objective_physical_ledger(
                analysis_mode="exact_modulo_z3",
                is_legal=True,
                steady_state_ii=2,
                pipe_cycles={"AIC.PIPE_M": 64, "AIC.PIPE_FIX": 64, "AIV0.PIPE_MTE2": 16},
                waterlines={},
                gm_roundtrip_bytes=0,
            ),
            _res(40.0),
            _res(25.0),
        ),
        (
            "R03_ACCUM_SPAN_FIXPIPE_RELIEF",
            build_objective_physical_ledger(
                analysis_mode="exact_modulo_z3",
                is_legal=True,
                steady_state_ii=3,
                pipe_cycles={"AIC.PIPE_M": 64, "AIC.PIPE_FIX": 96},
                waterlines={},
            ),
            build_objective_physical_ledger(
                analysis_mode="exact_modulo_z3",
                is_legal=True,
                steady_state_ii=2,
                pipe_cycles={"AIC.PIPE_M": 128, "AIC.PIPE_FIX": 96},
                waterlines={},
            ),
            _res(30.0),
            _res(20.0),
        ),
        (
            "R04_MTE2_SPARSE_GATHER_COALESCE",
            build_objective_physical_ledger(
                analysis_mode="readonly_envelope",
                is_legal=True,
                steady_state_ii=None,
                pipe_cycles={"AIV0.PIPE_MTE2": 64},
                waterlines={},
                mte2_inflight_per_aiv=24,
                unaligned_dma_padding_bytes=16,
                has_indirect_gather=True,
            ),
            build_objective_physical_ledger(
                analysis_mode="readonly_envelope",
                is_legal=True,
                steady_state_ii=None,
                pipe_cycles={"AIV0.PIPE_MTE2": 32},
                waterlines={},
                mte2_inflight_per_aiv=8,
                unaligned_dma_padding_bytes=0,
            ),
            _res(35.0),
            _res(20.0),
        ),
        (
            "R05_SSBUF_RING_METADATA_HANDOFF",
            build_objective_physical_ledger(
                analysis_mode="readonly_envelope",
                is_legal=True,
                steady_state_ii=None,
                pipe_cycles={"AIC.PIPE_M": 32, "AIV0.PIPE_V": 32},
                waterlines={},
                opaque_dynamic_regions=("dynamic_loop",),
            ),
            build_objective_physical_ledger(
                analysis_mode="readonly_envelope",
                is_legal=True,
                steady_state_ii=None,
                pipe_cycles={"AIC.PIPE_M": 32, "AIV0.PIPE_V": 32},
                waterlines={
                    "SSBUF": build_scope_waterline(
                        scope="SSBUF",
                        capacity_bytes=3072,
                        explicit_buffers={"meta": 512},
                        implicit_buffers={},
                    )
                },
            ),
            _res(40.0, sync_s=0.25),
            _res(28.0, sync_s=0.05),
        ),
        (
            "R06_ONLINE_SOFTMAX_SKIP_SCALE",
            build_objective_physical_ledger(
                analysis_mode="readonly_envelope",
                is_legal=True,
                steady_state_ii=None,
                pipe_cycles={"AIC.PIPE_M": 32, "AIV0.PIPE_V": 96},
                waterlines={},
                has_online_rescale_loop=True,
            ),
            build_objective_physical_ledger(
                analysis_mode="readonly_envelope",
                is_legal=True,
                steady_state_ii=None,
                pipe_cycles={"AIC.PIPE_M": 32, "AIV0.PIPE_V": 48},
                waterlines={},
                has_online_rescale_loop=True,
            ),
            _res(30.0),
            _res(20.0),
        ),
        (
            "R07_IMPLICIT_BLOAT_HEADROOM_TRIM",
            build_objective_physical_ledger(
                analysis_mode="readonly_envelope",
                is_legal=False,
                steady_state_ii=None,
                pipe_cycles={"AIV0.PIPE_V": 40},
                waterlines={
                    "UB": build_scope_waterline(
                        scope="UB",
                        capacity_bytes=253952,
                        explicit_buffers={"x": 220000},
                        implicit_buffers={"tmp": 65536},
                        degraded_single_buffers=("x",),
                    )
                },
            ),
            build_objective_physical_ledger(
                analysis_mode="readonly_envelope",
                is_legal=True,
                steady_state_ii=None,
                pipe_cycles={"AIV0.PIPE_V": 40},
                waterlines={"UB": ub_half},
            ),
            _res(50.0),
            _res(25.0),
        ),
        (
            "R08_PERSISTENT_SWIZZLE_L2_LOCALITY",
            build_objective_physical_ledger(
                analysis_mode="exact_modulo_z3",
                is_legal=True,
                steady_state_ii=4,
                pipe_cycles={"AIC.PIPE_M": 64},
                waterlines={},
                grid_blocks=56,
                wave_occupancy=1.0,
            ),
            build_objective_physical_ledger(
                analysis_mode="exact_modulo_z3",
                is_legal=True,
                steady_state_ii=4,
                pipe_cycles={"AIC.PIPE_M": 64},
                waterlines={},
                grid_blocks=56,
                wave_occupancy=1.0,
            ),
            _res(45.0, util=0.90, mte2_w=0.35, gap=0.35),
            _res(28.0, util=0.90, mte2_w=0.15, gap=0.15),
        ),
        (
            "R09_MIXED_QUANT_DEQUANT_STAGE_OVERLAP",
            build_objective_physical_ledger(
                analysis_mode="exact_modulo_z3",
                is_legal=True,
                steady_state_ii=6,
                pipe_cycles={"AIC.PIPE_M": 40, "AIV0.PIPE_V": 60},
                waterlines={},
                has_lowbit_quant=True,
            ),
            build_objective_physical_ledger(
                analysis_mode="exact_modulo_z3",
                is_legal=True,
                steady_state_ii=2,
                pipe_cycles={"AIC.PIPE_M": 40, "AIV0.PIPE_V": 30},
                waterlines={},
                has_lowbit_quant=True,
            ),
            _res(40.0),
            _res(24.0),
        ),
        (
            "R10_BACKWARD_SYNC_CV_PINGPONG",
            build_objective_physical_ledger(
                analysis_mode="exact_modulo_z3",
                is_legal=True,
                steady_state_ii=4,
                pipe_cycles={"AIC.PIPE_M": 40, "AIV0.PIPE_V": 40},
                waterlines={},
            ),
            build_objective_physical_ledger(
                analysis_mode="exact_modulo_z3",
                is_legal=True,
                steady_state_ii=2,
                pipe_cycles={"AIC.PIPE_M": 40, "AIV0.PIPE_V": 40},
                waterlines={},
            ),
            _res(35.0, sync_s=0.22),
            _res(22.0, sync_s=0.08),
        ),
        (
            "R11_L2_NZ_PAD_BANK_CONFLICT_AVOID",
            build_objective_physical_ledger(
                analysis_mode="readonly_envelope",
                is_legal=True,
                steady_state_ii=None,
                pipe_cycles={"AIV0.PIPE_V": 64},
                waterlines={"UB": ub_half},
                dominant_contig_bytes=512,
            ),
            build_objective_physical_ledger(
                analysis_mode="readonly_envelope",
                is_legal=True,
                steady_state_ii=None,
                pipe_cycles={"AIV0.PIPE_V": 64},
                waterlines={"UB": ub_half},
                dominant_contig_bytes=544,
            ),
            _res(40.0, gap=0.35, aiv_v=0.4),
            _res(24.0, gap=0.10, aiv_v=0.7),
        ),
        (
            "R12_UNIT_FLAG_FIXPIPE_CHAIN_FUSION",
            build_objective_physical_ledger(
                analysis_mode="exact_modulo_z3",
                is_legal=True,
                steady_state_ii=4,
                pipe_cycles={"AIC.PIPE_M": 64, "AIC.PIPE_FIX": 64},
                waterlines={},
            ),
            build_objective_physical_ledger(
                analysis_mode="exact_modulo_z3",
                is_legal=True,
                steady_state_ii=2,
                pipe_cycles={"AIC.PIPE_M": 64, "AIC.PIPE_FIX": 64},
                waterlines={},
            ),
            _res(30.0),
            _res(20.0),
        ),
        (
            "R13_SPLIT_KV_WAVE_RECOVERY",
            build_objective_physical_ledger(
                analysis_mode="exact_modulo_z3",
                is_legal=True,
                steady_state_ii=2,
                pipe_cycles={"AIC.PIPE_M": 64},
                waterlines={},
                grid_blocks=4,
                wave_occupancy=0.1428,
            ),
            build_objective_physical_ledger(
                analysis_mode="exact_modulo_z3",
                is_legal=True,
                steady_state_ii=2,
                pipe_cycles={"AIC.PIPE_M": 64},
                waterlines={},
                grid_blocks=28,
                wave_occupancy=1.0,
            ),
            _res(60.0),
            _res(18.0),
        ),
        (
            "R14_GQA_HEAD_M_FOLDING",
            build_objective_physical_ledger(
                analysis_mode="exact_modulo_z3",
                is_legal=True,
                steady_state_ii=2,
                pipe_cycles={"AIC.PIPE_M": 64},
                waterlines={},
                effective_cube_m=1,
            ),
            build_objective_physical_ledger(
                analysis_mode="exact_modulo_z3",
                is_legal=True,
                steady_state_ii=2,
                pipe_cycles={"AIC.PIPE_M": 64},
                waterlines={},
                effective_cube_m=16,
            ),
            _res(50.0),
            _res(20.0),
        ),
        (
            "R15_ASYMMETRIC_1C2V_STAGE_FISSION",
            build_objective_physical_ledger(
                analysis_mode="exact_modulo_z3",
                is_legal=True,
                steady_state_ii=12,
                pipe_cycles={"AIC.PIPE_M": 64, "AIV0.PIPE_V": 40, "AIV1.PIPE_V": 40},
                waterlines={"UB": ub_full},
            ),
            build_objective_physical_ledger(
                analysis_mode="exact_modulo_z3",
                is_legal=True,
                steady_state_ii=4,
                pipe_cycles={"AIC.PIPE_M": 64, "AIV0.PIPE_V": 40, "AIV1.PIPE_V": 40},
                waterlines={"UB": ub_trimmed},
            ),
            _res(45.0),
            _res(25.0),
        ),
        (
            "R16_PADDED_PITCH_TAIL_MASKING",
            build_objective_physical_ledger(
                analysis_mode="readonly_envelope",
                is_legal=True,
                steady_state_ii=None,
                pipe_cycles={"AIV0.PIPE_V": 64},
                waterlines={},
                unaligned_dma_padding_bytes=16,
                opaque_dynamic_regions=("simt_vf",),
            ),
            build_objective_physical_ledger(
                analysis_mode="readonly_envelope",
                is_legal=True,
                steady_state_ii=None,
                pipe_cycles={"AIV0.PIPE_V": 40},
                waterlines={},
                unaligned_dma_padding_bytes=0,
                opaque_dynamic_regions=(),
            ),
            _res(40.0),
            _res(18.0),
        ),
        (
            "R17_RMEM_ACCUM_AND_256KB_UB_UNLOCK",
            build_objective_physical_ledger(
                analysis_mode="readonly_envelope",
                is_legal=True,
                steady_state_ii=None,
                pipe_cycles={"AIV0.PIPE_V": 64},
                waterlines={"UB": ub_full},
                in_loop_ub_reload_stores=4,
            ),
            build_objective_physical_ledger(
                analysis_mode="readonly_envelope",
                is_legal=True,
                steady_state_ii=None,
                pipe_cycles={"AIV0.PIPE_V": 48},
                waterlines={"UB": ub_trimmed},
                in_loop_ub_reload_stores=0,
            ),
            _res(42.0),
            _res(22.0),
        ),
        (
            "R18_WAVE_QUANTIZED_BURST_TILING",
            build_objective_physical_ledger(
                analysis_mode="readonly_envelope",
                is_legal=True,
                steady_state_ii=None,
                pipe_cycles={"AIV0.PIPE_V": 40, "AIV0.PIPE_MTE2": 64},
                waterlines={"UB": ub_half},
                wave_occupancy=0.45,
                dma_burst_efficiency_ratio=0.57,
            ),
            build_objective_physical_ledger(
                analysis_mode="readonly_envelope",
                is_legal=True,
                steady_state_ii=None,
                pipe_cycles={"AIV0.PIPE_V": 40, "AIV0.PIPE_MTE2": 32},
                waterlines={"UB": ub_half},
                wave_occupancy=0.90,
                dma_burst_efficiency_ratio=0.96,
            ),
            _res(40.0),
            _res(25.0),
        ),
        (
            "R19_STATIC_PINGPONG_UNROLL_NO_ALIAS",
            build_objective_physical_ledger(
                analysis_mode="exact_modulo_z3",
                is_legal=True,
                steady_state_ii=1,
                pipe_cycles={"AIV0.PIPE_V": 40, "AIV0.PIPE_MTE2": 48},
                waterlines={"UB": ub_half},
                uses_dynamic_offset_pipelining=True,
            ),
            build_objective_physical_ledger(
                analysis_mode="readonly_envelope",
                is_legal=True,
                steady_state_ii=None,
                pipe_cycles={"AIV0.PIPE_V": 40, "AIV0.PIPE_MTE2": 32},
                waterlines={"UB": ub_half},
                uses_dynamic_offset_pipelining=False,
            ),
            StaticDynamicResidual(
                measured_latency_us=42.0,
                predicted_bottleneck_util=0.85,
                measured_aic_mac_ratio=0.0,
                measured_aiv_vec_ratio=0.25,
                measured_mte2_wait_ratio=0.42,
                measured_sync_stall_ratio=0.10,
                unexplained_gap_ratio=0.30,
                pseudo_double_buffer_serialized=True,
            ),
            StaticDynamicResidual(
                measured_latency_us=26.0,
                predicted_bottleneck_util=0.85,
                measured_aic_mac_ratio=0.0,
                measured_aiv_vec_ratio=0.45,
                measured_mte2_wait_ratio=0.20,
                measured_sync_stall_ratio=0.05,
                unexplained_gap_ratio=0.10,
                pseudo_double_buffer_serialized=False,
            ),
        ),
        (
            "R20_LOOP_FISSION_REDUCE_SFU_HIDING",
            build_objective_physical_ledger(
                analysis_mode="readonly_envelope",
                is_legal=True,
                steady_state_ii=None,
                pipe_cycles={"AIV0.PIPE_V": 96, "AIV0.PIPE_MTE2": 32},
                waterlines={"UB": ub_half},
                has_coupled_reduce_sfu_accum=True,
            ),
            build_objective_physical_ledger(
                analysis_mode="readonly_envelope",
                is_legal=True,
                steady_state_ii=None,
                pipe_cycles={"AIV0.PIPE_V": 64, "AIV0.PIPE_MTE2": 32},
                waterlines={"UB": ub_half},
                has_coupled_reduce_sfu_accum=False,
            ),
            _res(38.0),
            _res(24.0),
        ),
    ]

    for rule_id, pre_ledger, post_ledger, pre_res, post_res in cases:
        pre_ctx = build_predicate_context(pre_ledger, residual=pre_res)
        assert rules[rule_id].matches(pre_ctx) is True, f"{rule_id} failed pre match"
        post_ctx = build_predicate_context(
            post_ledger, residual=post_res, prev_ledger=pre_ledger
        )
        post_ctx["prev"].update(pre_res.to_dict())
        assert rules[rule_id].postcondition_realized(post_ctx) is True, (
            f"{rule_id} failed postcondition"
        )


def test_bayesian_rule_tracker_ranking_pruning_and_roundtrip(tmp_path: Path) -> None:
    """Verify Bayesian tracker ranks, prunes failing rules, and persists posteriors."""
    tracker = BayesianRuleTracker.from_seed_yaml()
    ledger = build_objective_physical_ledger(
        analysis_mode="exact_modulo_z3",
        is_legal=True,
        steady_state_ii=2,
        pipe_cycles={"AIC.PIPE_M": 64, "AIC.PIPE_FIX": 96},
        waterlines={},
    )
    ctx = build_predicate_context(
        ledger,
        features=extract_structural_features(ledger=ledger),
    )
    top_before = tracker.rank_applicable_rules(ctx, top_k=2)
    assert len(top_before) == 2

    for _ in range(7):
        tracker.record_outcome("R03_ACCUM_SPAN_FIXPIPE_RELIEF", postcondition_realized=False)
    for _ in range(4):
        tracker.record_outcome(
            "R12_UNIT_FLAG_FIXPIPE_CHAIN_FUSION",
            postcondition_realized=True,
            speed_gain=0.25,
        )
    pruned = tracker.prune_underperforming(min_trials=5, prune_threshold=0.30)
    assert "R03_ACCUM_SPAN_FIXPIPE_RELIEF" in pruned

    remaining_ids = [r.rule_id for r in tracker.rank_applicable_rules(ctx)]
    assert "R03_ACCUM_SPAN_FIXPIPE_RELIEF" not in remaining_ids
    assert "R12_UNIT_FLAG_FIXPIPE_CHAIN_FUSION" in remaining_ids

    custom_rule = AscendCausalRule(
        rule_id="R99_DISTILLED_CUSTOM",
        regime_guard="phi_2 > 0.0",
        precondition="AIC.PIPE_FIX > AIC.PIPE_M",
        transform_template="Adjust inner accumulation span cleanly.",
        expected_postcondition="AIC.PIPE_FIX <= AIC.PIPE_M",
        source_provenance="flywheel:ablation",
        alpha_prior=5.0,
        beta_prior=1.0,
    )
    tracker.admit_rule(custom_rule)
    state_file = tmp_path / "tracker.json"
    tracker.save_json(state_file)

    restored = BayesianRuleTracker.from_seed_yaml()
    restored.admit_rule(custom_rule)
    restored.load_posteriors_json(state_file)
    assert restored.posteriors["R03_ACCUM_SPAN_FIXPIPE_RELIEF"].pruned is True
    assert restored.posteriors["R12_UNIT_FLAG_FIXPIPE_CHAIN_FUSION"].mean_speedup == pytest.approx(
        0.25
    )


def test_bayesian_tracker_paradigm_bucketed_pruning_and_isolation(tmp_path: Path) -> None:
    """Verify paradigm-scoped pruning isolates pure_vector failures from cv_coupled."""
    tracker = BayesianRuleTracker.from_seed_yaml()
    vec_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 80, "AIV0.PIPE_MTE2": 100},
        waterlines={},
    )
    cube_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIC.PIPE_M": 40, "AIV0.PIPE_V": 80},
        waterlines={},
    )
    vec_ctx = build_predicate_context(vec_ledger)
    cube_ctx = build_predicate_context(cube_ledger)

    rule_id = "R01_DUAL_COPY_1C2V_SPLIT"
    for _ in range(10):
        tracker.record_outcome(rule_id, postcondition_realized=False, paradigm="pure_vector")
    tracker.record_outcome(
        rule_id,
        postcondition_realized=True,
        speed_gain=0.4,
        paradigm="cv_coupled",
    )

    pruned_vec = tracker.prune_underperforming(
        min_trials=5, prune_threshold=0.25, paradigm="pure_vector"
    )
    assert rule_id in pruned_vec
    assert tracker.is_pruned(rule_id, paradigm="pure_vector") is True
    assert tracker.is_pruned(rule_id, paradigm="cv_coupled") is False
    assert tracker.is_pruned(rule_id) is False

    vec_ids = [r.rule_id for r in tracker.rank_applicable_rules(vec_ctx, paradigm="pure_vector")]
    cube_ids = [r.rule_id for r in tracker.rank_applicable_rules(cube_ctx, paradigm="cv_coupled")]
    assert rule_id not in vec_ids
    assert rule_id in cube_ids

    ckpt_file = tmp_path / "paradigm_tracker.json"
    tracker.save_json(ckpt_file)
    restored = BayesianRuleTracker.from_seed_yaml()
    restored.load_posteriors_json(ckpt_file)
    assert restored.is_pruned(rule_id, paradigm="pure_vector") is True
    assert restored.is_pruned(rule_id, paradigm="cv_coupled") is False
    assert rule_id not in [
        r.rule_id for r in restored.rank_applicable_rules(vec_ctx, paradigm="pure_vector")
    ]
    assert rule_id in [
        r.rule_id for r in restored.rank_applicable_rules(cube_ctx, paradigm="cv_coupled")
    ]
