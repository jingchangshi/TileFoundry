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
)
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
    """Verify seed_rules.yaml loads R01..R12 with zero operator names or machine paths."""
    yaml_text = default_seed_rules_path().read_text(encoding="utf-8")
    assert "/home/" not in yaml_text
    assert "/Users/" not in yaml_text
    assert "/data" not in yaml_text
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
    assert len(rules) >= 10
    ids = {rule.rule_id for rule in rules}
    for prefix in ("R01_", "R02_", "R03_", "R04_", "R05_", "R06_", "R07_", "R08_", "R09_", "R10_"):
        assert any(rid.startswith(prefix) for rid in ids)


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
    feat_fix = extract_structural_features(ledger=before_fix)
    ctx_fix = build_predicate_context(before_fix, features=feat_fix)
    assert rules["R03_ACCUM_SPAN_FIXPIPE_RELIEF"].matches(ctx_fix) is True

    after_fix = build_objective_physical_ledger(
        analysis_mode="exact_modulo_z3",
        is_legal=True,
        steady_state_ii=2,
        pipe_cycles={"AIC.PIPE_M": 128, "AIC.PIPE_FIX": 96},
        waterlines={},
    )
    post_ctx_fix = build_predicate_context(after_fix, features=feat_fix, prev_ledger=before_fix)
    assert rules["R03_ACCUM_SPAN_FIXPIPE_RELIEF"].postcondition_realized(post_ctx_fix) is True

    before_split = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIC.PIPE_M": 40, "AIV0.PIPE_V": 80},
        waterlines={},
    )
    ctx_split = build_predicate_context(
        before_split,
        features=extract_structural_features(ledger=before_split),
    )
    assert rules["R01_DUAL_COPY_1C2V_SPLIT"].matches(ctx_split) is True

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
    )
    ctx_bloat = build_predicate_context(
        bloat_ledger,
        features=extract_structural_features(ledger=bloat_ledger, indirect_sparse_gather=True),
    )
    assert rules["R04_MTE2_SPARSE_GATHER_COALESCE"].matches(ctx_bloat) is True
    assert rules["R07_IMPLICIT_BLOAT_HEADROOM_TRIM"].matches(ctx_bloat) is True


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
