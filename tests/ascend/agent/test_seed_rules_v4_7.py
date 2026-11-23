"""Tests for v4.7 seed rules (R01..R40), reachable UB scalar loop detection, and L2 PMU gating."""

from __future__ import annotations

from textwrap import dedent

import pytest

from tilefoundry_ascend.agent.kb import (
    build_predicate_context,
    load_seed_rules,
    validate_rule_predicate,
)
from tilefoundry_ascend.agent.orchestrator.residual_beam import compute_static_dynamic_residual
from tilefoundry_ascend.ascend.cost_model import (
    count_reachable_ub_scalar_compute_loops,
    detect_scalar_compute_loop_fallback,
    parse_bisheng_res_usage,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    ObjectivePhysicalLedger,
    build_objective_physical_ledger,
)


def test_seed_rules_exact_40_clean_provenance_and_ast_validity() -> None:
    """Verify exact 40 seed rules R01..R40, zero legacy provenance/910B, and AST validity."""
    rules = load_seed_rules()
    assert len(rules) == 40
    rule_ids = [r.rule_id for r in rules]
    assert len(set(rule_ids)) == 40
    for idx, rule_id in enumerate(rule_ids, start=1):
        assert rule_id.startswith(f"R{idx:02d}_")

    by_id = {r.rule_id: r for r in rules}
    assert "R34_SCALAR_LOOP_TO_VECTOR_DMA_PIPELINE" in by_id
    assert "R38_COALESCE_SUB32B_TAIL_DMA_STORE" in by_id
    assert "R39_COUPLED_1C2V_FIXPIPE_L0C2UB_15ARG_PIPELINE" in by_id
    assert "R40_SIMTVF_FP32_WIDEN_BEFORE_SUBBYTE_QUANT" in by_id

    for rule in rules:
        assert rule.source_provenance != "ref:asc_devkit"
        assert "asc_devkit" not in rule.source_provenance
        combined_text = "\n".join(
            [
                rule.rule_id,
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
        validate_rule_predicate(rule.regime_guard)
        validate_rule_predicate(rule.precondition)
        validate_rule_predicate(rule.anti_precondition)
        validate_rule_predicate(rule.expected_postcondition)


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
