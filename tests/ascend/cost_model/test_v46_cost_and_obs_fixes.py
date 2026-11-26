"""Tests for Ascend cost model, KB observation L2>L1>L0 ranking, and anchor specialization."""

from __future__ import annotations

import math

import pytest

from examples.ascend950pr_ops.rms_norm.kernels import (
    _build_rms_norm_prim_func,
    _select_rms_norm_config,
)
from tilefoundry_ascend.agent.kb import (
    SymptomFamily,
    build_predicate_context,
    load_seed_rules,
    rank_symptom_families,
)
from tilefoundry_ascend.ascend.cost_model import (
    Ascend950PipeScheduleEvaluator,
    HybridAscendCostModel,
    build_cost_features_from_ledger,
    extract_microarch_profile_from_tirx,
    project_case_workload,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger
from tilefoundry_ascend.ascend.schedule.envelope_inspector import inspect_tilelang_primfunc
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes
from tilefoundry_ascend.ascend.target import AscendTarget


def test_rank_symptom_families_honors_l2_over_l1_over_l0() -> None:
    """Verify L2 PMU overrides missing L0/L1 cycles and ranks moe_gating without scalar trap."""
    high_vec = dict(
        rank_symptom_families(
            {
                "measured_aiv_vec_ratio": 0.85,
                "measured_aic_mac_ratio": 0.0,
                "l0_cube_cycles": 0,
                "l0_vec_cycles": 0,
                "has_l1_compiler_summary": True,
                "l1_vec_regs": 0,
            }
        )
    )
    assert high_vec[SymptomFamily.SCALAR_LOOP_TRAP] == 0.0

    zero_pmu = dict(
        rank_symptom_families(
            {
                "measured_aiv_vec_ratio": 0.0,
                "measured_aic_mac_ratio": 0.0,
                "l0_cube_cycles": 0,
                "l0_vec_cycles": 0,
            }
        )
    )
    assert zero_pmu[SymptomFamily.SCALAR_LOOP_TRAP] == 1.0

    moe_ranked = rank_symptom_families(
        {
            "measured_aiv_vec_ratio": 0.9999,
            "measured_aic_mac_ratio": 0.0,
            "effective_compute_eta": 0.0001,
            "transfer_inflation_ratio": 285.0,
            "l0_cube_cycles": 0,
            "l0_vec_cycles": 0,
            "has_simtvf": True,
        }
    )
    moe_map = dict(moe_ranked)
    assert moe_map[SymptomFamily.SCALAR_LOOP_TRAP] == 0.0
    assert moe_ranked[0][0] in (
        SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP,
        SymptomFamily.COMPUTE_INEFFECTIVE_THROUGHPUT,
    )
    assert moe_map[SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP] > 0.45
    assert moe_map[SymptomFamily.COMPUTE_INEFFECTIVE_THROUGHPUT] > 0.45
    assert sum(w for _, w in moe_ranked) == pytest.approx(1.0, abs=1e-6)


def test_seed_rules_41_r34_r19_r36_r37_predicates() -> None:
    """Verify 41 seed rules load and R34/R19/R36/R37 honor preconditions and anti-guards."""
    rules_tuple = load_seed_rules()
    assert len(rules_tuple) == 41
    rules = {r.rule_id: r for r in rules_tuple}

    simtvf_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 0, "AIV0.PIPE_MTE2": 100},
        waterlines={},
        opaque_dynamic_regions=("simt_vf",),
        dma_burst_efficiency_ratio=0.72,
    )
    ctx_r34_blocked = build_predicate_context(
        simtvf_ledger,
        residual={
            "l2_aiv_vec_ratio": 0.25,
            "l2_aic_mac_ratio": 0.0,
            "effective_compute_eta": 0.18,
            "l2_hit_rate": 0.40,
            "transfer_inflation_ratio": 1.15,
        },
        extra={"contig_atom_ratio": 0.25},
    )
    assert rules["R34_SCALAR_LOOP_TO_VECTOR_DMA_PIPELINE"].matches(ctx_r34_blocked) is False
    assert rules["R36_SUBATOM_CONTIGUOUS_ALIGNED_CORE_SLICE"].matches(ctx_r34_blocked) is True

    ctx_r37 = build_predicate_context(
        simtvf_ledger,
        residual={
            "l2_aiv_vec_ratio": 0.72,
            "effective_compute_eta": 0.12,
            "spill_bytes": 0,
        },
    )
    assert rules["R37_SIMTVF_MULTIROW_ILP_CONSTANT_HOIST"].matches(ctx_r37) is True

    pipelined_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 120, "AIV0.PIPE_MTE2": 200},
        waterlines={},
        uses_dynamic_offset_pipelining=True,
        in_loop_pipe_all_barriers=0,
        microarch_profile={"pipeline_depth": 2},
    )
    ctx_r19_blocked = build_predicate_context(
        pipelined_ledger,
        residual={
            "pseudo_double_buffer_serialized": False,
            "measured_mte2_wait_ratio": 0.48,
        },
    )
    assert rules["R19_STATIC_PINGPONG_UNROLL_NO_ALIAS"].matches(ctx_r19_blocked) is False


def test_rms_norm_case2_replay_and_simtvf_scalar_index_normalization() -> None:
    """Verify rms_norm Case 2 projection and SimtVF scalar index math stay within log bound."""
    target = AscendTarget("ascend.950pr")
    cm = HybridAscendCostModel(target)
    cfg = _select_rms_norm_config(1024, "bfloat16", m_rows=4096, schedule_mode="tuned")
    fn = _build_rms_norm_prim_func(
        d_dim=1024,
        tl_dtype="bfloat16",
        tile_m=int(cfg["tile_m"]),
        num_stages=int(cfg["num_stages"]),
        schedule_mode="tuned",
    )
    res = inspect_tilelang_primfunc(
        fn,
        target=target,
        pass_configs=cfg["pass_configs"],
        compile_flags=cfg["compile_flags"],
        var_bindings={"m_rows": 4096},
    )
    prof = extract_microarch_profile_from_tirx(fn, var_bindings={"m_rows": 4096})
    assert prof.scalar_ops > 0

    axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
    wl = project_case_workload(
        res.ledger,
        microarch_profile=prof,
        case_dims={"B": 32, "S": 128, "D": 1024, "m_rows": 4096},
        policy=cfg,
        axes=axes,
        target=target,
    )
    assert wl["profile_is_per_core_full_loop"] is True
    assert int(wl["total_tiles"]) == int(res.ledger.grid_blocks)
    assert float(wl["gm_bytes"]) < 64.0e6

    pred = cm.predict(wl)
    assert abs(math.log(49.37 / max(1e-6, pred.pred_us))) < 1.5

    fn_simt = _build_rms_norm_prim_func(
        d_dim=16,
        tl_dtype="bfloat16",
        tile_m=256,
        num_stages=2,
        schedule_mode="tuned",
    )
    prof_simt = extract_microarch_profile_from_tirx(fn_simt, var_bindings={"m_rows": 4096})
    assert prof_simt.scalar_ops > 0
    assert prof_simt.int_alu_ops == 0
    ev = Ascend950PipeScheduleEvaluator(target)
    simt_sched = ev.evaluate_profile(prof_simt, total_tiles=56)
    assert simt_sched.ave_compute_ns < 5000.0


def test_dynamic_total_elements_t_copy_specializes_mte2_and_gm_to_ub() -> None:
    """Verify T.dynamic('total_elements', 'int32') specializes T.copy without zero MTE2."""
    import tilelang.ascend.language as T  # noqa: PLC0415

    total_elements = T.dynamic("total_elements", "int32")
    s_pad_dim = T.dynamic("s_pad_dim", "int32")

    @T.prim_func
    def dyn_copy_kernel(
        X: T.Tensor((total_elements,), "float16"),
        Y: T.Tensor((total_elements,), "float16"),
    ):
        with T.Kernel(56) as cid:
            x_ub = T.alloc_shared((256,), "float16")
            n_iters = (total_elements + 56 * 256 - 1) // (56 * 256)
            for it in T.serial(n_iters):
                off = (it * 56 + cid) * 256
                cur = T.min(256, T.max(0, total_elements - off))
                T.copy(X[off : off + cur], x_ub[:cur])
                T.copy(x_ub[:cur], Y[off : off + cur])
            _pad_guard = s_pad_dim

    res = inspect_tilelang_primfunc(
        dyn_copy_kernel,
        target=AscendTarget("ascend.950pr"),
        var_bindings={"M": 64, "D": 256, "S": 100},
    )
    mte2_cycles = max(
        res.ledger.pipe_cycles.get("AIV0.PIPE_MTE2", 0),
        res.ledger.pipe_cycles.get("AIV1.PIPE_MTE2", 0),
    )
    assert mte2_cycles > 0
    assert int(res.ledger.microarch_profile.get("gm_to_ub_bytes", 0)) > 0


def test_ub_scalar_compute_loops_propagates_to_cost_features_and_projection() -> None:
    """Propagate ub_scalar_compute_loops from ledger to 1.0 FLOP/cycle scalar penalty."""
    target = AscendTarget("ascend.950pr")
    model = HybridAscendCostModel(target)
    axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")

    clean_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 4000, "AIV0.PIPE_MTE2": 200},
        waterlines={},
        grid_blocks=56,
        gm_roundtrip_bytes=65536,
        dominant_contig_bytes=32768,
        ub_scalar_compute_loops=0,
    )
    scalar_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 4000, "AIV0.PIPE_MTE2": 200},
        waterlines={},
        grid_blocks=56,
        gm_roundtrip_bytes=65536,
        dominant_contig_bytes=32768,
        ub_scalar_compute_loops=2,
    )

    cf_clean = build_cost_features_from_ledger(clean_ledger, target=target, axes=axes)
    cf_scalar = build_cost_features_from_ledger(scalar_ledger, target=target, axes=axes)
    assert "scalar_compute_loop_fallback" not in cf_clean.extra_signals
    assert cf_scalar.extra_signals.get("scalar_compute_loop_fallback") == pytest.approx(2.0)

    pred_cf_clean = model.predict(cf_clean)
    pred_cf_scalar = model.predict(cf_scalar)
    assert pred_cf_scalar.vec_floor_us == pytest.approx(pred_cf_clean.vec_floor_us * 128.0)
    assert pred_cf_scalar.pred_us > pred_cf_clean.pred_us * 10.0

    wl_clean = project_case_workload(
        clean_ledger,
        case_dims={"M": 512, "N": 128},
        axes=axes,
        target=target,
    )
    wl_scalar = project_case_workload(
        scalar_ledger,
        case_dims={"M": 512, "N": 128},
        axes=axes,
        target=target,
    )
    assert wl_scalar["extra_signals"] == {"scalar_compute_loop_fallback": 2.0}

    pred_wl_clean = model.predict(wl_clean)
    pred_wl_scalar = model.predict(wl_scalar)
    assert pred_wl_scalar.vec_floor_us == pytest.approx(pred_wl_clean.vec_floor_us * 128.0)
    assert pred_wl_scalar.pred_us > pred_wl_clean.pred_us * 10.0

    cf_mapped = build_cost_features_from_ledger(
        {
            "grid_blocks": 56,
            "pipe_cycles": {"AIV0.PIPE_V": 4000},
            "gm_roundtrip_bytes": 65536,
            "ub_scalar_compute_loops": 3,
        },
        target=target,
        live_plan={"extra_signals": {"scalar_compute_loop_fallback": 1.0}},
        axes=axes,
    )
    assert cf_mapped.extra_signals.get("scalar_compute_loop_fallback") == pytest.approx(3.0)
