"""Tests for RegimePiecewiseSolver, bisheng_res_parser, and HybridAscendCostModel."""

from __future__ import annotations

import pathlib

import pytest

from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.hir.pipeline import PipelineSpec
from tilefoundry.ir.tir.memory import AllocTensor
from tilefoundry.ir.tir.stmts import Evaluate, For, LetStmt, Sequential
from tilefoundry.ir.types import DType, StorageKind, TensorType
from tilefoundry.ir.types.dim import DimVar
from tilefoundry_ascend.ascend.cost_model import (
    BishengResourceSummary,
    HybridAscendCostModel,
    parse_bisheng_res_usage,
    predict_cannbench_eq3_score,
)
from tilefoundry_ascend.ascend.ir.tir.instructions import AscendGmToUb, AscendUbToGm
from tilefoundry_ascend.ascend.schedule.regime_solver import RegimePiecewiseSolver
from tilefoundry_ascend.ascend.sil import (
    AdvisoryEnvelope,
    HardwareLegalityBounds,
    PolicyEnvelope,
    RegimeSlice,
    SILVerifier,
    SchedulePolicy,
)
from tilefoundry_ascend.ascend.target import AscendTarget


def test_regime_solver_static_affine_loop_projects_to_z3():
    """Verify static-affine / dim-substitutable TIR For loops project concrete dim_hints to Z3."""
    d_dyn = DimVar("D", 32, 2048)
    gm_ty = TensorType((4, d_dyn), DType.f32, None, StorageKind.GMEM)
    ub_ty = TensorType((d_dyn,), DType.f32, None, StorageKind.SMEM)
    sc = TensorType.umat_scalar()
    k = Var("k", type=sc)
    gm_in = Var("gm_in", type=gm_ty)
    gm_out = Var("gm_out", type=gm_ty)
    ub_var = Var("ub_buf", type=ub_ty)

    loop = For(
        k,
        Constant(0, type=sc),
        Constant(4, type=sc),
        Constant(1, type=sc),
        Sequential(
            (
                LetStmt(ub_var, Call(AllocTensor(tensor_type=ub_ty), (), type=ub_ty)),
                Evaluate(AscendGmToUb(), (gm_in, ub_var)),
                Evaluate(AscendUbToGm(), (ub_var, gm_out)),
            )
        ),
        pipeline=PipelineSpec(max_stage=1, max_buffer_versions=2, enable_offset=True),
    )

    from tilefoundry_ascend.ascend.sil.schema import TopologyAxes

    solver = RegimePiecewiseSolver()
    simd_axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
    res = solver.solve_regime(
        {
            "regime_tag": "large_row",
            "representative_dims": {"M": 4, "N": 1024, "D": 1024},
            "has_unaligned_tail": False,
        },
        op_name="rms_norm",
        topology_axes=simd_axes,
        loop=loop,
    )
    assert res.solver_mode == "z3_projected"
    assert res.ii is not None and res.ii >= 1
    assert res.buffer_versions["ub_buf"] == 2
    sil_pol = SchedulePolicy.from_dict(res.to_dict())
    assert sil_pol.regime_tag == "large_row"


def test_regime_solver_ragged_recurrent_vm_and_sil_verifier():
    """Verify analytical piecewise planning for all 5 ragged/recurrent/VM ops passes SILVerifier."""
    from tilefoundry_ascend.ascend.sil.schema import TopologyAxes

    solver = RegimePiecewiseSolver()
    verifier = SILVerifier()
    cv_ring_axes = TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "irregular_simt")
    grouped_axes = TopologyAxes("coupled_1c2v", "l0c_ub_epilogue", "ragged_group")
    kda_axes = TopologyAxes("coupled_1c2v", "persistent_onchip_state", "affine_simd")
    sinkhorn_axes = TopologyAxes("aiv_only", "persistent_onchip_state", "affine_simd")
    vm_axes = TopologyAxes("aiv_only", "persistent_onchip_state", "dynamic_opcode_stream")

    # 1. sparse_flash_attention (bidirectional_cv_ring, Cube-bound -> M >= 82, Half-L1, L0C->UB, UB->L1)
    sfa_reg = RegimeSlice(
        regime_tag="prefill_sparse",
        case_nums=(1, 2),
        representative_dims={"B": 2, "M": 512, "S_kv": 4096, "D": 128},
        has_unaligned_tail=True,
    )
    sfa_res = solver.solve_regime(
        sfa_reg,
        op_name="sparse_flash_attention",
        topology_axes=cv_ring_axes,
        roofline={"dominant_bound": "cube", "arithmetic_intensity": 256.0},
    )
    assert sfa_res.solver_mode == "analytical_piecewise"
    assert sfa_res.policy["tile_mnk"][0] >= 82
    assert sfa_res.policy["l1_half_bank_split"] is True
    assert "l0c_to_ub_split_m" in sfa_res.policy["cv_datapath"]
    assert "ub_to_l1_nd2nz" in sfa_res.policy["cv_datapath"]
    assert sfa_res.policy["ub_bank_pad_datablocks"] >= 1
    assert sfa_res.policy["simt_threads"] == 512

    # 2. grouped_matmul (l0c_ub_epilogue, empty group + unaligned tail)
    gmm_reg = RegimeSlice(
        regime_tag="ragged_empty",
        case_nums=(3,),
        representative_dims={"M": 192, "N": 256, "K": 128},
        has_unaligned_tail=True,
        has_empty_group=True,
    )
    gmm_res = solver.solve_regime(
        gmm_reg,
        op_name="grouped_matmul",
        topology_axes=grouped_axes,
        roofline={"dominant_bound": "cube"},
    )
    assert gmm_res.policy["ensure_empty_subblock_sync"] is True
    assert gmm_res.policy["l1_half_bank_split"] is True
    assert gmm_res.policy["tile_mnk"][0] >= 82

    # 3. kimi_delta_attention (persistent_onchip_state -> L1/UB state residency)
    kda_reg = RegimeSlice(
        regime_tag="chunk_64",
        case_nums=(4,),
        representative_dims={"B": 4, "S": 1024, "D_k": 128, "D_v": 128},
        has_unaligned_tail=False,
    )
    kda_res = solver.solve_regime(
        kda_reg,
        op_name="kimi_delta_attention",
        topology_axes=kda_axes,
        has_recurrence=True,
    )
    assert kda_res.persistent_state_space in ("L1", "UB")
    assert kda_res.persistent_state_bytes == 128 * 128 * 4

    # 4. mhc_sinkhorn (persistent FP32 UB state residency, inf/nan safe shifted exp)
    sink_reg = RegimeSlice(
        regime_tag="sinkhorn_4x4",
        case_nums=(5,),
        representative_dims={"M": 32, "N": 64},
        has_unaligned_tail=False,
    )
    sink_res = solver.solve_regime(
        sink_reg,
        op_name="mhc_sinkhorn",
        topology_axes=sinkhorn_axes,
        has_nan_inf_cases=True,
    )
    assert sink_res.persistent_state_space == "UB"
    assert sink_res.policy["fast_unshifted_exp"] is False

    # 5. tensor_program_vm (dynamic_opcode_stream -> horizontal W-slice, zero sync)
    vm_reg = RegimeSlice(
        regime_tag="vm_striped",
        case_nums=(6,),
        representative_dims={"R": 16, "W": 4096},
        has_unaligned_tail=True,
    )
    vm_res = solver.solve_regime(
        vm_reg,
        op_name="tensor_program_vm",
        topology_axes=vm_axes,
    )
    assert vm_res.persistent_state_space == "UB"
    assert vm_res.policy["sync_mode"] == "none"
    assert vm_res.policy["direct_no_fusion_vf"] is True

    # Verify full SIL PolicyEnvelope roundtrip and SILVerifier pass for sparse_flash_attention
    plan = solver.solve_all_regimes(
        [sfa_reg],
        op_name="sparse_flash_attention",
        topology_axes=cv_ring_axes,
        roofline={"dominant_bound": "cube"},
    )
    pol_env = PolicyEnvelope.from_dict(plan.to_dict())
    advisory = AdvisoryEnvelope(
        op_name="sparse_flash_attention",
        cannbot_task="level4/sparse_flash_attention",
        level=4,
        topology_axes=cv_ring_axes,
        input_dtypes=("f16",),
        output_dtypes=("f16",),
        roofline={"dominant_bound": "cube"},
        structural_features={},
        regimes=(sfa_reg,),
        legality_bounds=HardwareLegalityBounds(),
    )
    report = verifier.verify_envelope(pol_env, advisory)
    assert report.passed is True, report.errors


def test_bisheng_res_parser_extracts_regs_spills_loops_and_membar():
    """Verify bisheng --cce-res-usage and .s / dump parsing."""
    res_output = """
    Function: fused_rmsnorm_kernel
    Vector Register Usage: 48
    Scalar Register Usage: 36
    Stack Frame Size: 256 bytes
    Register Spill: 128 bytes
    """
    asm_vloopv2 = """
    .text
    // __simd_vf__ kernel
    VLOOPv2 r0, 64
    VADD v0, v1, v2
    """
    parsed_good = parse_bisheng_res_usage(res_output, asm_vloopv2)
    assert parsed_good == BishengResourceSummary(
        vec_regs=48,
        scalar_regs=36,
        spill_bytes=128,
        stack_bytes=256,
        vloopv2_loops=1,
        software_loops=0,
        has_local_membar_in_vf_call=False,
    )
    assert parsed_good.has_spill is True

    asm_sw_loop = """
    __simd_vf__ void vf_body() {
        LocalMemBar();
    }
    .LBB0_1:
        sub x0, x0, #1
        cbnz x0, .LBB0_1
    """
    parsed_bad = parse_bisheng_res_usage("VecRegs: 64\nScalarRegs: 64\nSpillSize: 2 KiB", asm_sw_loop)
    assert parsed_bad.spill_bytes == 2048
    assert parsed_bad.vloopv2_loops == 0
    assert parsed_bad.software_loops >= 1
    assert parsed_bad.has_local_membar_in_vf_call is True


def test_hybrid_cost_model_physics_penalties_calibration_and_ranking(tmp_path: pathlib.Path):
    """Verify Tier 0 physics, Tier 1 penalties, Tier 2 Ridge calibration + Eq.3/Eq.4, and ranking."""
    model = HybridAscendCostModel(AscendTarget("ascend.950pr"))

    # Tier 0: per-dtype Cube MAC scaling (FP4 > FP8 > FP16 > FP32)
    fp4_pred = model.predict({"cube_flops": 1e12, "cube_dtype": "f4e2m1", "total_tiles": 28})
    fp8_pred = model.predict({"cube_flops": 1e12, "cube_dtype": "fp8e4m3", "total_tiles": 28})
    fp16_pred = model.predict({"cube_flops": 1e12, "cube_dtype": "f16", "total_tiles": 28})
    fp32_pred = model.predict({"cube_flops": 1e12, "cube_dtype": "f32", "total_tiles": 28})
    assert fp4_pred.cube_floor_us < fp8_pred.cube_floor_us < fp16_pred.cube_floor_us < fp32_pred.cube_floor_us
    assert fp4_pred.tier2_residual_factor == pytest.approx(1.0)

    # Tier 0: Wave quantization (29 tiles on 28 AICs takes 2 waves vs 28 tiles 1 wave)
    wave1 = model.predict({"cube_flops": 1e11, "cube_dtype": "f16", "total_tiles": 28})
    wave2 = model.predict({"cube_flops": 1e11, "cube_dtype": "f16", "total_tiles": 29})
    assert wave2.wave_quantization_ratio > 1.8
    assert wave2.pred_us > wave1.pred_us * 1.5

    # Tier 0: Segmented HBM burst curve + SWAT L2 reuse
    small_burst = model.predict({"gm_bytes": 64 * 1024 * 1024, "contig_burst_bytes": 4096, "swizzle_window": 1})
    large_burst = model.predict({"gm_bytes": 64 * 1024 * 1024, "contig_burst_bytes": 32768, "swizzle_window": 1})
    swat_reuse = model.predict({"gm_bytes": 64 * 1024 * 1024, "contig_burst_bytes": 32768, "swizzle_window": 4, "l2_working_set_bytes": 32 * 1024 * 1024})
    assert small_burst.hbm_floor_us > large_burst.hbm_floor_us
    assert swat_reuse.l2_hit_rate > 0.0
    assert swat_reuse.hbm_floor_us < large_burst.hbm_floor_us

    # Tier 0 & Tier 1: Bank conflict + compiler spill + software loop ranking
    optimal = {
        "vec_flops": 1e10,
        "gm_bytes": 32 * 1024 * 1024,
        "contig_burst_bytes": 32768,
        "ub_row_stride_bytes": 544,
        "ub_bank_pad_datablocks": 1,
        "vloopv2_loops": 1,
        "software_loops": 0,
        "spill_bytes": 0,
    }
    bank_conflict = dict(optimal, ub_row_stride_bytes=512, ub_bank_pad_datablocks=0)
    spilling = dict(optimal)
    sw_loop = dict(optimal)

    p_opt = model.predict(optimal)
    p_bank = model.predict(bank_conflict)
    p_spill = model.predict(spilling, bisheng_res=BishengResourceSummary(spill_bytes=512, vloopv2_loops=1))
    p_sw = model.predict(sw_loop, bisheng_res=BishengResourceSummary(vloopv2_loops=0, software_loops=2))
    assert p_opt.pred_us < p_bank.pred_us
    assert p_opt.pred_us < p_spill.pred_us
    assert p_opt.pred_us < p_sw.pred_us

    # Tier 2: Online Ridge calibration + JSONL persistence + Eq.3/HAP scoring
    cal_path = tmp_path / "cal.jsonl"
    cal_model = HybridAscendCostModel(calibration_path=cal_path)
    base_pred = cal_model.predict(optimal).pred_us
    for _ in range(5):
        cal_model.observe(optimal, measured_us=base_pred * 1.4)
    adapted_pred = cal_model.predict(optimal).pred_us
    assert adapted_pred > base_pred * 1.15
    assert cal_path.is_file()

    reloaded = HybridAscendCostModel(calibration_path=cal_path)
    assert reloaded.sample_count == 5
    assert reloaded.predict(optimal).pred_us == pytest.approx(adapted_pred)

    eq3 = predict_cannbench_eq3_score(pred_us=10.0, ref_perf_us=20.0, t_hw_us=8.0)
    assert eq3.eq3_score == pytest.approx((20.0 - 8.0) / ((10.0 - 8.0) + (20.0 - 8.0)))
    assert eq3.hap_reward == pytest.approx(0.8)
    eq3_fast = cal_model.predict_cannbench_eq3_score(pred_us=2.0, ref_perf_us=20.0, t_hw_us=8.0)
    assert eq3_fast.hap_reward == pytest.approx(3.0)
