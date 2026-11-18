"""Tests for the Dual-Track SIL 7-Paradigm Template Library and Ascend 950PR operators."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import torch

from tilefoundry_ascend.ascend.schedule.envelope_inspector import (
    inspect_tilelang_primfunc,
)
from tilefoundry_ascend.ascend.templates import (
    CANONICAL_TOPOLOGY_PRESETS,
    SCHEDULE_POLICY_KEYS,
    SHELL_AIC_GEMM_FIXPIPE,
    SHELL_AIV_SIMD_STREAM,
    SHELL_AIV_SIMT_ATOMIC,
    SHELL_AIV_STRIPED_VM,
    SHELL_CV_PERSISTENT_STATE,
    SHELL_CV_RAGGED_EPILOGUE,
    SHELL_CV_RING_PIPELINE,
    TopologyAxes,
    get_template_registry,
    normalize_schedule_policy,
    render_template,
)

def _resolve_cannbot_ops_root() -> Path:
    env_repo = os.environ.get("TILEFOUNDRY_NPU_CANNBENCH_REPO", "").strip()
    if env_repo:
        cand = Path(env_repo).expanduser()
        return (cand / "operators") if (cand / "operators").is_dir() else cand
    return Path(__file__).resolve().parents[3].parent / "cannbot-ops" / "operators"


CANNBOT_OPS_ROOT = _resolve_cannbot_ops_root()


def _load_golden(level: str, op_name: str):
    path = CANNBOT_OPS_ROOT / level / op_name / "golden.py"
    spec = importlib.util.spec_from_file_location(f"_golden_{op_name}", str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, op_name)


def test_template_registry_covers_all_canonical_axes_and_policy_knobs():
    reg = get_template_registry()
    assert reg.list_topology_axes() == CANONICAL_TOPOLOGY_PRESETS
    assert len(CANONICAL_TOPOLOGY_PRESETS) == 7

    for axes in CANONICAL_TOPOLOGY_PRESETS:
        default_pol = reg.default_policy(axes)
        for key in SCHEDULE_POLICY_KEYS:
            assert key in default_pol, f"Missing SchedulePolicy key {key} in {axes.key()}"


@pytest.mark.parametrize("axes", CANONICAL_TOPOLOGY_PRESETS)
def test_all_canonical_shell_templates_lower_stage_a_and_render_ascendc(axes: TopologyAxes):
    policy_overrides = {
        SHELL_AIV_SIMD_STREAM: {"tile_mnk": (4, 128, 1), "row_pack_factor": 2, "multi_row_unroll": 2, "ub_256kb_unlock": True, "ub_bank_pad_datablocks": 1, "dim_size": 193, "dtype": "bfloat16"},
        SHELL_AIV_SIMT_ATOMIC: {"simt_threads": 512, "ub_bank_pad_datablocks": 8, "tile_mnk": (8, 64, 1)},
        SHELL_AIC_GEMM_FIXPIPE: {"tile_mnk": (128, 128, 64), "swizzle_window": 4, "l1_stages": 1, "l1_half_bank_split": True, "sync_mode": "unit_flag", "w4_mode": "mxa8w4"},
        SHELL_CV_RAGGED_EPILOGUE: {"tile_mnk": (64, 128, 64), "l1_stages": 2, "cv_datapath": "direct_ub"},
        SHELL_CV_RING_PIPELINE: {"tile_mnk": (128, 64, 64), "l1_stages": 2, "dim_size": 128, "cv_datapath": "ub_l1_ring"},
        SHELL_CV_PERSISTENT_STATE: {"dim_size": 64, "ub_256kb_unlock": True},
        SHELL_AIV_STRIPED_VM: {"tile_mnk": (16, 128, 1), "ub_256kb_unlock": True, "dtype": "bfloat16"},
    }
    rendered = render_template(axes, policy_overrides[axes], track="both")
    assert rendered.prim_func is not None
    assert rendered.ascendc_cpp

    inspection = inspect_tilelang_primfunc(
        rendered.prim_func,
        pass_configs=rendered.pass_configs,
        compile_flags=rendered.compile_flags,
    )
    assert inspection.lowering_error is None, f"{axes.key()} failed Stage-A lowering: {inspection.lowering_error}"
    assert inspection.ledger.is_legal, f"{axes.key()} ledger illegal: max_overflow={inspection.ledger.max_overflow_bytes}"


def test_ascendc_templates_reflect_policy_knobs_and_hardware_idioms():
    cpp_reg = render_template(SHELL_AIV_SIMD_STREAM, {"ub_bank_pad_datablocks": 1, "multi_row_unroll": 2}, track="ascendc").ascendc_cpp
    assert "__no_simd_vf_fusion__ __simd_vf__" in cpp_reg
    assert "DIST_UNPACK_B16" in cpp_reg and "DIST_PACK_B32" in cpp_reg
    assert "ReduceDataBlock" in cpp_reg

    cpp_simt = render_template(SHELL_AIV_SIMT_ATOMIC, {"simt_threads": 512, "ub_bank_pad_datablocks": 8}, track="ascendc").ascendc_cpp
    assert "__simt_vf__ __launch_bounds__(512)" in cpp_simt
    assert "asc_atomic_cas" in cpp_simt and "__shfl" in cpp_simt and "Simt::UintDiv" in cpp_simt
    assert "512U" in cpp_simt

    cpp_gemm = render_template(SHELL_AIC_GEMM_FIXPIPE, {"swizzle_window": 4, "l1_stages": 1, "sync_mode": "unit_flag", "w4_mode": "w4a16"}, track="ascendc").ascendc_cpp
    assert "windowLen = 4U" in cpp_gemm
    assert "(TOTAL_L1_SIZE >> 1)" in cpp_gemm
    assert "CACHE_MODE_DISABLE" in cpp_gemm
    assert "FIXPIPE_DEQ_SCALE = 64U" in cpp_gemm

    cpp_grp = render_template(SHELL_CV_RAGGED_EPILOGUE, {}, track="ascendc").ascendc_cpp
    assert "(endBlockIdx + 1U) % coreNum" in cpp_grp
    assert "CrossCoreSetFlag" in cpp_grp and "dualDstCtl = 0b01U" in cpp_grp

    cpp_cv = render_template(SHELL_CV_RING_PIPELINE, {"tile_mnk": (96, 112, 64), "l1_stages": 3}, track="ascendc").ascendc_cpp
    assert "CUBE_TILE_M = 96U" in cpp_cv and "CUBE_TILE_N = 112U" in cpp_cv
    assert "Nd2NzParams" in cpp_cv

    cpp_pcs = render_template(SHELL_CV_PERSISTENT_STATE, {}, track="ascendc").ascendc_cpp
    assert "0.5f * gTail" in cpp_pcs
    assert "cmatrixInitVal = false" in cpp_pcs

    cpp_vm = render_template(SHELL_AIV_STRIPED_VM, {"tile_mnk": (32, 256, 1)}, track="ascendc").ascendc_cpp
    assert "VM_NUM_REGS = 32U" in cpp_vm and "VM_W_SLICE = 256U" in cpp_vm
    assert "LocalMemBar" in cpp_vm

    norm = normalize_schedule_policy(SHELL_AIV_SIMD_STREAM, {"row_pack_factor": 8})
    assert norm["row_pack_factor"] == 8


@pytest.mark.parametrize(
    "op_module,cfg_name,builder_name,cases",
    [
        (
            "examples.ascend950pr_ops.swi_glu.kernels",
            "_select_swi_glu_config",
            "_build_swi_glu_prim_func",
            [
                {"half_dim": 1024, "tl_dtype": "bfloat16", "schedule_mode": "tuned"},
                {"half_dim": 1009, "tl_dtype": "float16", "schedule_mode": "tuned"},
                {"half_dim": 67, "tl_dtype": "float32", "schedule_mode": "baseline"},
            ],
        ),
        (
            "examples.ascend950pr_ops.rms_norm.kernels",
            "_select_rms_norm_config",
            "_build_rms_norm_prim_func",
            [
                {"d_dim": 1024, "tl_dtype": "bfloat16", "schedule_mode": "tuned"},
                {"d_dim": 769, "tl_dtype": "float16", "schedule_mode": "tuned"},
                {"d_dim": 373, "tl_dtype": "float32", "schedule_mode": "tuned"},
            ],
        ),
        (
            "examples.ascend950pr_ops.softmax.kernels",
            "_select_softmax_config",
            "_build_softmax_prim_func",
            [
                {"d_dim": 1024, "tl_dtype": "float16", "schedule_mode": "tuned"},
                {"d_dim": 1021, "tl_dtype": "bfloat16", "schedule_mode": "tuned"},
                {"d_dim": 101, "tl_dtype": "float32", "schedule_mode": "tuned"},
            ],
        ),
        (
            "examples.ascend950pr_ops.add_rms_norm_dynamic_quant.kernels",
            "_select_add_rms_norm_dynamic_quant_config",
            "_build_add_rms_norm_dynamic_quant_prim_func",
            [
                {"d_dim": 2048, "tl_dtype": "bfloat16", "schedule_mode": "tuned"},
                {"d_dim": 2053, "tl_dtype": "float16", "schedule_mode": "tuned"},
            ],
        ),
        (
            "examples.ascend950pr_ops.mhc_sinkhorn.kernels",
            "_select_mhc_sinkhorn_config",
            "_build_mhc_sinkhorn_prim_func",
            [
                {"hc_mult": 4, "iter_step": 20, "schedule_mode": "tuned"},
                {"hc_mult": 3, "iter_step": 5, "schedule_mode": "tuned"},
                {"hc_mult": 16, "iter_step": 1, "schedule_mode": "tuned"},
            ],
        ),
        (
            "examples.ascend950pr_ops.gqa.kernels",
            "_select_gqa_config",
            "_build_gqa_prim_func",
            [
                {"s_dim": 128, "n_q": 32, "n_kv": 8, "d_dim": 128, "tl_dtype": "float16", "is_causal": True},
                {"s_dim": 1, "n_q": 32, "n_kv": 8, "d_dim": 128, "tl_dtype": "bfloat16", "is_causal": False},
            ],
        ),
        (
            "examples.ascend950pr_ops.kimi_delta_attention.kernels",
            "_select_kimi_delta_attention_config",
            "_build_kimi_delta_attention_prim_func",
            [
                {"dk": 128, "dv": 128, "tl_dtype": "bfloat16", "use_qk_l2norm": True},
                {"dk": 33, "dv": 31, "tl_dtype": "float32", "use_qk_l2norm": False},
            ],
        ),
        (
            "examples.ascend950pr_ops.tensor_program_vm.kernels",
            "_select_tensor_program_vm_config",
            "_build_tensor_program_vm_prim_func",
            [
                {"num_regs": 16, "w_dim": 1024, "tl_dtype": "bfloat16"},
                {"num_regs": 8, "w_dim": 256, "tl_dtype": "float32"},
            ],
        ),
    ],
)
def test_operator_prim_funcs_lower_cleanly_stage_a(op_module, cfg_name, builder_name, cases):
    mod = __import__(op_module, fromlist=[cfg_name, builder_name])
    selector = getattr(mod, cfg_name)
    builder = getattr(mod, builder_name)
    for kw in cases:
        cfg = selector(**kw)
        prog = builder(**cfg)
        res = inspect_tilelang_primfunc(
            prog,
            pass_configs=cfg.get("pass_configs"),
            compile_flags=cfg.get("compile_flags"),
        )
        assert res.lowering_error is None, f"{op_module} {kw} failed: {res.lowering_error}"
        assert res.ledger.is_legal, f"{op_module} {kw} overflow: {res.ledger.max_overflow_bytes}"


def test_swi_glu_parity_with_golden():
    from examples.ascend950pr_ops.swi_glu import kernel as swi_glu_kernel

    golden = _load_golden("level1", "swi_glu")
    for shape, dtype, dim in [
        ((16, 2048), torch.float16, -1),
        ((17, 2018), torch.float32, -1),
        ((12, 17), torch.bfloat16, 0),
        ((11, 13, 16, 67), torch.float32, 2),
    ]:
        x = torch.randn(shape, dtype=dtype)
        ref = golden(x, dim=dim)
        out = swi_glu_kernel(x, dim=dim)
        assert out.dtype == ref.dtype and out.shape == ref.shape
        torch.testing.assert_close(out.float(), ref.float(), rtol=1e-3, atol=1e-3)

    # inf and nan edge cases
    x_inf = torch.tensor([[float("inf"), float("-inf")], [float("-inf"), 2.0]], dtype=torch.bfloat16)
    torch.testing.assert_close(swi_glu_kernel(x_inf), golden(x_inf), equal_nan=True)
    x_nan = torch.full((4, 8), float("nan"), dtype=torch.float32)
    assert torch.isnan(swi_glu_kernel(x_nan)).all()


def test_rms_norm_parity_with_golden():
    from examples.ascend950pr_ops.rms_norm import kernel as rms_norm_kernel

    golden = _load_golden("level2", "rms_norm")
    for shape, dtype, eps in [
        ((4, 17, 769), torch.float16, 1e-6),
        ((3, 19, 373), torch.float32, 1e-5),
        ((2, 13, 1024), torch.bfloat16, 1e-12),
    ]:
        x = torch.randn(shape, dtype=dtype)
        g = torch.randn(shape[-1], dtype=dtype)
        ref = golden(x, g, epsilon=eps)
        out = rms_norm_kernel(x, g, epsilon=eps)
        assert out.dtype == ref.dtype and out.shape == ref.shape
        torch.testing.assert_close(out.float(), ref.float(), rtol=1e-3, atol=1e-3)


def test_softmax_parity_with_golden():
    from examples.ascend950pr_ops.softmax import kernel as softmax_kernel

    golden = _load_golden("level2", "softmax")
    for shape, dtype, dim in [
        ((17, 1009), torch.float16, -1),
        ((13, 17, 101), torch.float32, 1),
        ((7, 19, 127), torch.bfloat16, -2),
    ]:
        x = torch.randn(shape, dtype=dtype)
        ref = golden(x, dim=dim)
        out = softmax_kernel(x, dim=dim)
        assert out.dtype == ref.dtype and out.shape == ref.shape
        torch.testing.assert_close(out.float(), ref.float(), rtol=1e-3, atol=1e-3)

    x_inf = torch.tensor([[float("-inf"), float("inf")], [1.0, float("-inf")]], dtype=torch.float16)
    torch.testing.assert_close(softmax_kernel(x_inf), golden(x_inf), equal_nan=True)


def test_add_rms_norm_dynamic_quant_parity_with_golden():
    from examples.ascend950pr_ops.add_rms_norm_dynamic_quant import (
        kernel as armdq_kernel,
    )

    golden = _load_golden("level3", "add_rms_norm_dynamic_quant")
    for shape, dtype, eps in [
        ((17, 2053), torch.float16, 1e-5),
        ((13, 1023), torch.bfloat16, 1e-6),
        ((8, 256), torch.float16, 1e-6),
    ]:
        x1 = torch.randn(shape, dtype=dtype) * 10.0
        x2 = torch.randn(shape, dtype=dtype) * 10.0
        gamma = torch.randn(shape[-1], dtype=dtype)
        ref_y, ref_xout, ref_scale = golden(x1, x2, gamma, epsilon=eps)
        out_y, out_xout, out_scale = armdq_kernel(x1, x2, gamma, epsilon=eps)
        assert out_y.dtype == torch.int8 and out_y.shape == ref_y.shape
        assert out_xout.dtype == ref_xout.dtype and out_scale.dtype == torch.float32
        assert (out_y.int() - ref_y.int()).abs().max().item() <= 1
        torch.testing.assert_close(out_xout.float(), ref_xout.float(), rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(out_scale, ref_scale, rtol=1e-4, atol=1e-6)


def test_mhc_sinkhorn_parity_with_golden():
    from examples.ascend950pr_ops.mhc_sinkhorn import kernel as sinkhorn_kernel

    golden = _load_golden("level3", "mhc_sinkhorn")
    for b, hc, steps, eps in [(8, 4, 20, 1e-6), (13, 3, 5, 1e-7), (4, 16, 1, 1e-8)]:
        comb = torch.randn(b, hc, hc, dtype=torch.float32)
        ref = golden(comb, iter_step=steps, eps=eps)
        out = sinkhorn_kernel(comb, iter_step=steps, eps=eps)
        assert out.dtype == torch.float32 and out.shape == comb.shape
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_gqa_parity_with_golden():
    from examples.ascend950pr_ops.gqa import kernel as gqa_kernel

    golden = _load_golden("level4", "gqa")
    for (b, s, s_kv, n_q, n_kv, d), dtype, causal, scale in [
        ((2, 16, 16, 8, 2, 64), torch.float16, True, -1.0),
        ((2, 12, 20, 12, 4, 64), torch.bfloat16, False, -1.0),
        ((2, 1, 32, 8, 2, 64), torch.float16, False, 0.125),
    ]:
        q = torch.randn(b, s, n_q, d, dtype=dtype)
        k = torch.randn(b, s_kv, n_kv, d, dtype=dtype)
        v = torch.randn(b, s_kv, n_kv, d, dtype=dtype)
        ref = golden(q, k, v, scaleValue=scale, is_causal=causal)
        out = gqa_kernel(q, k, v, scaleValue=scale, is_causal=causal)
        assert out.dtype == ref.dtype and out.shape == ref.shape
        torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


def test_kimi_delta_attention_parity_with_golden():
    from examples.ascend950pr_ops.kimi_delta_attention import kernel as kda_kernel

    golden = _load_golden("level5", "kimi_delta_attention")
    for (b, t, h, dk, dv), dtype, l2norm, scale in [
        ((1, 17, 3, 33, 31), torch.float32, True, -1.0),
        ((2, 12, 4, 64, 64), torch.bfloat16, True, -1.0),
        ((1, 8, 2, 16, 24), torch.float16, False, 0.25),
    ]:
        q = torch.randn(b, t, h, dk, dtype=dtype) * 0.2
        k = torch.randn(b, t, h, dk, dtype=dtype) * 0.2
        v = torch.randn(b, t, h, dv, dtype=dtype) * 0.2
        g = -torch.rand(b, t, h, dk, dtype=torch.float32) * 0.2
        beta = torch.rand(b, t, h, dtype=torch.float32) * 0.8 + 0.1
        init_state = torch.randn(b, h, dk, dv, dtype=torch.float32) * 0.1
        init_copy = init_state.clone()

        ref_y, ref_s = golden(q, k, v, g, beta, init_state, use_qk_l2norm=l2norm, scaleValue=scale)
        out_y, out_s = kda_kernel(q, k, v, g, beta, init_state, use_qk_l2norm=l2norm, scaleValue=scale)
        # initial_state must not be modified in place
        torch.testing.assert_close(init_state, init_copy)
        assert out_y.dtype == ref_y.dtype and out_s.dtype == ref_s.dtype
        torch.testing.assert_close(out_y.float(), ref_y.float(), rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(out_s, ref_s, rtol=1e-2, atol=1e-2)


def test_tensor_program_vm_parity_with_golden():
    from examples.ascend950pr_ops.tensor_program_vm import kernel as vm_kernel

    golden = _load_golden("level5", "tensor_program_vm")
    for p_len, r_cnt, w_dim, dtype in [
        (64, 8, 17, torch.float32),
        (128, 16, 256, torch.bfloat16),
        (32, 4, 128, torch.float16),
    ]:
        program = torch.randint(0, 1_000_000_000, (p_len, 4), dtype=torch.int32)
        init_regs = torch.randn(r_cnt, w_dim, dtype=dtype)
        ref = golden(program, init_regs)
        out = vm_kernel(program, init_regs)
        assert out.dtype == ref.dtype and out.shape == ref.shape
        torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)
