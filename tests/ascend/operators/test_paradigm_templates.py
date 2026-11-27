"""Tests for the Dual-Track SIL 7-Paradigm Template Library and Ascend 950PR operators."""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path

import pytest
import torch

from examples.ascend950pr_ops.add_rms_norm_dynamic_quant import kernel as armdq_kernel
from examples.ascend950pr_ops.gqa import kernel as gqa_kernel
from examples.ascend950pr_ops.kimi_delta_attention import kernel as kda_kernel
from examples.ascend950pr_ops.mhc_sinkhorn import kernel as sinkhorn_kernel
from examples.ascend950pr_ops.rms_norm import kernel as rms_norm_kernel
from examples.ascend950pr_ops.softmax import kernel as softmax_kernel
from examples.ascend950pr_ops.swi_glu import kernel as swi_glu_kernel
from examples.ascend950pr_ops.tensor_program_vm import kernel as vm_kernel
from tilefoundry_ascend.ascend.schedule.envelope_inspector import inspect_tilelang_primfunc
from tilefoundry_ascend.ascend.target.ast_semantic_gate import check_tilelang_semantic_contracts
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


def _resolve_cannbench_tasks_root() -> Path:
    env_repo = os.environ.get("TILEFOUNDRY_NPU_CANNBENCH_REPO", "").strip()
    if env_repo:
        cand = Path(env_repo).expanduser()
        return (cand / "tasks") if (cand / "tasks").is_dir() else cand
    return Path(__file__).resolve().parents[3].parent / "cann-bench" / "tasks"


CANNBENCH_TASKS_ROOT = _resolve_cannbench_tasks_root()


def _load_golden(level: str, op_name: str):
    path = CANNBENCH_TASKS_ROOT / level / op_name / "golden.py"
    spec = importlib.util.spec_from_file_location(f"_golden_{op_name}", str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, op_name)


def test_template_registry_covers_all_canonical_axes_and_policy_knobs() -> None:
    """Verify all 7 canonical shell templates expose complete SchedulePolicy keys."""
    reg = get_template_registry()
    assert reg.list_topology_axes() == CANONICAL_TOPOLOGY_PRESETS
    assert len(CANONICAL_TOPOLOGY_PRESETS) == 7
    for axes in CANONICAL_TOPOLOGY_PRESETS:
        default_pol = reg.default_policy(axes)
        for key in SCHEDULE_POLICY_KEYS:
            assert key in default_pol


@pytest.mark.parametrize("axes", CANONICAL_TOPOLOGY_PRESETS)
def test_all_canonical_shell_templates_lower_stage_a_and_render_ascendc(
    axes: TopologyAxes,
) -> None:
    """Lower all 7 paradigm shells through Stage-A and verify Track-B C++20 rendering."""
    overrides = {
        SHELL_AIV_SIMD_STREAM: {
            "tile_mnk": (4, 128, 1),
            "row_pack_factor": 2,
            "multi_row_unroll": 2,
            "ub_256kb_unlock": True,
            "ub_bank_pad_datablocks": 1,
            "dim_size": 193,
            "dtype": "bfloat16",
        },
        SHELL_AIV_SIMT_ATOMIC: {"simt_threads": 512, "ub_bank_pad_datablocks": 8, "tile_mnk": (8, 64, 1)},
        SHELL_AIC_GEMM_FIXPIPE: {
            "tile_mnk": (128, 128, 64),
            "swizzle_window": 4,
            "l1_stages": 1,
            "l1_half_bank_split": True,
            "sync_mode": "unit_flag",
            "w4_mode": "mxa8w4",
        },
        SHELL_CV_RAGGED_EPILOGUE: {"tile_mnk": (64, 128, 64), "l1_stages": 2, "cv_datapath": "direct_ub"},
        SHELL_CV_RING_PIPELINE: {"tile_mnk": (128, 64, 64), "l1_stages": 2, "dim_size": 128, "cv_datapath": "ub_l1_ring"},
        SHELL_CV_PERSISTENT_STATE: {"dim_size": 64, "ub_256kb_unlock": True},
        SHELL_AIV_STRIPED_VM: {"tile_mnk": (16, 128, 1), "ub_256kb_unlock": True, "dtype": "bfloat16"},
    }
    rendered = render_template(axes, overrides[axes], track="both")
    assert rendered.prim_func is not None and rendered.ascendc_cpp
    res = inspect_tilelang_primfunc(
        rendered.prim_func, pass_configs=rendered.pass_configs, compile_flags=rendered.compile_flags
    )
    assert res.lowering_error is None, f"{axes.key()}: {res.lowering_error}"
    assert res.ledger.is_legal, f"{axes.key()}: overflow={res.ledger.max_overflow_bytes}"


def test_ascendc_templates_reflect_policy_knobs_and_hardware_idioms() -> None:
    """Verify each AscendC paradigm template reflects policy knobs and hardware idioms."""
    cpp_reg = render_template(SHELL_AIV_SIMD_STREAM, {"ub_bank_pad_datablocks": 1, "multi_row_unroll": 2}, track="ascendc").ascendc_cpp
    assert "__no_simd_vf_fusion__ __simd_vf__" in cpp_reg
    assert "DIST_UNPACK_B16" in cpp_reg and "DIST_PACK_B32" in cpp_reg and "ReduceDataBlock" in cpp_reg

    cpp_simt = render_template(SHELL_AIV_SIMT_ATOMIC, {"simt_threads": 512, "ub_bank_pad_datablocks": 8}, track="ascendc").ascendc_cpp
    assert "<c_api/asc_cube.h>" not in cpp_simt and "SIMT_THREADS = 512U;" in cpp_simt
    assert "asc_atomic_cas" in cpp_simt and "asc_shfl(slot, 0)" in cpp_simt and "Simt::UintDiv" in cpp_simt

    cpp_gemm = render_template(SHELL_AIC_GEMM_FIXPIPE, {"swizzle_window": 4, "l1_stages": 1, "sync_mode": "unit_flag", "w4_mode": "w4a16"}, track="ascendc").ascendc_cpp
    assert "TOTAL_L1_SIZE = 524288U;" in cpp_gemm and "ascend_gemm_l1<" in cpp_gemm and "FIXPIPE_DEQ_SCALE = 64U" in cpp_gemm

    cpp_grp = render_template(SHELL_CV_RAGGED_EPILOGUE, {}, track="ascendc").ascendc_cpp
    assert "asc_sync_block_arrive" in cpp_grp and "asc_copy_l0c2ub" in cpp_grp

    cpp_cv = render_template(SHELL_CV_RING_PIPELINE, {"tile_mnk": (96, 112, 64), "l1_stages": 3}, track="ascendc").ascendc_cpp
    assert "CUBE_TILE_M = 96U" in cpp_cv and "asc_copy_ub2l1" in cpp_cv

    cpp_pcs = render_template(SHELL_CV_PERSISTENT_STATE, {}, track="ascendc").ascendc_cpp
    assert "0.5f * gTail" in cpp_pcs and "asc_sync_pipe(PIPE_V)" in cpp_pcs

    cpp_vm = render_template(SHELL_AIV_STRIPED_VM, {"tile_mnk": (32, 256, 1)}, track="ascendc").ascendc_cpp
    assert "VM_NUM_REGS = 32U" in cpp_vm and "LocalMemBar" in cpp_vm
    assert normalize_schedule_policy(SHELL_AIV_SIMD_STREAM, {"row_pack_factor": 8})["row_pack_factor"] == 8


@pytest.mark.parametrize(
    ("op_module", "cfg_name", "builder_name", "kw", "run_lower"),
    [
        ("examples.ascend950pr_ops.swi_glu.kernels", "_select_swi_glu_config", "_build_swi_glu_prim_func", {"half_dim": 1009, "tl_dtype": "float16", "schedule_mode": "tuned"}, True),
        ("examples.ascend950pr_ops.rms_norm.kernels", "_select_rms_norm_config", "_build_rms_norm_prim_func", {"d_dim": 769, "tl_dtype": "float16", "schedule_mode": "tuned"}, True),
        ("examples.ascend950pr_ops.softmax.kernels", "_select_softmax_config", "_build_softmax_prim_func", {"d_dim": 1021, "tl_dtype": "bfloat16", "schedule_mode": "tuned"}, True),
        ("examples.ascend950pr_ops.add_rms_norm_dynamic_quant.kernels", "_select_add_rms_norm_dynamic_quant_config", "_build_add_rms_norm_dynamic_quant_prim_func", {"d_dim": 2053, "tl_dtype": "float16", "schedule_mode": "tuned"}, True),
        ("examples.ascend950pr_ops.mhc_sinkhorn.kernels", "_select_mhc_sinkhorn_config", "_build_mhc_sinkhorn_prim_func", {"hc_mult": 4, "iter_step": 20, "schedule_mode": "tuned"}, True),
        ("examples.ascend950pr_ops.kimi_delta_attention.kernels", "_select_kimi_delta_attention_config", "_build_kimi_delta_attention_prim_func", {"dk": 128, "dv": 128, "tl_dtype": "bfloat16", "use_qk_l2norm": True}, True),
        ("examples.ascend950pr_ops.gqa.kernels", "_select_gqa_config", "_build_gqa_prim_func", {"s_dim": 128, "n_q": 32, "n_kv": 8, "d_dim": 128, "tl_dtype": "float16", "is_causal": True}, False),
        ("examples.ascend950pr_ops.tensor_program_vm.kernels", "_select_tensor_program_vm_config", "_build_tensor_program_vm_prim_func", {"num_regs": 16, "w_dim": 1024, "tl_dtype": "bfloat16"}, False),
    ],
)
def test_operator_prim_funcs_lower_cleanly_stage_a(
    op_module: str,
    cfg_name: str,
    builder_name: str,
    kw: dict[str, object],
    run_lower: bool,
) -> None:
    """Verify PrimFunc construction, AST gate contracts, and Stage-A legality across operators."""
    mod = importlib.import_module(op_module)
    cfg = getattr(mod, cfg_name)(**kw)
    prog = getattr(mod, builder_name)(**cfg)
    assert prog is not None
    assert check_tilelang_semantic_contracts(Path(mod.__file__).read_text(encoding="utf-8")) == ()
    if run_lower:
        res = inspect_tilelang_primfunc(
            prog, pass_configs=cfg.get("pass_configs"), compile_flags=cfg.get("compile_flags")
        )
        assert res.lowering_error is None, f"{op_module}: {res.lowering_error}"
        assert res.ledger.is_legal, f"{op_module}: overflow={res.ledger.max_overflow_bytes}"


def test_operator_cpu_reference_parity_with_golden() -> None:
    """Verify CPU fallback parity against CANN-Bench golden implementations for all 8 ops."""
    g_swi = _load_golden("level1", "swi_glu")
    for shape, dtype, dim in [((16, 2048), torch.float16, -1), ((11, 13, 16, 67), torch.float32, 2)]:
        x = torch.randn(shape, dtype=dtype)
        torch.testing.assert_close(swi_glu_kernel(x, dim=dim).float(), g_swi(x, dim=dim).float(), rtol=1e-3, atol=1e-3)
    x_inf = torch.tensor([[float("inf"), float("-inf")], [float("-inf"), 2.0]], dtype=torch.bfloat16)
    torch.testing.assert_close(swi_glu_kernel(x_inf), g_swi(x_inf), equal_nan=True)
    assert torch.isnan(swi_glu_kernel(torch.full((4, 8), float("nan"), dtype=torch.float32))).all()

    g_rms = _load_golden("level2", "rms_norm")
    for shape, dtype, eps in [((4, 17, 769), torch.float16, 1e-6), ((3, 19, 373), torch.float32, 1e-5)]:
        x, g = torch.randn(shape, dtype=dtype), torch.randn(shape[-1], dtype=dtype)
        torch.testing.assert_close(rms_norm_kernel(x, g, epsilon=eps).float(), g_rms(x, g, epsilon=eps).float(), rtol=1e-3, atol=1e-3)

    g_sm = _load_golden("level2", "softmax")
    for shape, dtype, dim in [((17, 1009), torch.float16, -1), ((13, 17, 101), torch.float32, 1)]:
        x = torch.randn(shape, dtype=dtype)
        torch.testing.assert_close(softmax_kernel(x, dim=dim).float(), g_sm(x, dim=dim).float(), rtol=1e-3, atol=1e-3)

    g_ar = _load_golden("level3", "add_rms_norm_dynamic_quant")
    x1, x2 = torch.randn((17, 2053), dtype=torch.float16) * 10.0, torch.randn((17, 2053), dtype=torch.float16) * 10.0
    gamma = torch.randn(2053, dtype=torch.float16)
    ry, rx, rs = g_ar(x1, x2, gamma, epsilon=1e-5)
    oy, ox, os_ = armdq_kernel(x1, x2, gamma, epsilon=1e-5)
    assert (oy.int() - ry.int()).abs().max().item() <= 1
    torch.testing.assert_close(ox.float(), rx.float(), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(os_, rs, rtol=1e-4, atol=1e-6)

    g_mhc = _load_golden("level3", "mhc_sinkhorn")
    comb = torch.randn(13, 3, 3, dtype=torch.float32)
    torch.testing.assert_close(sinkhorn_kernel(comb, iter_step=5, eps=1e-7), g_mhc(comb, iter_step=5, eps=1e-7), rtol=1e-5, atol=1e-5)

    g_gqa = _load_golden("level4", "gqa")
    q, k, v = torch.randn(2, 16, 8, 64, dtype=torch.float16), torch.randn(2, 16, 2, 64, dtype=torch.float16), torch.randn(2, 16, 2, 64, dtype=torch.float16)
    torch.testing.assert_close(gqa_kernel(q, k, v, scaleValue=-1.0, is_causal=True).float(), g_gqa(q, k, v, scaleValue=-1.0, is_causal=True).float(), rtol=2e-2, atol=2e-2)

    g_kda = _load_golden("level5", "kimi_delta_attention")
    q, k, v = torch.randn(1, 17, 3, 33, dtype=torch.float32) * 0.2, torch.randn(1, 17, 3, 33, dtype=torch.float32) * 0.2, torch.randn(1, 17, 3, 31, dtype=torch.float32) * 0.2
    g = -torch.rand(1, 17, 3, 33, dtype=torch.float32) * 0.2
    beta, init_s = torch.rand(1, 17, 3, dtype=torch.float32) * 0.8 + 0.1, torch.randn(1, 3, 33, 31, dtype=torch.float32) * 0.1
    ry_k, rs_k = g_kda(q, k, v, g, beta, init_s, use_qk_l2norm=True, scaleValue=-1.0)
    oy_k, os_k = kda_kernel(q, k, v, g, beta, init_s, use_qk_l2norm=True, scaleValue=-1.0)
    torch.testing.assert_close(oy_k.float(), ry_k.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(os_k, rs_k, rtol=1e-2, atol=1e-2)

    g_vm = _load_golden("level5", "tensor_program_vm")
    prog_t, regs_t = torch.randint(0, 1_000_000_000, (64, 4), dtype=torch.int32), torch.randn(8, 17, dtype=torch.float32)
    torch.testing.assert_close(vm_kernel(prog_t, regs_t).float(), g_vm(prog_t, regs_t).float(), rtol=1e-2, atol=1e-2)
