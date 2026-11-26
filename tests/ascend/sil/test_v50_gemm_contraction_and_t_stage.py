"""Regression tests for B5 gemm contraction routing, T.Stage AST verifier, and AR configs."""

from __future__ import annotations

import importlib.util
import time
from dataclasses import fields
from pathlib import Path
from textwrap import dedent

import pytest
import torch

from tilefoundry_ascend.agent.campaign import marathon
from tilefoundry_ascend.agent.campaign.marathon import (
    MarathonConfig,
    build_marathon_arg_parser,
)
from tilefoundry_ascend.ascend.sil import DEFAULT_OPS_ROOT
from tilefoundry_ascend.ascend.sil.pytorch_frontend import PyTorchSemanticAnalyzer
from tilefoundry_ascend.ascend.sil.schema import ConstraintRelaxationMask, TopologyAxes
from tilefoundry_ascend.ascend.sil.verifier import SILVerifier, verify_tilelang_ast
from tilefoundry_ascend.ascend.target import (
    TILELANG_DSL_SURFACE,
    load_cann_surface,
)
from tilefoundry_ascend.ascend.target.dialect_surface import filter_primitives_on_demand

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AR_KERNELS_PATH = _REPO_ROOT / "examples" / "ascend950pr_ops" / "attention_residual" / "kernels.py"
_CANN_92_TOML_PATH = (
    _REPO_ROOT
    / "targets"
    / "ascend"
    / "tilefoundry_ascend"
    / "ascend"
    / "target"
    / "cann_surface"
    / "cann_9_2.toml"
)


def _load_ar_kernels_module():
    spec = importlib.util.spec_from_file_location("_test_ar_kernels_mod", str(_AR_KERNELS_PATH))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_attention_residual_routes_to_aiv_only_and_gemm_einsum_keeps_cube() -> None:
    """Route non-GEMM einsum ops like attention_residual to aiv_only while keeping GEMM on cube."""
    analyzer = PyTorchSemanticAnalyzer()
    ar_golden = DEFAULT_OPS_ROOT / "level4" / "attention_residual" / "golden.py"
    if ar_golden.is_file():
        ar_src = ar_golden.read_text(encoding="utf-8")
    else:
        ar_src = dedent(
            """
            import math
            import torch

            def attention_residual(v_stack, x, w, gamma, alpha, epsilon=1e-6):
                v = v_stack.to(torch.float32)
                rms = torch.sqrt(v.pow(2).mean(dim=-1, keepdim=True) + float(epsilon))
                k = v / rms * gamma.to(torch.float32)
                s = torch.einsum("nbtd,d->nbt", k, w.to(torch.float32)) / math.sqrt(v.shape[-1])
                p = torch.softmax(s, dim=0)
                agg = torch.einsum("nbt,nbtd->btd", p, v)
                return (1.0 - float(alpha)) * x.to(torch.float32) + float(alpha) * agg
            """
        )
    ar_sem = analyzer.analyze_source(ar_src, entry_name="attention_residual")
    assert ar_sem.has_gemm_contraction is False
    assert ar_sem.base_axes.compute_engine == "aiv_only"
    assert ar_sem.base_axes.memory_ring == "gm_ub_stream"

    assert _AR_KERNELS_PATH.is_file()
    ar_kernel_text = _AR_KERNELS_PATH.read_text(encoding="utf-8")
    assert "T.Kernel(" in ar_kernel_text
    assert "T.SimdVF()" in ar_kernel_text

    gemm_src = dedent(
        """
        import torch

        def batched_gemm(a, b):
            return torch.einsum("bik,bkj->bij", a, b)
        """
    )
    gemm_sem = analyzer.analyze_source(gemm_src, entry_name="batched_gemm")
    assert gemm_sem.has_gemm_contraction is True
    assert gemm_sem.base_axes.compute_engine in ("aic_only", "coupled_1c2v")


def test_marathon_dead_prewarm_removed_and_cost_model_mode_added() -> None:
    """Confirm prewarm_l3/build_prewarmed_l3_runner are gone and cost_model_mode defaults off."""
    assert not hasattr(marathon, "build_prewarmed_l3_runner")
    assert "build_prewarmed_l3_runner" not in marathon.__all__
    field_names = {f.name for f in fields(MarathonConfig)}
    assert "prewarm_l3" not in field_names
    assert "cost_model_mode" in field_names

    cfg = MarathonConfig(operator_name="mla")
    assert cfg.cost_model_mode == "off"
    parsed = build_marathon_arg_parser().parse_args(
        ["run", "--operator", "mla", "--cost-model-mode", "advisory"]
    )
    assert parsed.cost_model_mode == "advisory"


def test_sil_verifier_tilelang_ast_stage_and_parallel_redlines() -> None:
    """Reject illegal T.Stage and T.Parallel constructs in milliseconds while allowing valid DSL."""
    bad_parallel_copy = dedent(
        """
        import tilelang.ascend.language as T

        @T.prim_func
        def k(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
            for i in T.Parallel(1):
                T.copy(A, B)
        """
    )
    bad_parallel_gemm = dedent(
        """
        import tilelang.ascend.language as T

        @T.prim_func
        def k(A, B, C):
            for i in T.Parallel(1):
                T.gemm(A, B, C)
        """
    )
    bad_simdvf_stage = dedent(
        """
        import tilelang.ascend.language as T

        @T.prim_func
        def k():
            for n in T.Pipelined(4, num_stages=2):
                with T.SimdVF():
                    with T.Stage(1):
                        pass
        """
    )
    bad_unpipelined_stage = dedent(
        """
        import tilelang.ascend.language as T

        @T.prim_func
        def k():
            for n in T.serial(4):
                with T.Stage(1):
                    pass
        """
    )
    bad_single_stage_pipelined = dedent(
        """
        import tilelang.ascend.language as T

        @T.prim_func
        def k():
            for n in T.Pipelined(4, num_stages=1):
                with T.Stage(1):
                    pass
        """
    )
    valid_pipelined = dedent(
        """
        import tilelang.ascend.language as T

        @T.prim_func
        def k(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
            for i in T.Parallel(16):
                B[i] = A[i] + 1.0
            for n in T.Pipelined(4, num_stages=2):
                with T.Stage(0):
                    T.copy(A, B)
                with T.Stage(1):
                    pass
        """
    )
    t0 = time.monotonic()
    verifier = SILVerifier()
    rep_copy = verifier.verify_tilelang_ast(bad_parallel_copy)
    rep_gemm = verifier.verify_tilelang_ast(bad_parallel_gemm)
    rep_simdvf = verify_tilelang_ast(bad_simdvf_stage)
    rep_unpiped = verify_tilelang_ast(bad_unpipelined_stage)
    rep_single = verify_tilelang_ast(bad_single_stage_pipelined)
    rep_valid = verifier.verify_tilelang_ast(valid_pipelined)
    elapsed_ms = (time.monotonic() - t0) * 1000.0

    assert not rep_copy.passed and any("T.copy" in err for err in rep_copy.errors)
    assert not rep_gemm.passed and any("T.gemm" in err for err in rep_gemm.errors)
    assert not rep_simdvf.passed and any("T.SimdVF" in err for err in rep_simdvf.errors)
    assert not rep_unpiped.passed and any("T.Stage(1)" in err for err in rep_unpiped.errors)
    assert not rep_single.passed and any("T.Stage(1)" in err for err in rep_single.errors)
    assert rep_valid.passed is True
    assert elapsed_ms < 500.0


def test_attention_residual_pass_configs_and_vf_stack_flag_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Handle sequence pass_configs safely and gate VF stack disable flag at > 224 KiB UB."""
    ar_mod = _load_ar_kernels_module()
    ar_mod._KERNEL_CACHE.clear()
    ar_mod._DISPATCH_CACHE.clear()

    compile_calls: list[dict[str, object]] = []

    def _fake_compile(_prog, **kwargs):
        compile_calls.append(dict(kwargs))

        def _dummy_kernel(*_args):
            return torch.zeros((1, 256), dtype=torch.bfloat16)

        return _dummy_kernel

    monkeypatch.setattr(ar_mod.tilelang, "compile", _fake_compile)

    ar_mod._get_kernel(
        2,
        256,
        "bfloat16",
        1,
        "tuned",
        pass_configs=["--cce-disable-vf-stack-reserved-ubuf"],
    )
    assert len(compile_calls) == 1
    assert compile_calls[0]["pass_configs"] is None
    assert "--cce-disable-vf-stack-reserved-ubuf" in (compile_calls[0]["compile_flags"] or [])

    small_cfg = ar_mod._select_ar_small_decode_config(
        n_depth=2,
        d_dim=4096,
        tl_dtype="bfloat16",
        schedule_mode="tuned",
        m_rows=1,
    )
    assert int(small_cfg["static_ub_bytes"]) <= 224 * 1024
    assert "--cce-disable-vf-stack-reserved-ubuf" not in small_cfg["compile_flags"]

    large_cfg = ar_mod._select_ar_aligned_config(
        n_depth=8,
        d_dim=4096,
        tl_dtype="bfloat16",
        schedule_mode="tuned",
        m_rows=2048,
    )
    assert int(large_cfg["static_ub_bytes"]) > 224 * 1024
    assert "--cce-disable-vf-stack-reserved-ubuf" in large_cfg["compile_flags"]

    compile_calls.clear()
    monkeypatch.setattr(
        ar_mod,
        "_select_ar_small_decode_config",
        lambda *args, **kwargs: {
            "tile_m": 1,
            "pass_configs": ["--cce-disable-vf-stack-reserved-ubuf"],
            "compile_flags": [],
        },
    )
    v_stack = torch.zeros((2, 1, 1, 256), dtype=torch.bfloat16)
    x = torch.zeros((1, 1, 256), dtype=torch.bfloat16)
    w = torch.zeros((256,), dtype=torch.bfloat16)
    g = torch.zeros((256,), dtype=torch.bfloat16)
    out = ar_mod.attention_residual(v_stack, x, w, g, 0.5)
    assert tuple(out.shape) == (1, 1, 256)
    assert compile_calls[-1]["pass_configs"] is None
    assert "--cce-disable-vf-stack-reserved-ubuf" not in (
        compile_calls[-1]["compile_flags"] or []
    )


def test_cann_9_3_surface_adds_new_primitives_and_leaves_cann_9_2_inert() -> None:
    """Expose T.Stage, T.assume_no_conflict, and AscendMGroupedTileScheduler on CANN 9.3 only."""
    spec_93 = load_cann_surface("cann_9_3")
    by_sym_93 = {entry.symbol: entry for entry in spec_93.primitives}
    assert "T.Stage" in by_sym_93
    assert by_sym_93["T.Stage"].pipe == "PIPE_SYNC"
    assert by_sym_93["T.Stage"].facet == "pipeline_and_sync_graph"
    assert "T.assume_no_conflict" in by_sym_93
    assert by_sym_93["T.assume_no_conflict"].facet == "sram_layout_and_aliasing"
    assert "AscendMGroupedTileScheduler" in by_sym_93
    assert by_sym_93["AscendMGroupedTileScheduler"].facet == "grid_and_workspace_topology"

    tl_syms = {entry.symbol for entry in TILELANG_DSL_SURFACE.primitives}
    assert {"T.Stage", "T.assume_no_conflict", "AscendMGroupedTileScheduler"} <= tl_syms

    ragged_axes = TopologyAxes("coupled_1c2v", "l0c_ub_epilogue", "ragged_group")
    affine_axes = TopologyAxes("coupled_1c2v", "l0c_ub_epilogue", "affine_simd")
    ragged_prims = {
        e.symbol
        for e in filter_primitives_on_demand(
            TILELANG_DSL_SURFACE,
            topology_axes=ragged_axes,
            freedom_mask=ConstraintRelaxationMask.bottom(),
        )
    }
    affine_prims = {
        e.symbol
        for e in filter_primitives_on_demand(
            TILELANG_DSL_SURFACE,
            topology_axes=affine_axes,
            freedom_mask=ConstraintRelaxationMask.bottom(),
        )
    }
    assert "AscendMGroupedTileScheduler" in ragged_prims
    assert "AscendMGroupedTileScheduler" not in affine_prims

    cann_92_text = _CANN_92_TOML_PATH.read_text(encoding="utf-8")
    for sym in ("T.Stage", "T.assume_no_conflict", "AscendMGroupedTileScheduler"):
        assert sym not in cann_92_text
