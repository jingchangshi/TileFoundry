"""Structural and Stage-A contracts for the 6 Ascend 950PR operators in V5.2."""

from __future__ import annotations

import ast
import importlib
import inspect
import math
from pathlib import Path

import torch

from examples.ascend950pr_ops.add_rms_norm_dynamic_quant import kernels as rms_kernels
from examples.ascend950pr_ops.attention_residual import kernels as ar_kernels
from examples.ascend950pr_ops.grouped_matmul import kernels as gmm_kernels
from examples.ascend950pr_ops.kimi_delta_attention import kernels as kda_kernels
from examples.ascend950pr_ops.mla import kernels as mla_kernels
from examples.ascend950pr_ops.sparse_flash_attention import kernels as sfa_kernels
from tilefoundry_ascend.agent.orchestrator.gate_funnel import run_stage_a_static
from tilefoundry_ascend.ascend.schedule.envelope_inspector import (
    inspect_tilelang_primfunc,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _assert_stage_a_legal(
    prog: object,
    pass_configs: dict[str, object] | None,
    compile_flags: list[str] | tuple[str, ...] | None,
    label: str,
) -> None:
    inspection = inspect_tilelang_primfunc(
        prog,
        pass_configs=pass_configs,
        compile_flags=compile_flags,
    )
    assert inspection.lowering_error is None, f"{label}: {inspection.lowering_error}"
    assert inspection.ledger.is_legal, (
        f"{label}: overflow={inspection.ledger.max_overflow_bytes}"
    )


def test_six_ops_public_entrypoint_signatures_and_exports() -> None:
    """Verify public entrypoint signatures and kernel aliases for all 6 operators."""
    expected_params: dict[str, tuple[str, ...]] = {
        "kimi_delta_attention": (
            "q",
            "k",
            "v",
            "g",
            "beta",
            "initial_state",
            "use_qk_l2norm",
            "scaleValue",
            "schedule_mode",
        ),
        "grouped_matmul": (
            "x",
            "weight",
            "bias",
            "group_list",
            "split_item",
            "transpose_weight",
            "schedule_mode",
        ),
        "sparse_flash_attention": (
            "query",
            "key",
            "value",
            "sparseIndices",
            "scaleValue",
            "inputLayout",
            "is_causal",
            "schedule_mode",
        ),
        "mla": (
            "q_nope",
            "q_rope",
            "k_nope",
            "k_rope",
            "v",
            "numKVHeads",
            "scaleValue",
            "inputLayout",
            "is_causal",
            "schedule_mode",
            "num_kv_splits_override",
        ),
        "add_rms_norm_dynamic_quant": (
            "x1",
            "x2",
            "gamma",
            "epsilon",
            "schedule_mode",
        ),
        "attention_residual": (
            "v_stack",
            "x",
            "w",
            "gamma",
            "alpha",
            "epsilon",
            "schedule_mode",
        ),
    }
    kernel_alias_ops = {
        "kimi_delta_attention",
        "sparse_flash_attention",
        "add_rms_norm_dynamic_quant",
        "attention_residual",
    }
    for op_name, params in expected_params.items():
        mod = importlib.import_module(f"examples.ascend950pr_ops.{op_name}.kernels")
        pkg = importlib.import_module(f"examples.ascend950pr_ops.{op_name}")
        entry = getattr(mod, op_name)
        assert callable(entry), f"{op_name} entrypoint missing"
        assert getattr(pkg, op_name) is entry, f"{op_name} package export mismatch"
        sig = inspect.signature(entry)
        assert tuple(sig.parameters.keys()) == params, f"{op_name} signature drift"
        assert sig.parameters["schedule_mode"].default == "tuned"
        if op_name in kernel_alias_ops:
            assert getattr(mod, "kernel") is entry, f"{op_name} kernel alias mismatch"


def test_add_rms_norm_dynamic_quant_reg_resident_tightening_v52() -> None:
    """Verify use_reg_s=(d_dim<=128 and rows<=2) and UB packing in RMSNorm."""
    sel = rms_kernels._select_add_rms_norm_dynamic_quant_config
    cfg_reg = sel(d_dim=128, tl_dtype="bfloat16", m_rows=112, schedule_mode="tuned")
    assert cfg_reg["tile_m"] <= 2
    assert cfg_reg["regime"] == "aligned_reg_resident"

    cfg_128_multi = sel(
        d_dim=128,
        tl_dtype="bfloat16",
        m_rows=524288,
        schedule_mode="tuned",
    )
    assert cfg_128_multi["tile_m"] > 2
    assert cfg_128_multi["regime"] == "aligned_single_pass_ub"

    cfg_128_ov = sel(
        d_dim=128,
        tl_dtype="bfloat16",
        m_rows=112,
        schedule_mode="tuned",
        tile_m_override=4,
    )
    assert cfg_128_ov["tile_m"] == 4
    assert cfg_128_ov["regime"] == "aligned_single_pass_ub"

    cfg_256 = sel(
        d_dim=256,
        tl_dtype="float16",
        m_rows=256,
        schedule_mode="tuned",
        tile_m_override=1,
    )
    assert cfg_256["regime"] == "aligned_single_pass_ub"

    for d_dim in (128, 1024, 2048, 4096, 7168):
        cfg = sel(d_dim=d_dim, tl_dtype="bfloat16", m_rows=4096, schedule_mode="tuned")
        assert int(cfg["tile_m"]) >= 1
        assert int(cfg["num_stages"]) in (1, 2)
        assert cfg["regime"] == "aligned_single_pass_ub"


def test_add_rms_norm_dynamic_quant_selector_overrides_and_stage_a_v52() -> None:
    """Verify selector overrides, Stage-A legality, and CPU reference semantics."""
    sel = rms_kernels._select_add_rms_norm_dynamic_quant_config
    bld = rms_kernels._build_add_rms_norm_dynamic_quant_prim_func

    cfg_ov = sel(
        d_dim=2048,
        tl_dtype="bfloat16",
        m_rows=4096,
        schedule_mode="tuned",
        tile_m_override=4,
        num_stages_override=1,
    )
    assert cfg_ov["tile_m"] == 4
    assert cfg_ov["num_stages"] == 1

    shapes = [
        (128, 112, "bfloat16"),
        (128, 524288, "float16"),
        (256, 256, "bfloat16"),
        (1024, 4096, "bfloat16"),
        (2048, 8192, "bfloat16"),
        (4096, 4096, "bfloat16"),
        (7168, 2048, "bfloat16"),
        (2053, 4093, "float16"),
    ]
    for d_dim, m_rows, dtype in shapes:
        cfg = sel(d_dim=d_dim, tl_dtype=dtype, m_rows=m_rows, schedule_mode="tuned")
        prog = bld(**cfg)
        _assert_stage_a_legal(
            prog,
            dict(cfg["pass_configs"]),
            list(cfg["compile_flags"]),
            f"rms_d{d_dim}_m{m_rows}",
        )

    src_path = (
        REPO_ROOT / "examples/ascend950pr_ops/add_rms_norm_dynamic_quant/kernels.py"
    )
    res = run_stage_a_static(
        src_path.read_text(encoding="utf-8"),
        factory_name="_build_add_rms_norm_dynamic_quant_prim_func",
        config_selector_name="_select_add_rms_norm_dynamic_quant_config",
        dims={"D": 4096, "M": 4096},
    )
    assert res.passed, f"RMS Stage-A failed: {res.error_message}"

    x1 = torch.randn(4, 256, dtype=torch.bfloat16)
    x2 = torch.randn(4, 256, dtype=torch.bfloat16)
    gamma = torch.randn(256, dtype=torch.bfloat16)
    y_q, x_out, scale = rms_kernels.kernel(x1, x2, gamma, epsilon=1e-6)
    assert y_q.dtype == torch.int8 and y_q.shape == (4, 256)
    assert x_out.dtype == torch.bfloat16 and x_out.shape == (4, 256)
    assert scale.dtype == torch.float32 and scale.shape == (4,)


def test_attention_residual_small_decode_2d_saturation_v52() -> None:
    """Verify 2D (row, d_blk) core saturation and knob overrides on AttentionResidual."""
    for m_rows, d_dim, min_tasks in [
        (1, 4096, 16),
        (4, 4096, 56),
        (8, 4096, 56),
        (16, 3584, 56),
        (32, 2560, 56),
    ]:
        cfg = ar_kernels._select_ar_small_decode_config(
            n_depth=4,
            d_dim=d_dim,
            tl_dtype="bfloat16",
            schedule_mode="tuned",
            m_rows=m_rows,
        )
        d_sub = int(cfg["d_subtile"])
        assert d_sub >= 256 and (d_sub % 256) == 0
        num_d_blks = (d_dim + d_sub - 1) // d_sub
        assert m_rows * num_d_blks >= min_tasks
        assert int(cfg["tile_m"]) == 1
        assert int(cfg["num_stages"]) == 2

    for sel in (
        ar_kernels._select_ar_small_decode_config,
        ar_kernels._select_ar_unaligned_config,
        ar_kernels._select_ar_aligned_config,
    ):
        cfg = sel(
            n_depth=4,
            d_dim=2048,
            tl_dtype="bfloat16",
            schedule_mode="tuned",
            m_rows=1024,
            tile_m_override=2,
            num_stages_override=1,
        )
        assert cfg["tile_m"] == 2
        assert cfg["num_stages"] == 1

    src = (
        REPO_ROOT / "examples/ascend950pr_ops/attention_residual/kernels.py"
    ).read_text(encoding="utf-8")
    assert "elif m_rows < 16 or (m_rows < 56 and d_dim >= 1024):" in src


def test_attention_residual_stage_a_all_regimes_and_tail_v52() -> None:
    """Verify Stage-A legality across all AttentionResidual regimes and d_subtile tails."""
    cases = [
        (
            "small_d4096_m1",
            ar_kernels._select_ar_small_decode_config,
            ar_kernels._build_ar_small_decode_func,
            {"n_depth": 2, "d_dim": 4096, "m_rows": 1},
        ),
        (
            "small_d3584_m16_tail",
            ar_kernels._select_ar_small_decode_config,
            ar_kernels._build_ar_small_decode_func,
            {"n_depth": 4, "d_dim": 3584, "m_rows": 16},
        ),
        (
            "small_d1088_m8_tail",
            ar_kernels._select_ar_small_decode_config,
            ar_kernels._build_ar_small_decode_func,
            {"n_depth": 4, "d_dim": 1088, "m_rows": 8},
        ),
        (
            "small_d512_m4",
            ar_kernels._select_ar_small_decode_config,
            ar_kernels._build_ar_small_decode_func,
            {"n_depth": 4, "d_dim": 512, "m_rows": 4},
        ),
        (
            "unaligned_d2047",
            ar_kernels._select_ar_unaligned_config,
            ar_kernels._build_ar_unaligned_func,
            {"n_depth": 3, "d_dim": 2047, "m_rows": 333},
        ),
        (
            "aligned_d4096",
            ar_kernels._select_ar_aligned_config,
            ar_kernels._build_ar_aligned_func,
            {"n_depth": 8, "d_dim": 4096, "m_rows": 8192},
        ),
        (
            "aligned_d7168",
            ar_kernels._select_ar_aligned_config,
            ar_kernels._build_ar_aligned_func,
            {"n_depth": 4, "d_dim": 7168, "m_rows": 4096},
        ),
    ]
    for label, sel, bld, kw in cases:
        cfg = sel(**kw, schedule_mode="tuned")
        prog = bld(**kw, **cfg, schedule_mode="tuned")
        _assert_stage_a_legal(
            prog,
            dict(cfg["pass_configs"]),
            list(cfg["compile_flags"]),
            label,
        )

    ar_src = (
        REPO_ROOT / "examples/ascend950pr_ops/attention_residual/kernels.py"
    ).read_text(encoding="utf-8")
    for bld_name, sel_name, dims in [
        (
            "_build_ar_small_decode_func",
            "_select_ar_small_decode_config",
            {"N": 2, "B": 1, "S": 1, "D": 4096},
        ),
        (
            "_build_ar_unaligned_func",
            "_select_ar_unaligned_config",
            {"N": 8, "B": 1, "S": 16, "D": 1000},
        ),
        (
            "_build_ar_aligned_func",
            "_select_ar_aligned_config",
            {"N": 8, "B": 1, "S": 4096, "D": 4096},
        ),
    ]:
        res = run_stage_a_static(
            ar_src,
            factory_name=bld_name,
            config_selector_name=sel_name,
            dims=dims,
        )
        assert res.passed, f"AR Stage-A {bld_name} failed: {res.error_message}"


def test_kimi_delta_attention_contract_d_and_stage_a_v52() -> None:
    """Verify KDA Contract-D knobs (chunk_t, dv_tile) and Stage-A legality."""
    sel = kda_kernels._select_kimi_delta_attention_config
    cfg_small = sel(dk=128, dv=128, tl_dtype="bfloat16", schedule_mode="tuned")
    assert int(cfg_small["chunk_t"]) >= 1
    assert int(cfg_small["dv_tile"]) >= 64

    cfg_large = sel(dk=128, dv=256, tl_dtype="bfloat16", schedule_mode="tuned")
    assert int(cfg_large["chunk_t"]) >= 1
    assert int(cfg_large["dv_tile"]) == 64

    for dv in (128, 256):
        cfg = sel(dk=128, dv=dv, tl_dtype="bfloat16", schedule_mode="tuned")
        prog = kda_kernels._build_kimi_delta_attention_prim_func(**cfg)
        _assert_stage_a_legal(
            prog,
            dict(cfg["pass_configs"]),
            list(cfg["compile_flags"]),
            f"kda_dv{dv}",
        )


def test_grouped_matmul_contract_d_and_source_v52() -> None:
    """Verify GroupedMatmul Contract-D knobs and persistent launch source."""
    sel = gmm_kernels._select_grouped_matmul_config
    cfg = sel(
        group_sizes=(512, 512),
        k_dim=1024,
        n_dim=2048,
        tl_dtype="float16",
        schedule_mode="tuned",
    )
    for key in ("bm", "bn", "bk", "num_stages"):
        assert key in cfg, f"grouped_matmul config missing {key}"

    cfg_ov = sel(
        group_sizes=(512, 512),
        k_dim=1024,
        n_dim=2048,
        tl_dtype="float16",
        schedule_mode="tuned",
        bm_override=64,
        bn_override=128,
        bk_override=64,
        num_stages_override=1,
    )
    assert cfg_ov["bm"] == 64
    assert cfg_ov["bn"] == 128
    assert cfg_ov["bk"] == 64
    assert cfg_ov["num_stages"] == 1

    src = (
        REPO_ROOT / "examples/ascend950pr_ops/grouped_matmul/kernels.py"
    ).read_text(encoding="utf-8")
    ast.parse(src)
    assert "linear_tile" in src


def test_sparse_flash_attention_contract_d_and_stage_a_v52() -> None:
    """Verify SparseFlashAttention Contract-D knobs and Stage-A legality."""
    sel = sfa_kernels._select_sparse_flash_attention_config
    cfg = sel(
        g_ratio=16,
        dk=128,
        dv=128,
        dtype="float16",
        top_k=512,
        schedule_mode="tuned",
    )
    for key in ("bm", "bn", "num_stages"):
        assert key in cfg, f"sparse_flash_attention config missing {key}"
    assert cfg["use_pipelined"] is True

    prog = sfa_kernels._build_sparse_flash_attention_prim_func(**cfg)
    _assert_stage_a_legal(
        prog,
        cfg.get("pass_configs") if isinstance(cfg.get("pass_configs"), dict) else None,
        list(cfg.get("compile_flags") or ()),
        "sfa_top_k_512",
    )


def test_mla_contract_d_and_stage_a_v52() -> None:
    """Verify MLA Contract-D knobs across selectors and Stage-A legality."""
    for sel in (
        mla_kernels._select_mla_decode_config,
        mla_kernels._select_mla_bnsd_config,
        mla_kernels._select_mla_prefill_config,
    ):
        cfg = sel(schedule_mode="tuned")
        for key in ("bm", "bn", "num_stages"):
            assert key in cfg, f"{sel.__name__} missing {key}"

    cfg_dec = mla_kernels._select_mla_decode_config(
        b_dim=1,
        s_dim=1,
        s_kv=2048,
        n_q=128,
        n_kv=1,
        schedule_mode="tuned",
    )
    prog = mla_kernels._build_mla_split_kv_decode_func(**cfg_dec)
    _assert_stage_a_legal(
        prog,
        dict(cfg_dec["pass_configs"]),
        list(cfg_dec["compile_flags"]),
        "mla_decode",
    )
    assert math.isfinite(float(cfg_dec["bm"]))
