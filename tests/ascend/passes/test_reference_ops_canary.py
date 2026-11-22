"""Canary tests for anchor specialization, Two-Snapshot diff, and reference Ascend ops."""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

from examples.ascend950pr_ops.add_rms_norm_dynamic_quant.kernels import (
    _build_add_rms_norm_dynamic_quant_prim_func,
)
from examples.ascend950pr_ops.attention_residual.kernels import (
    _build_attention_residual_prim_func,
)
from examples.ascend950pr_ops.gqa.kernels import _build_gqa_prim_func
from examples.ascend950pr_ops.grouped_matmul.kernels import (
    _render_trackb_tu as _render_gmm_trackb_tu,
)
from examples.ascend950pr_ops.mla.kernels import (
    _build_mla_prim_func,
    _render_trackb_tu as _render_mla_trackb_tu,
)
from examples.ascend950pr_ops.rms_norm.kernels import _build_rms_norm_prim_func
from examples.ascend950pr_ops.softmax.kernels import _build_softmax_prim_func
from examples.ascend950pr_ops.swi_glu.kernels import _build_swi_glu_prim_func
from tilefoundry_ascend.agent.orchestrator.gate_funnel import build_stage_b_driver_source
from tilefoundry_ascend.ascend.schedule import inspect_tilelang_primfunc
from tilefoundry_ascend.ascend.sil import (
    AdvisoryEnvelope,
    ConstraintRelaxationMask,
    HardwareLegalityBounds,
    PolicyEnvelope,
    RegimeSlice,
    SchedulePolicy,
    SILVerifier,
    TopologyAxes,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_first_benchmark_case(op_name: str) -> dict[str, object]:
    cases_path = _REPO_ROOT / "tests" / "ascend" / "benchmarks" / op_name / "cases.yaml"
    payload = yaml.safe_load(cases_path.read_text(encoding="utf-8"))
    cases = payload["cases"]
    assert isinstance(cases, list) and cases
    return dict(cases[0])


def _mte2_total(pipe_cycles: dict[str, int]) -> int:
    return sum(int(v) for k, v in pipe_cycles.items() if k.endswith("PIPE_MTE2"))


def test_reference_ops_anchor_specialization_and_mte2_nonzero() -> None:
    """Verify real-anchor specialization preserves Two-Snapshot diff and nonzero MTE2."""
    from tilelang import tvm  # noqa: PLC0415

    ar_case = _load_first_benchmark_case("attention_residual")
    ar_v_shape = ar_case["input_shape"][0]
    ar_n, ar_b, ar_s, ar_d = [int(x) for x in ar_v_shape]

    mla_case = _load_first_benchmark_case("mla")
    mla_q_shape = mla_case["input_shape"][0]
    mla_b, mla_s, mla_nq, mla_dnope = [int(x) for x in mla_q_shape]

    rms_2d = _build_rms_norm_prim_func(d_dim=1024, tl_dtype="bfloat16", tile_m=4, num_stages=2)
    orig_rms_shapes = [list(buf.shape) for buf in rms_2d.buffer_map.values()]

    canary_specs = [
        ("rms_norm_2d", rms_2d, {"m_rows": 4096}, False),
        (
            "rms_norm_1d_d2",
            _build_rms_norm_prim_func(d_dim=2, tl_dtype="bfloat16", tile_m=256, num_stages=2),
            {"m_rows": 4096},
            False,
        ),
        (
            "softmax_2d",
            _build_softmax_prim_func(d_dim=1024, tl_dtype="bfloat16", tile_m=2, num_stages=2),
            {"m_rows": 4096},
            False,
        ),
        (
            "swi_glu_2d",
            _build_swi_glu_prim_func(d_dim=1024, tl_dtype="bfloat16"),
            {"m_rows": 4096},
            False,
        ),
        (
            "add_rms_norm_dynamic_quant",
            _build_add_rms_norm_dynamic_quant_prim_func(d_dim=1024, tl_dtype="bfloat16"),
            {"m_rows": 4096},
            False,
        ),
        (
            "attention_residual",
            _build_attention_residual_prim_func(ar_n, ar_d, "bfloat16", 4, "tuned"),
            {"m_rows": ar_b * ar_s, "v_rows": ar_n * ar_b * ar_s},
            False,
        ),
        (
            "gqa",
            _build_gqa_prim_func(
                n_q=32, n_kv=8, d_dim=128, tl_dtype="float16", bm=64, bn=128, l1_stages=2
            ),
            {"b_dim": 2, "s_dim": 128, "s_kv": 128},
            True,
        ),
        (
            "mla",
            _build_mla_prim_func(mla_nq, 1, mla_dnope, 64, "float16", "BSND", True, 64, True, "tuned"),
            {"b_dim": mla_b, "q_rows": mla_b * mla_s, "s_dim": mla_s, "s_kv": mla_s},
            True,
        ),
    ]

    assert tvm.target.Target.current(allow_none=True) is None
    for op_name, fn, bindings, expect_cube in canary_specs:
        res = inspect_tilelang_primfunc(fn, var_bindings=bindings)
        assert res.lowering_error is None, f"{op_name} lowering error: {res.lowering_error}"
        assert res.ledger.is_legal is True, f"{op_name} illegal ledger"
        assert "tl.MaterializeMultiBuffer" in res.lowered_passes_observed, op_name
        assert "tl.MergeUBAllocations" in res.lowered_passes_observed, op_name
        assert _mte2_total(res.ledger.pipe_cycles) > 0, f"{op_name} PIPE_MTE2 == 0"
        assert res.ledger.min_gm_dma_bytes > 0, f"{op_name} min_gm_dma_bytes == 0"
        if expect_cube:
            assert int(res.ledger.pipe_cycles.get("AIC.PIPE_M", 0)) > 0, f"{op_name} AIC.PIPE_M == 0"
        if op_name == "rms_norm_1d_d2":
            ub_wl = res.ledger.waterlines["UB"]
            assert ub_wl.explicit_buffers["x_ub"] == 256 * 2 * 2
            assert ub_wl.explicit_buffers["y_ub"] == 256 * 2 * 2
            assert res.ledger.min_gm_dma_bytes == 256 * 2 * 2

    assert [list(buf.shape) for buf in rms_2d.buffer_map.values()] == orig_rms_shapes
    assert tvm.target.Target.current(allow_none=True) is None


def test_dynamic_2d_row_copy_specializes_for_anchor_bytes(tmp_path: Path) -> None:
    """Verify 2D dynamic T.copy(X[row, 0], ub) computes exact MTE2 bytes after specialization."""
    import importlib.util  # noqa: PLC0415

    src = (
        "import tilelang.ascend.language as T\n"
        "def make_row_copy():\n"
        "    m_rows = T.dynamic('m_rows')\n"
        "    @T.prim_func\n"
        "    def main(X: T.Tensor((m_rows, 1024), 'bfloat16'), "
        "Y: T.Tensor((m_rows, 1024), 'bfloat16')):\n"
        "        with T.Kernel(1) as _:\n"
        "            ub = T.alloc_shared((1024,), 'bfloat16')\n"
        "            for row in T.serial(m_rows):\n"
        "                T.copy(X[row, 0], ub)\n"
        "                T.copy(ub, Y[row, 0])\n"
        "    return main\n"
    )
    mod_path = tmp_path / "dyn_row_copy.py"
    mod_path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_tf_dyn_row_copy", str(mod_path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    fn = mod.make_row_copy()
    orig_shapes = [list(buf.shape) for buf in fn.buffer_map.values()]
    res = inspect_tilelang_primfunc(fn, var_bindings={"m_rows": 64})
    assert res.lowering_error is None, res.lowering_error
    assert res.ledger.is_legal is True
    assert _mte2_total(res.ledger.pipe_cycles) > 0
    assert res.ledger.min_gm_dma_bytes == 2048
    assert res.ledger.dma_burst_efficiency_ratio > 0.0
    assert res.ledger.opaque_dynamic_regions == ()
    assert [list(buf.shape) for buf in fn.buffer_map.values()] == orig_shapes


def test_reference_ops_sil_verifier_and_trackb_canary() -> None:
    """Verify SILVerifier passes Track-B translation units and canonical op policies."""
    verifier = SILVerifier()
    top_mask = ConstraintRelaxationMask.top()

    gmm_tu = _render_gmm_trackb_tu(m_dim=128, k_dim=256, n_dim=512, tl_dtype="float16")
    gmm_rep = verifier.verify_source_against_facets(gmm_tu, top_mask)
    assert gmm_rep.passed is True, f"grouped_matmul Track-B failed SILVerifier: {gmm_rep.errors}"

    mla_tu = _render_mla_trackb_tu()
    mla_rep = verifier.verify_source_against_facets(mla_tu, top_mask)
    assert mla_rep.passed is True, f"mla Track-B failed SILVerifier: {mla_rep.errors}"

    bounds = HardwareLegalityBounds()
    aiv_axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
    cv_axes = TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd")
    ragged_axes = TopologyAxes("coupled_1c2v", "l0c_ub_epilogue", "ragged_group")

    policy_checks = [
        (
            "rms_norm",
            aiv_axes,
            SchedulePolicy(
                regime_tag="aligned_packed",
                topology_axes=aiv_axes,
                tile_mnk=(4, 1024, 1),
                vector_mode="simdvf",
                ub_256kb_unlock=True,
                ub_estimated_bytes=22528,
            ),
            False,
        ),
        (
            "softmax",
            aiv_axes,
            SchedulePolicy(
                regime_tag="aligned_packed_regbase",
                topology_axes=aiv_axes,
                tile_mnk=(2, 1024, 1),
                vector_mode="simdvf",
                ub_256kb_unlock=True,
                ub_estimated_bytes=12288,
            ),
            True,
        ),
        (
            "gqa",
            cv_axes,
            SchedulePolicy(
                regime_tag="prefill",
                topology_axes=cv_axes,
                tile_mnk=(64, 128, 128),
                l1_stages=2,
                l0c_buffers=1,
                l1_estimated_bytes=98304,
                ub_estimated_bytes=77824,
                ub_256kb_unlock=True,
                cv_datapath="ub_l1_ring",
            ),
            False,
        ),
        (
            "grouped_matmul",
            ragged_axes,
            SchedulePolicy(
                regime_tag="ragged_group",
                topology_axes=ragged_axes,
                tile_mnk=(64, 128, 64),
                l1_stages=2,
                l0c_buffers=1,
                l1_estimated_bytes=65536,
                ub_estimated_bytes=65536,
                ub_256kb_unlock=True,
                cv_datapath="l0c2ub_split",
                ensure_empty_subblock_sync=True,
            ),
            False,
        ),
    ]

    for op_name, axes, policy, has_nan_inf in policy_checks:
        regime = RegimeSlice(
            regime_tag=policy.regime_tag,
            case_nums=(1,),
            representative_dims={"M": 128, "N": 128, "D": 1024},
            has_unaligned_tail=False,
            has_empty_group=(axes.iteration_regularity == "ragged_group"),
        )
        advisory = AdvisoryEnvelope(
            op_name=op_name,
            cannbot_task=f"level4/{op_name}",
            level=4,
            topology_axes=axes,
            input_dtypes=("float16",),
            output_dtypes=("float16",),
            roofline={},
            structural_features={},
            regimes=(regime,),
            legality_bounds=bounds,
            has_nan_inf_cases=has_nan_inf,
        )
        envelope = PolicyEnvelope(op_name=op_name, policies={policy.regime_tag: policy})
        rep = verifier.verify_envelope(envelope, advisory)
        assert rep.passed is True, f"{op_name} failed SILVerifier: {rep.errors}"


def test_stage_b_driver_source_contiguous_context_and_no_target_none() -> None:
    """Verify build_stage_b_driver_source keeps tilelang.lower inside _ascend_compile_session."""
    driver_specs = [
        ("rms_norm", "_build_rms_norm_prim_func", {"d_dim": 1024}),
        ("softmax", "_build_softmax_prim_func", {"d_dim": 1024}),
        ("gqa", "_build_gqa_prim_func", {"d_dim": 128}),
        ("grouped_matmul", "_select_grouped_matmul_config", {"k_dim": 256, "n_dim": 512}),
    ]
    for op_name, factory_name, dims in driver_specs:
        cand_path = _REPO_ROOT / "examples" / "ascend950pr_ops" / op_name / "kernels.py"
        driver = build_stage_b_driver_source(
            cand_path,
            factory_name=factory_name,
            dims=dims,
        )
        tree = ast.parse(driver)
        assert "_ascend_compile_session" in driver, op_name
        assert "Target(None)" not in driver, op_name

        lower_inside_session = False

        def _visit(node: ast.AST, in_session: bool) -> None:
            nonlocal lower_inside_session
            next_session = in_session
            if isinstance(node, ast.With):
                for item in node.items:
                    ctx_expr = item.context_expr
                    if (
                        isinstance(ctx_expr, ast.Call)
                        and isinstance(ctx_expr.func, ast.Name)
                        and ctx_expr.func.id == "_ascend_compile_session"
                    ):
                        next_session = True
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "tilelang"
                and node.func.attr == "lower"
                and next_session
            ):
                lower_inside_session = True
            for child in ast.iter_child_nodes(node):
                _visit(child, next_session)

        _visit(tree, False)
        assert lower_inside_session is True, f"{op_name} tilelang.lower outside session"
