"""Zero-opname routing invariance, Track-B predicate canary, and reference ops canaries."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
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
from tilefoundry_ascend.agent.gates.gate_funnel import (
    _detect_cann_include_dirs,
    _detect_tilelang_include_dir,
    build_stage_b_driver_source,
)
from tilefoundry_ascend.agent.router import capability_gate
from tilefoundry_ascend.agent.router.capability_gate import route_from_op_spec
from tilefoundry_ascend.ascend.schedule import inspect_tilelang_primfunc, pipeline_solver
from tilefoundry_ascend.ascend.schedule.pipeline_solver import read_from_sil
from tilefoundry_ascend.ascend.schedule.regime_solver import RegimePiecewiseSolver
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
from tilefoundry_ascend.ascend.sil.op_analyzer import (
    OpAdvisoryExtractor,
    synthesize_default_policy,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CANNBENCH_TASKS = _REPO_ROOT.parent / "cann-bench" / "tasks"

_TRACKB_PREDICATE_CANARY = """\
#include <tl_templates/ascend/common.h>
#include <c_api/asc_simd.h>
#include <simt_api/asc_simt.h>

__simd_vf__ void vf_predicate_canary(__ubuf__ float* dst_ub, __ubuf__ float* scalar_ub, int rem) {
    vector_bool m32 = asc_create_mask_b32(PAT_ALL);
    vector_bool m16 = asc_create_mask_b16(PAT_ALL);
    vector_bool m32_tail = simd_inst::update_mask_b32(rem);
    vector_bool m16_tail = simd_inst::update_mask_b16(rem * 2);
    auto brc = simd_inst::vlds_brc_elem<float>(scalar_ub, 0);
    auto v0 = simd_inst::vdup<float>(1.0f, m32, MODE_ZEROING);
    auto v1 = simd_inst::vadd(v0, brc, m32_tail, MODE_ZEROING);
    auto v16 = simd_inst::vcvt<half>(v1, m16_tail, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
    (void)m16;
    (void)v16;
    simd_inst::vsts_1st(v1, dst_ub, 0, m32);
}

__simt_vf__ __launch_bounds__(128) void simt_tid_canary(
    __gm__ float* out,
    __gm__ const float* in,
    int n
) {
    uint32_t tid = threadIdx.x;
    if ((int)tid < n) {
        out[tid] = in[tid] + 1.0f;
    }
}

extern "C" __global__ __vector__ void canary_kernel(__gm__ float* out, __gm__ float* in, int n) {
    asc_vf_call<simt_tid_canary>(dim3(128, 1, 1), out, in, n);
}
"""


def _scrub_proto_names(proto: dict[str, object], alias: str) -> dict[str, object]:
    copied = copy.deepcopy(proto)
    for key in ("id", "name", "op_name", "cannbot_task", "schema"):
        if key in copied:
            copied[key] = alias
    op_block = copied.get("operator")
    if isinstance(op_block, dict):
        for key in ("id", "name", "schema"):
            if key in op_block:
                op_block[key] = alias
    return copied


def _load_first_case(op_name: str) -> dict[str, object]:
    cases_path = _REPO_ROOT / "tests" / "ascend" / "benchmarks" / op_name / "cases.yaml"
    cases = yaml.safe_load(cases_path.read_text(encoding="utf-8"))["cases"]
    assert isinstance(cases, list) and cases
    return dict(cases[0])


def _mte2_total(pipe_cycles: dict[str, int]) -> int:
    return sum(int(v) for k, v in pipe_cycles.items() if k.endswith("PIPE_MTE2"))


def _build_canary_primfunc(op_name: str) -> tuple[object, dict[str, int], bool]:
    if op_name == "rms_norm_2d":
        return _build_rms_norm_prim_func(d_dim=1024, tl_dtype="bfloat16", tile_m=4, num_stages=2), {"m_rows": 4096}, False
    if op_name == "rms_norm_1d_d2":
        return _build_rms_norm_prim_func(d_dim=2, tl_dtype="bfloat16", tile_m=256, num_stages=2), {"m_rows": 4096}, False
    if op_name == "softmax_2d":
        return _build_softmax_prim_func(d_dim=1024, tl_dtype="bfloat16", tile_m=2, num_stages=2), {"m_rows": 4096}, False
    if op_name == "swi_glu_2d":
        return _build_swi_glu_prim_func(d_dim=1024, tl_dtype="bfloat16"), {"m_rows": 4096}, False
    if op_name == "add_rms_norm_dynamic_quant":
        return _build_add_rms_norm_dynamic_quant_prim_func(d_dim=1024, tl_dtype="bfloat16"), {"m_rows": 4096}, False
    if op_name == "attention_residual":
        ar_n, ar_b, ar_s, ar_d = [int(x) for x in _load_first_case("attention_residual")["input_shape"][0]]
        return _build_attention_residual_prim_func(ar_n, ar_d, "bfloat16", 4, "tuned"), {"m_rows": ar_b * ar_s, "v_rows": ar_n * ar_b * ar_s}, False
    if op_name == "gqa":
        return _build_gqa_prim_func(n_q=32, n_kv=8, d_dim=128, tl_dtype="float16", bm=64, bn=128, l1_stages=2), {"b_dim": 2, "s_dim": 128, "s_kv": 128}, True
    if op_name == "mla":
        mla_b, mla_s, mla_nq, mla_dnope = [int(x) for x in _load_first_case("mla")["input_shape"][0]]
        return _build_mla_prim_func(mla_nq, 1, mla_dnope, 64, "float16", "BSND", True, 64, True, "tuned"), {"b_dim": mla_b, "q_rows": mla_b * mla_s, "s_dim": mla_s, "s_kv": mla_s}, True
    raise ValueError(f"unknown canary op: {op_name}")


@pytest.mark.skipif(not _CANNBENCH_TASKS.is_dir(), reason="cann-bench tasks missing")
def test_obfuscated_60_ops_zero_opname_invariance(tmp_path: Path) -> None:
    """Scrubbing all 60 CANNBench names preserves axes, contraction, routing, and regime plan."""
    assert not hasattr(capability_gate, "_CANNBENCH_OP_ROUTING")
    assert not hasattr(pipeline_solver, "fallback_from_sil")
    assert not hasattr(pipeline_solver, "_FallbackRooflineSummary")
    assert read_from_sil(tmp_path / "nonexistent_sil.json") is None

    proto_files = sorted(_CANNBENCH_TASKS.glob("level*/*/proto.yaml"))
    assert len(proto_files) == 60
    orig_extractor = OpAdvisoryExtractor(_CANNBENCH_TASKS)
    orig_perf = orig_extractor.perf_metadata
    anon_root = tmp_path / "anon_tasks"
    (anon_root / "metadata").mkdir(parents=True, exist_ok=True)
    anon_perf: dict[str, dict[str, object]] = {}
    anon_map: list[tuple[str, str, dict[str, object], dict[str, object]]] = []
    manifest_ops: list[dict[str, object]] = []

    for idx, pf in enumerate(proto_files):
        op_dir = pf.parent
        level_dir = op_dir.parent.name
        orig_name, alias = op_dir.name, f"anon_op_{idx:02d}"
        dest_dir = anon_root / level_dir / alias
        dest_dir.mkdir(parents=True, exist_ok=True)
        orig_proto = yaml.safe_load(pf.read_text(encoding="utf-8"))
        obf_proto = _scrub_proto_names(orig_proto, alias)
        (dest_dir / "proto.yaml").write_text(yaml.safe_dump(obf_proto), encoding="utf-8")
        if (cases_file := op_dir / "cases.yaml").is_file():
            shutil.copy2(cases_file, dest_dir / "cases.yaml")
        if isinstance(op_perf := orig_perf.get(level_dir, {}).get(orig_name), dict):
            anon_perf.setdefault(level_dir, {})[alias] = copy.deepcopy(op_perf)
        manifest_ops.append({"id": alias, "level": f"L{level_dir.removeprefix('level')}", "path": f"tasks/{level_dir}/{alias}"})
        anon_map.append((level_dir, orig_name, orig_proto, obf_proto))

    (anon_root / "metadata" / "950pr.json").write_text(json.dumps(anon_perf), encoding="utf-8")
    (anon_root / "suite_manifest.json").write_text(json.dumps({"operators": manifest_ops}), encoding="utf-8")
    anon_extractor = OpAdvisoryExtractor(anon_root)

    for idx, (level_dir, orig_name, orig_proto, obf_proto) in enumerate(anon_map):
        alias = f"anon_op_{idx:02d}"
        orig_adv, anon_adv = orig_extractor.extract(orig_name, level=level_dir), anon_extractor.extract(alias, level=level_dir)
        assert anon_adv.topology_axes == orig_adv.topology_axes, orig_name
        assert {k: v for k, v in anon_adv.structural_features.items() if k != "schema"} == {
            k: v for k, v in orig_adv.structural_features.items() if k != "schema"
        }, orig_name
        assert anon_adv.roofline == orig_adv.roofline, orig_name
        assert [r.representative_dims for r in anon_adv.regimes] == [r.representative_dims for r in orig_adv.regimes], orig_name
        orig_pol, anon_pol = synthesize_default_policy(orig_adv), synthesize_default_policy(anon_adv)
        assert [orig_pol.policies[t].l1_slot_aliasing for t in orig_pol.policies] == [
            anon_pol.policies[t].l1_slot_aliasing for t in anon_pol.policies
        ], orig_name
        orig_rt, anon_rt = route_from_op_spec(orig_proto), route_from_op_spec(obf_proto)
        assert (anon_rt.topology_axes, anon_rt.track, anon_rt.recommended_backend_tier, anon_rt.features) == (
            orig_rt.topology_axes, orig_rt.track, orig_rt.recommended_backend_tier, orig_rt.features
        ), orig_name


def test_regime_solver_zero_opname_and_strict_sil_contract() -> None:
    """Verify RegimePiecewiseSolver relies strictly on structural axes and flags."""
    solver = RegimePiecewiseSolver()
    axes = TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd")
    regime = {"regime_tag": "main", "representative_dims": {"M": 128, "N": 128, "K": 64}, "has_unaligned_tail": False}
    plan_mla = solver.solve_regime(regime, op_name="mla", topology_axes=axes)
    plan_vec = solver.solve_regime(regime, op_name="unrelated_vec_op", topology_axes=axes)
    assert plan_mla.policy == plan_vec.policy and plan_mla.policy["l1_slot_aliasing"] == {}
    plan_alias = solver.solve_regime(regime, op_name="unrelated_vec_op", topology_axes=axes, has_dead_l1_slice=True)
    assert plan_alias.policy["l1_slot_aliasing"] == {"reused_stage": "dead_l1_slice"}


def test_bisheng_dav3510_compiles_trackb_predicate_and_simt_canary() -> None:
    """Compile verified mask, broadcast, vadd, and SIMT threadIdx primitives on bisheng."""
    if (bisheng_bin := shutil.which("bisheng")) is None:
        pytest.skip("bisheng compiler not present on PATH")
    if not (tl_inc := _detect_tilelang_include_dir()):
        pytest.skip("tl_templates include directory not available")
    if not (cann_incs := _detect_cann_include_dirs(bisheng_bin=bisheng_bin)):
        pytest.skip("CANN include directories not available")

    with tempfile.TemporaryDirectory(prefix="tf_pred_canary_") as tmp_dir:
        src_path, obj_path = Path(tmp_dir) / "canary.asc", Path(tmp_dir) / "canary.o"
        src_path.write_text(_TRACKB_PREDICATE_CANARY, encoding="utf-8")
        cmd = [
            bisheng_bin, "--npu-arch=dav-3510", "--cce-aicore-only", "-std=c++20",
            f"-I{tl_inc}", *[f"-I{inc}" for inc in cann_incs], "-c", str(src_path), "-o", str(obj_path),
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        assert proc.returncode == 0, f"bisheng failed:\n{proc.stderr}\n{proc.stdout}"
        assert obj_path.is_file()


@pytest.mark.parametrize(
    "op_name",
    [
        "rms_norm_2d",
        "rms_norm_1d_d2",
        "softmax_2d",
        "swi_glu_2d",
        "add_rms_norm_dynamic_quant",
        "attention_residual",
        "gqa",
        "mla",
    ],
)
def test_reference_ops_anchor_specialization_and_mte2_nonzero(op_name: str) -> None:
    """Verify real-anchor specialization preserves Two-Snapshot diff and nonzero MTE2."""
    from tilelang import tvm  # noqa: PLC0415

    assert tvm.target.Target.current(allow_none=True) is None
    fn, bindings, expect_cube = _build_canary_primfunc(op_name)
    orig_shapes = [list(buf.shape) for buf in fn.buffer_map.values()]
    res = inspect_tilelang_primfunc(fn, var_bindings=bindings)
    assert res.lowering_error is None, f"{op_name} lowering error: {res.lowering_error}"
    assert res.ledger.is_legal is True, f"{op_name} illegal ledger"
    assert "tl.MaterializeMultiBuffer" in res.lowered_passes_observed
    assert "tl.MergeUBAllocations" in res.lowered_passes_observed
    assert _mte2_total(res.ledger.pipe_cycles) > 0 and res.ledger.min_gm_dma_bytes > 0
    if expect_cube:
        assert int(res.ledger.pipe_cycles.get("AIC.PIPE_M", 0)) > 0
    if op_name == "rms_norm_1d_d2":
        ub_wl = res.ledger.waterlines["UB"]
        assert ub_wl.explicit_buffers["x_ub"] == 1024 and ub_wl.explicit_buffers["y_ub"] == 1024
        assert res.ledger.min_gm_dma_bytes == 1024
    assert [list(buf.shape) for buf in fn.buffer_map.values()] == orig_shapes
    assert tvm.target.Target.current(allow_none=True) is None


def test_dynamic_2d_row_copy_specializes_for_anchor_bytes(tmp_path: Path) -> None:
    """Verify 2D dynamic T.copy(X[row, 0], ub) computes exact MTE2 bytes after specialization."""
    src = (
        "import tilelang.ascend.language as T\ndef make_row_copy():\n"
        "    m_rows = T.dynamic('m_rows')\n    @T.prim_func\n"
        "    def main(X: T.Tensor((m_rows, 1024), 'bfloat16'), Y: T.Tensor((m_rows, 1024), 'bfloat16')):\n"
        "        with T.Kernel(1) as _:\n            ub = T.alloc_shared((1024,), 'bfloat16')\n"
        "            for row in T.serial(m_rows):\n"
        "                T.copy(X[row, 0], ub)\n                T.copy(ub, Y[row, 0])\n"
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
    assert res.lowering_error is None and res.ledger.is_legal is True
    assert _mte2_total(res.ledger.pipe_cycles) > 0 and res.ledger.min_gm_dma_bytes == 2048
    assert res.ledger.dma_burst_efficiency_ratio > 0.0 and res.ledger.opaque_dynamic_regions == ()
    assert [list(buf.shape) for buf in fn.buffer_map.values()] == orig_shapes


def test_reference_ops_sil_verifier_and_trackb_canary() -> None:
    """Verify SILVerifier passes Track-B translation units and canonical op policies."""
    verifier, top_mask = SILVerifier(), ConstraintRelaxationMask.top()
    assert verifier.verify_source_against_facets(
        _render_gmm_trackb_tu(m_dim=128, k_dim=256, n_dim=512, tl_dtype="float16"), top_mask
    ).passed
    assert verifier.verify_source_against_facets(_render_mla_trackb_tu(), top_mask).passed

    bounds = HardwareLegalityBounds()
    aiv_axes = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
    cv_axes = TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd")
    ragged_axes = TopologyAxes("coupled_1c2v", "l0c_ub_epilogue", "ragged_group")
    checks = [
        ("rms_norm", aiv_axes, SchedulePolicy("aligned_packed", aiv_axes, tile_mnk=(4, 1024, 1), vector_mode="simdvf", ub_256kb_unlock=True, ub_estimated_bytes=22528), False),
        ("softmax", aiv_axes, SchedulePolicy("aligned_packed_regbase", aiv_axes, tile_mnk=(2, 1024, 1), vector_mode="simdvf", ub_256kb_unlock=True, ub_estimated_bytes=12288), True),
        ("gqa", cv_axes, SchedulePolicy("prefill", cv_axes, tile_mnk=(64, 128, 128), l1_stages=2, l0c_buffers=1, l1_estimated_bytes=98304, ub_estimated_bytes=77824, ub_256kb_unlock=True, cv_datapath="ub_l1_ring"), False),
        ("grouped_matmul", ragged_axes, SchedulePolicy("ragged_group", ragged_axes, tile_mnk=(64, 128, 64), l1_stages=2, l0c_buffers=1, l1_estimated_bytes=65536, ub_estimated_bytes=65536, ub_256kb_unlock=True, cv_datapath="l0c2ub_split", ensure_empty_subblock_sync=True), False),
    ]
    for op_name, axes, policy, has_nan_inf in checks:
        reg = RegimeSlice(policy.regime_tag, (1,), {"M": 128, "N": 128, "D": 1024}, False, axes.iteration_regularity == "ragged_group")
        adv = AdvisoryEnvelope(op_name, f"level4/{op_name}", 4, axes, ("float16",), ("float16",), {}, {}, (reg,), bounds, has_nan_inf_cases=has_nan_inf)
        rep = verifier.verify_envelope(PolicyEnvelope(op_name, {policy.regime_tag: policy}), adv)
        assert rep.passed is True, f"{op_name}: {rep.errors}"


def test_stage_b_driver_source_contiguous_context_and_no_target_none() -> None:
    """Verify build_stage_b_driver_source keeps tilelang.lower inside _ascend_compile_session."""
    for op_name, factory_name, dims in (
        ("rms_norm", "_build_rms_norm_prim_func", {"d_dim": 1024}),
        ("softmax", "_build_softmax_prim_func", {"d_dim": 1024}),
        ("gqa", "_build_gqa_prim_func", {"d_dim": 128}),
        ("grouped_matmul", "_select_grouped_matmul_config", {"k_dim": 256, "n_dim": 512}),
    ):
        cand_path = _REPO_ROOT / "examples" / "ascend950pr_ops" / op_name / "kernels.py"
        driver = build_stage_b_driver_source(cand_path, factory_name=factory_name, dims=dims)
        assert "_ascend_compile_session" in driver and "Target(None)" not in driver

        lower_inside = False

        def _visit(node: ast.AST, in_session: bool) -> None:
            nonlocal lower_inside
            nxt = in_session
            if isinstance(node, ast.With) and any(
                isinstance(it.context_expr, ast.Call)
                and isinstance(it.context_expr.func, ast.Name)
                and it.context_expr.func.id == "_ascend_compile_session"
                for it in node.items
            ):
                nxt = True
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "tilelang"
                and node.func.attr == "lower"
                and nxt
            ):
                lower_inside = True
            for ch in ast.iter_child_nodes(node):
                _visit(ch, nxt)

        _visit(ast.parse(driver), False)
        assert lower_inside is True, f"{op_name} tilelang.lower outside session"
