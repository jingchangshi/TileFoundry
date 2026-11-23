"""Tests for M3 SIL facets/sync pairing, twin_bridge, proxy twin, and op _render_trackb_tu."""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch

from tilefoundry_ascend.agent.orchestrator.e2e_evolution import _wrap_funnel_with_proxy
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    GateFunnel,
    StageAResult,
    check_npu_entrypoint_purity,
)
from tilefoundry_ascend.agent.orchestrator.twin_bridge import (
    _synthesize_contract_tensor,
    verify_numerical_twin,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger
from tilefoundry_ascend.ascend.sil.schema import ConstraintRelaxationMask
from tilefoundry_ascend.ascend.sil.verifier import verify_source_against_facets

_REPO_ROOT = Path(__file__).resolve().parents[3]
_M3_OPS: tuple[str, ...] = (
    "softmax",
    "grouped_matmul",
    "mla",
    "mhc_sinkhorn",
    "apply_rotary_pos_emb",
    "q4_0_dequant_matmul",
    "kimi_delta_attention",
)

_ADD_HIR = (
    "from __future__ import annotations\n"
    "from tilefoundry.dsl import *\n"
    "from tilefoundry_ascend import AscendTarget\n"
    "TOPOS = (Topology('block', 1), Topology('aic', 1), "
    "Topology('aiv', 2), Topology('lane', 64))\n"
    "@module(entry='add_op', target=AscendTarget('ascend.950pr'), topologies=TOPOS)\n"
    "class AddMod:\n"
    "    @func\n"
    "    def add_op(a: Tensor[(4, 8), 'f32'], b: Tensor[(4, 8), 'f32']) -> Tensor[(4, 8), 'f32']:\n"
    "        return tf.cast(a + b, dtype='f32')\n"
)


def _load_op_module(op_name: str):
    op_path = _REPO_ROOT / "examples" / "ascend950pr_ops" / op_name / "kernels.py"
    spec = importlib.util.spec_from_file_location(f"_test_m3_{op_name}", str(op_path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, op_path.read_text(encoding="utf-8")


def test_sil_verifier_tl_templates_facets_and_sync_lock_pairing() -> None:
    """Verify tl_templates 7-facet locks, asc_sync/asc_lock pairing, and warn_symbolic=False."""
    top = ConstraintRelaxationMask.top()
    bot = ConstraintRelaxationMask.bottom()
    no_alg = ConstraintRelaxationMask(alg_compute_and_monoid=False)

    assert not verify_source_against_facets(
        "asc_mmad(c, a, b, 16, 16, 16, 0, true, false, true);",
        no_alg,
    ).passed
    assert not verify_source_against_facets(
        "ascend_gemm_l1<64, 64, 64, 32, false, half, float>(c, a, b, 1, 0, 0);",
        no_alg,
    ).passed
    assert not verify_source_against_facets(
        "asc_mmad_mx(c, a, b, 16, 16, 16, 0);",
        no_alg,
    ).passed
    assert verify_source_against_facets(
        "asc_mmad(c, a, b, 16, 16, 16, 0, true, false, true);",
        bot,
    ).passed
    assert verify_source_against_facets(
        "ascend_gemm_l1<64, 64, 64, 32, false, half, float>(c, a, b, 1, 0, 0);",
        bot,
    ).passed
    assert verify_source_against_facets(
        "asc_mmad_mx(c, a, b, 16, 16, 16, 0);",
        bot,
    ).passed
    assert not verify_source_against_facets(
        "asc_copy_gm2ub_align(d, s, 1, 64, 0, 0, 0, 0, 64, 64);",
        bot,
    ).passed
    assert verify_source_against_facets(
        "asc_copy_gm2ub_align(d, s, 1, 64, 0, 0, 0, 0, 64, 64);",
        bot.unlock("dma_and_cache_hierarchy"),
    ).passed
    assert not verify_source_against_facets(
        "simd_inst::vlds_norm<float>(p, 0);",
        bot,
    ).passed
    assert not verify_source_against_facets(
        "asc_sync_notify(PIPE_M, PIPE_FIX, EVENT_ID0); asc_sync_wait(PIPE_M, PIPE_FIX, EVENT_ID0);",
        bot,
    ).passed
    assert not verify_source_against_facets(
        "asc_sync_block_arrive(PIPE_FIX, 0); asc_sync_block_wait(PIPE_V, 0);",
        bot,
    ).passed
    assert not verify_source_against_facets(
        "asc_sync_pipe(PIPE_V);",
        bot,
    ).passed
    assert not verify_source_against_facets(
        "asc_lock(PIPE_MTE1, 0, ASC_LOCK_BLOCK); asc_unlock(PIPE_MTE1, 0, ASC_LOCK_BLOCK);",
        bot,
    ).passed

    assert not verify_source_against_facets(
        "asc_sync_notify(PIPE_M, PIPE_FIX, EVENT_ID0);",
        top,
    ).passed
    assert not verify_source_against_facets(
        "asc_sync_notify(PIPE_M, PIPE_FIX, EVENT_ID0); asc_sync_wait(PIPE_M, PIPE_FIX, EVENT_ID1);",
        top,
    ).passed
    assert not verify_source_against_facets(
        "void k(bool c) { if (c) { asc_sync_notify(PIPE_M, PIPE_FIX, ev); } asc_sync_wait(PIPE_M, PIPE_FIX, ev); }",
        top,
    ).passed
    assert not verify_source_against_facets(
        "asc_lock(PIPE_MTE1, slot, ASC_LOCK_BLOCK);",
        top,
    ).passed
    assert verify_source_against_facets(
        "asc_lock(PIPE_MTE1, slot, ASC_LOCK_BLOCK); asc_unlock(PIPE_MTE1, slot, ASC_LOCK_BLOCK);",
        top,
    ).passed

    sym_rep = verify_source_against_facets(
        "void k(int sid, int p) { if (ASCEND_IS_AIC) { CrossCoreSetFlag<0x2, PIPE_FIX>(sid ^ 1); } "
        "if (ASCEND_IS_AIV) { CrossCoreWaitFlag(p); } }",
        top,
    )
    assert sym_rep.passed is False
    assert sym_rep.errors


def test_twin_bridge_value_range_zero_avoidance_and_numerical_twin() -> None:
    """Verify param-name-free zero avoidance and verify_numerical_twin pass/fail behavior."""
    gen_a = torch.Generator(device="cpu")
    gen_a.manual_seed(7)
    t_scale = _synthesize_contract_tensor(
        (256,),
        "float32",
        None,
        gen_a,
        param_name="scale",
    )
    gen_b = torch.Generator(device="cpu")
    gen_b.manual_seed(7)
    t_other = _synthesize_contract_tensor(
        (256,),
        "float32",
        None,
        gen_b,
        param_name="custom_tensor_xyz",
    )
    assert t_scale is not None and t_other is not None
    assert torch.equal(t_scale, t_other)
    assert float(t_scale.abs().min().item()) >= 1e-4

    pos_gen = torch.Generator(device="cpu")
    pos_gen.manual_seed(11)
    t_pos = _synthesize_contract_tensor(
        (128,),
        "float32",
        [0.0, 1.0],
        pos_gen,
        param_name="custom_tensor_xyz",
    )
    assert t_pos is not None
    assert float(t_pos.min().item()) >= 0.0 and float(t_pos.max().item()) <= 1.0

    ok_cand = "def add_op(a, b):\n    return a + b\n"
    bad_cand = "def add_op(a, b):\n    return a - b\n"
    ok_res = verify_numerical_twin(_ADD_HIR, ok_cand)
    bad_res = verify_numerical_twin(_ADD_HIR, bad_cand)
    assert ok_res.passed is True, ok_res.error_message
    assert bad_res.passed is False
    assert "twin_numeric_mismatch" in bad_res.error_message


def test_e2e_evolution_proxy_device_runs_numerical_twin() -> None:
    """Verify _wrap_funnel_with_proxy rejects wrong math operators via CPU twin comparison."""
    funnel = GateFunnel(skip_stage_b=True)
    wrapped = _wrap_funnel_with_proxy(funnel, None, hir_source=_ADD_HIR, anchor_dims={})
    assert wrapped._device_runner is not None
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=32,
        pipe_cycles={"AIV0.PIPE_V": 32},
        waterlines={},
    )
    stage_a = StageAResult(passed=True, prim_func=None, ledger=ledger)

    ok_stage_c = wrapped._device_runner("def add_op(a, b):\n    return a + b\n", stage_a)
    bad_stage_c = wrapped._device_runner("def add_op(a, b):\n    return a - b\n", stage_a)
    assert ok_stage_c.passed is True
    assert ok_stage_c.accuracy_passed is True
    assert bad_stage_c.passed is False
    assert bad_stage_c.accuracy_passed is False
    assert "twin_numeric_mismatch" in bad_stage_c.error_message


def test_m3_ops_render_trackb_tu_purity_and_cann_discovery() -> None:
    """Verify all 7 M3 ops expose _render_trackb_tu, pass purity, and satisfy SILVerifier."""
    top = ConstraintRelaxationMask.top()
    for op_name in _M3_OPS:
        mod, raw_src = _load_op_module(op_name)
        assert check_npu_entrypoint_purity(raw_src, op_name) == []
        render_fn = getattr(mod, "_render_trackb_tu", None)
        assert callable(render_fn), f"{op_name} missing _render_trackb_tu"
        tu = render_fn()
        assert isinstance(tu, str) and 'extern "C"' in tu
        rep = verify_source_against_facets(tu, top)
        assert rep.passed, f"{op_name} _render_trackb_tu failed SILVerifier: {rep.errors}"

    softmax_mod, _ = _load_op_module("softmax")
    assert callable(getattr(softmax_mod, "_cann_include_flags", None))
    asc_home = os.environ.get("ASCEND_HOME_PATH", "").strip()
    if asc_home and Path(asc_home).is_dir():
        flags = softmax_mod._cann_include_flags()
        assert any(flag.startswith("-I") for flag in flags)


def test_m3_rendered_trackb_tu_live_bisheng_instantiation() -> None:
    """Verify bisheng compiles _render_trackb_tu() into non-empty .text kernel symbols."""
    bisheng = shutil.which("bisheng")
    nm_bin = shutil.which("llvm-nm") or shutil.which("nm")
    if not bisheng or not nm_bin:
        pytest.skip("bisheng or nm not available")
    import tilelang  # noqa: PLC0415

    tl_inc = str(Path(tilelang.__file__).resolve().parent.parent / "src")
    softmax_mod, _ = _load_op_module("softmax")
    inc_flags = [f"-I{tl_inc}", *softmax_mod._cann_include_flags()]

    for op_name in ("softmax", "grouped_matmul", "mla"):
        mod, _ = _load_op_module(op_name)
        tu = mod._render_trackb_tu()
        with tempfile.TemporaryDirectory(prefix=f"tf_m3_{op_name}_") as tmp_dir:
            asc_path = Path(tmp_dir) / "kernel.asc"
            obj_path = Path(tmp_dir) / "kernel.o"
            asc_path.write_text(tu, encoding="utf-8")
            proc = subprocess.run(
                [
                    bisheng,
                    "--npu-arch=dav-3510",
                    "--cce-aicore-only",
                    "-O2",
                    "-std=c++20",
                    *inc_flags,
                    "-c",
                    str(asc_path),
                    "-o",
                    str(obj_path),
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            assert proc.returncode == 0, f"{op_name} bisheng failed: {proc.stderr[-800:]}"
            nm_proc = subprocess.run(
                [nm_bin, "--defined-only", str(obj_path)],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            defined_text = [
                ln.strip()
                for ln in (nm_proc.stdout or "").splitlines()
                if re.search(r"\b[Tt]\s+\S+", ln)
            ]
            assert defined_text, f"{op_name} produced zero defined .text symbols"
