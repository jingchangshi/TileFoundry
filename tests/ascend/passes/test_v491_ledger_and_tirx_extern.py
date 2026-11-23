"""Tests for Track-B ledger parsing, single-pass lower, and tirx call_extern unwrapping."""

from __future__ import annotations

import contextlib
import importlib.util
import pathlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import tilelang.ascend.language as T

from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    _build_ascendc_static_ledger,
    _safe_eval_cpp_int,
    build_stage_b_driver_source,
    run_stage_b_ascendc_compile,
)
from tilefoundry_ascend.ascend.schedule.envelope_inspector import (
    _count_in_loop_pipe_all_barriers,
    _count_tirx_ub_scalar_compute_loops,
    _effective_tirx_call_name,
    inspect_tilelang_primfunc,
)


def _load_grouped_matmul_kernels() -> object:
    repo_root = Path(__file__).resolve().parents[3]
    mod_path = repo_root / "examples" / "ascend950pr_ops" / "grouped_matmul" / "kernels.py"
    spec = importlib.util.spec_from_file_location("v491_gmm_kernels_mod", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cpp_int_eval_strips_casts_and_rewrites_sizeof() -> None:
    """Verify _safe_eval_cpp_int strips C/static casts and rewrites sizeof widths."""
    env = {"BN": 64, "BM": 32}
    assert _safe_eval_cpp_int("(uint16_t)BN", env) == 64
    assert _safe_eval_cpp_int("(uint32_t)(BN * 2)", env) == 128
    assert _safe_eval_cpp_int("(int32_t)BM + (int)4", env) == 36
    assert _safe_eval_cpp_int("(size_t)sizeof(float)", env) == 4
    assert _safe_eval_cpp_int("static_cast<uint16_t>(BN) * sizeof(__bf16)", env) == 128
    assert _safe_eval_cpp_int("sizeof(int32_t) + sizeof(uint32_t)", env) == 8
    assert _safe_eval_cpp_int("sizeof(half) + sizeof(bfloat16_t)", env) == 4
    assert _safe_eval_cpp_int("sizeof(uint16_t) + sizeof(int16_t)", env) == 4
    assert _safe_eval_cpp_int("sizeof(int8_t) + sizeof(uint8_t)", env) == 2


def test_trackb_ledger_on_grouped_matmul_and_l0c2gm_arg_order() -> None:
    """Verify real grouped_matmul Track-B ledger and asc_copy_l0c2gm (cols, rows) widths."""
    gmm_mod = _load_grouped_matmul_kernels()
    tu_src = gmm_mod._render_trackb_tu(m_dim=128, k_dim=256, n_dim=512, tl_dtype="float16")
    gmm_ledger = _build_ascendc_static_ledger(tu_src)
    assert gmm_ledger.min_gm_dma_bytes > 0
    assert gmm_ledger.dominant_contig_bytes > 0
    assert 0.0 < gmm_ledger.dma_burst_efficiency_ratio <= 1.0

    for dst_type, elem_bytes in (("float", 4), ("__bf16", 2), ("int8_t", 1)):
        cpp_src = f"""
extern "C" __global__ __cube__ void l0c_probe(__gm__ {dst_type}* out_gm) {{
    __cc__ float* l0c = (__cc__ float*)(0x0000);
    asc_copy_l0c2gm(
        out_gm, l0c,
        (uint16_t)(32 + sizeof({dst_type})),
        (uint16_t)16,
        (uint32_t)64,
        (uint16_t)16,
        static_cast<asc_store_l2_cache_mode>(0),
        static_cast<asc_unit_flag_mode>(0),
        static_cast<asc_quant_mode>(0),
        static_cast<asc_relu_pre_mode>(0),
        false, true, false, false
    );
}}
"""
        cols = 32 + elem_bytes
        rows = 16
        led = _build_ascendc_static_ledger(cpp_src)
        assert led.dominant_contig_bytes == cols * elem_bytes
        assert led.min_gm_dma_bytes == cols * rows * elem_bytes


def test_run_stage_b_ascendc_compile_populates_generated_cce_on_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify run_stage_b_ascendc_compile keeps generated_cce on compile and object failure."""
    import tilefoundry_ascend.agent.orchestrator.gate_funnel as gf

    sample_cpp = 'extern "C" __global__ __vector__ void k(__gm__ float* out) { asc_init(); }\n'
    monkeypatch.setattr(gf, "_resolve_bisheng_bin", lambda *a, **kw: "/usr/bin/false")

    def _fail_run(cmd, **kw):
        del cmd, kw
        return SimpleNamespace(returncode=1, stdout="", stderr="kernel.asc:1:1: error: bad")

    monkeypatch.setattr(gf.subprocess, "run", _fail_run)
    res_compile_fail = run_stage_b_ascendc_compile(sample_cpp)
    assert res_compile_fail.passed is False
    assert res_compile_fail.generated_cce == sample_cpp

    def _ok_create_obj(cmd, **kw):
        del kw
        out_idx = cmd.index("-o") + 1
        Path(cmd[out_idx]).write_bytes(b"\x7fELF" + b"\x00" * 64)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gf.subprocess, "run", _ok_create_obj)
    monkeypatch.setattr(
        gf,
        "_verify_object_has_instantiated_kernel",
        lambda *a, **kw: (False, "stage_b_ascendc_uninstantiated_template: empty .text"),
    )
    monkeypatch.setattr(gf, "_disassemble_object", lambda *a, **kw: "")
    res_inst_fail = run_stage_b_ascendc_compile(sample_cpp)
    assert res_inst_fail.passed is False
    assert res_inst_fail.generated_cce == sample_cpp


def test_envelope_inspector_unwraps_tirx_call_extern_and_intrin() -> None:
    """Verify _effective_tirx_call_name unwraps call_extern/call_intrin in inspector passes."""
    from tvm import tirx

    for intrin in ("call_extern", "call_pure_extern", "call_intrin", "call_llvm_intrin"):
        for target_sym in ("simd_inst::vadd", "pipe_barrier", "asc_copy_gm2ubuf", "tl.ascend_copy"):
            fake_call = SimpleNamespace(op=f"tirx.{intrin}", args=[target_sym])
            assert _effective_tirx_call_name(fake_call) == target_sym

    @T.prim_func
    def extern_vec_kernel(
        x: T.Tensor((128,), "float32"),
        y: T.Tensor((128,), "float32"),
    ) -> None:
        with T.Kernel(1):
            ub = T.alloc_shared((128,), "float32")
            T.copy(x[0:128], ub)
            for _k in T.serial(4):
                T.call_extern("handle", "simd_inst::vadd", ub.data, ub.data, ub.data, 128)
                T.call_extern("handle", "pipe_barrier", "PIPE_ALL")
            T.copy(ub, y[0:128])

    assert isinstance(extern_vec_kernel, tirx.PrimFunc)
    assert _count_in_loop_pipe_all_barriers(extern_vec_kernel.body, tirx=tirx) == 1
    assert _count_tirx_ub_scalar_compute_loops(extern_vec_kernel.body, {"ub"}, tirx=tirx) == 0
    res = inspect_tilelang_primfunc(extern_vec_kernel)
    assert res.ledger.ub_scalar_compute_loops == 0
    assert res.ledger.in_loop_pipe_all_barriers == 1


def test_build_stage_b_driver_source_single_pass_lower(tmp_path: Path) -> None:
    """Verify build_stage_b_driver_source invokes tilelang.lower once on success."""
    kernel_py = tmp_path / "k.py"
    kernel_py.write_text("def make_kernel():\n    return 'dummy_fn'\n", encoding="utf-8")
    tir_dump = tmp_path / "out.tir"
    cce_dump = tmp_path / "out.cce"

    driver_src = build_stage_b_driver_source(
        kernel_py,
        factory_name="make_kernel",
        tir_dump_path=str(tir_dump),
        kernel_dump_path=str(cce_dump),
    )

    calls: list[bool] = []
    should_fail = [False]

    def _fake_lower(fn, *, target: str, enable_device_compile: bool):
        assert fn == "dummy_fn"
        assert target == "ascend"
        calls.append(enable_device_compile)
        if enable_device_compile and should_fail[0]:
            raise RuntimeError("device compile failed")
        return SimpleNamespace(device_mod="LOWERED_TIR", kernel_source="// CCE_SRC")

    fake_tilelang = SimpleNamespace(lower=_fake_lower)

    @contextlib.contextmanager
    def _fake_session(_pc):
        yield

    import json

    marker = "_cfg = json.loads("
    prefix, rest = driver_src.split(marker, 1)
    del prefix
    cfg_line, suffix = rest.split("\n", 1)
    cfg_obj = json.loads(eval(cfg_line.rstrip(")")))
    lower_idx = suffix.index("        with _ascend_compile_session(_merged_pc):")
    exec_body = suffix[lower_idx:]
    ns = {
        "_cfg": cfg_obj,
        "_fn": "dummy_fn",
        "_merged_pc": {},
        "pathlib": pathlib,
        "tilelang": fake_tilelang,
        "_ascend_compile_session": _fake_session,
    }
    exec("if True:\n" + exec_body, ns)
    assert calls == [True]
    assert tir_dump.read_text(encoding="utf-8") == "LOWERED_TIR"
    assert cce_dump.read_text(encoding="utf-8") == "// CCE_SRC"

    calls.clear()
    should_fail[0] = True
    tir_dump.unlink()
    cce_dump.unlink()
    with pytest.raises(RuntimeError, match="device compile failed"):
        exec("if True:\n" + exec_body, ns)
    assert calls == [True, False]
    assert tir_dump.read_text(encoding="utf-8") == "LOWERED_TIR"
    assert cce_dump.read_text(encoding="utf-8") == "// CCE_SRC"
