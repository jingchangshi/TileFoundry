"""Tests for M3 instantiated Track-B L1 compilation, ELF verification, and telemetry."""

from __future__ import annotations

import inspect
import subprocess
import tempfile
from pathlib import Path

import pytest

from tilefoundry_ascend.agent.orchestrator import gate_funnel as gf
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    GateFunnel,
    _build_ascendc_static_ledger,
    _detect_cann_include_dirs,
    _parse_cce_res_usage_with_summary,
    _resolve_bisheng_bin,
    _verify_object_has_instantiated_kernel,
    parse_cce_res_usage,
    run_stage_a_static,
    run_stage_b_ascendc_compile,
    run_stage_b_compile,
)
from tilefoundry_ascend.ascend.sil import ConstraintRelaxationMask


def _require_live_bisheng() -> str:
    bisheng = _resolve_bisheng_bin()
    if bisheng is None:
        pytest.skip("bisheng compiler not available in current environment")
    return bisheng


def _require_tl_templates() -> Path:
    tl_inc = gf._detect_tilelang_include_dir()
    if not tl_inc or not (Path(tl_inc) / "tl_templates" / "ascend" / "common.h").is_file():
        pytest.skip("tl_templates/ascend headers not available in current environment")
    return Path(tl_inc)


_TRACKB_INSTANTIATED_VEC_TU = r"""
#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/simd_inst.h>

template <int TILE_M, int TILE_N>
__simd_vf__ inline void vf_scale_add(__ubuf__ float* x_ub, __ubuf__ float* y_ub, int chunks) {
    vector_bool m32 = asc_create_mask_b32(PAT_ALL);
    for (int c = 0; c < chunks; ++c) {
        int off = c << 6;
        auto vx = simd_inst::vlds_norm<float>(x_ub + off, 0);
        auto vy = simd_inst::vadd(vx, vx, m32, MODE_ZEROING);
        simd_inst::vsts_norm(vy, y_ub + off, 0, m32);
    }
}

template <int TILE_M, int TILE_N>
__global__ __vector__ void instantiated_vec_kernel(__gm__ float* x, __gm__ float* y) {
    asc_init();
    constexpr int ELEMS = TILE_M * TILE_N;
    constexpr int BYTES = ELEMS * 4;
    __ubuf__ float* bx = (__ubuf__ float*)0;
    __ubuf__ float* by = (__ubuf__ float*)(BYTES);
    asc_copy_gm2ub_align(
        (__ubuf__ uint8_t*)bx, (__gm__ uint8_t*)x, 1, BYTES,
        0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), BYTES, BYTES);
    asc_sync_notify(PIPE_MTE2, PIPE_V, (event_t)0);
    asc_sync_wait(PIPE_MTE2, PIPE_V, (event_t)0);
    vf_scale_add<TILE_M, TILE_N>(bx, by, ELEMS >> 6);
    asc_sync_notify(PIPE_V, PIPE_MTE3, (event_t)0);
    asc_sync_wait(PIPE_V, PIPE_MTE3, (event_t)0);
    asc_copy_ub2gm_align(
        (__gm__ uint8_t*)y, (__ubuf__ uint8_t*)by, 1, BYTES,
        static_cast<asc_store_l2_cache_mode>(4), BYTES, BYTES);
}

template __global__ __vector__ void instantiated_vec_kernel<4, 64>(__gm__ float*, __gm__ float*);
"""


_TRACKB_SCALAR_LOOP_TU = r"""
#include <tl_templates/ascend/common.h>

template <int N>
__global__ __vector__ void scalar_loop_kernel(__gm__ int* x, __gm__ int* y) {
    asc_init();
    int acc = 0;
    for (int i = 0; i < N; ++i) {
        acc += x[i];
    }
    y[0] = acc;
}

template __global__ __vector__ void scalar_loop_kernel<64>(__gm__ int*, __gm__ int*);
"""


_TRACKB_UNINSTANTIATED_TU = r"""
#include <tl_templates/ascend/common.h>

template <int N>
__global__ __vector__ void uninstantiated_tmpl_kernel(__gm__ float* x, __gm__ float* y) {
    asc_init();
    y[0] = x[0];
}
"""


def test_live_bisheng_instantiated_trackb_l1_telemetry_and_ledger() -> None:
    """Verify concrete C++20 Track-B template compiles with real objdump/res-usage telemetry."""
    _require_live_bisheng()
    _require_tl_templates()

    inc_dirs = _detect_cann_include_dirs()
    assert len(inc_dirs) >= 1

    sb = run_stage_b_ascendc_compile(_TRACKB_INSTANTIATED_VEC_TU)
    assert sb.passed is True, sb.error_message
    assert sb.compiler_summary is not None
    assert sb.compiler_summary.scalar_regs > 0
    assert sb.compiler_summary.vec_regs >= 1
    assert sb.compiler_summary.vloopv2_loops >= 1
    assert sb.compiler_summary.software_loops == 0
    assert sb.vloopv2_hit is True
    assert sb.microarch_profile is not None
    assert sb.microarch_profile.get("fp_alu_ops", 0) >= 0

    tu_with_const = (
        "constexpr int TILE_M = 4;\nconstexpr int TILE_N = 64;\n" + _TRACKB_INSTANTIATED_VEC_TU
    )
    ledger = _build_ascendc_static_ledger(
        tu_with_const,
        compiler_summary=sb.compiler_summary,
    )
    assert ledger.analysis_mode == "readonly_envelope"
    assert ledger.is_legal is True
    assert ledger.steady_state_ii is not None and ledger.steady_state_ii > 0
    assert "UB" in ledger.waterlines
    assert ledger.waterlines["UB"].explicit_bytes == 2048
    assert ledger.waterlines["UB"].explicit_bytes != 65536
    assert ledger.pipe_cycles["AIV0.PIPE_MTE2"] > 0
    assert ledger.pipe_cycles["AIV0.PIPE_MTE3"] > 0
    assert ledger.pipe_cycles["AIV0.PIPE_V"] > 0

    funnel = GateFunnel(skip_stage_b=False)
    ev = funnel.evaluate(
        tu_with_const,
        backend_kind="ascendc_template",
        freedom_mask=ConstraintRelaxationMask.top(),
    )
    assert ev.passed is True, ev.error_message
    assert ev.stage_b is not None and ev.stage_b.vloopv2_hit is True
    assert ev.stage_c is not None and ev.stage_c.residual is not None
    assert ev.stage_c.residual.vloopv2_hit is True


def test_live_bisheng_scalar_loop_reports_vloopv2_miss() -> None:
    """Verify kernel without VLOOPv2 hardware loop reports vloopv2_hit=False accurately."""
    _require_live_bisheng()
    _require_tl_templates()

    sb_sw = run_stage_b_ascendc_compile(_TRACKB_SCALAR_LOOP_TU)
    assert sb_sw.passed is True, sb_sw.error_message
    assert sb_sw.compiler_summary is not None
    assert sb_sw.compiler_summary.vloopv2_loops == 0
    assert sb_sw.compiler_summary.software_loops >= 1
    assert sb_sw.vloopv2_hit is False

    spill_no_ev, hit_no_ev, summary_no_ev = _parse_cce_res_usage_with_summary(
        "Vector Register Usage: 16\nScalar Register Usage: 12\nStack Frame Size: 64 bytes\n"
    )
    assert spill_no_ev == 0
    assert summary_no_ev.vloopv2_loops == 0
    assert hit_no_ev is False

    spill_ok, hit_ok = parse_cce_res_usage("Stack Frame Size: 64 bytes\nVLOOPv2 hit\n")
    assert spill_ok == 0
    assert hit_ok is True


def test_live_bisheng_rejects_uninstantiated_template_object() -> None:
    """Verify uninstantiated C++20 template .o is rejected by ELF symbol check and Stage B."""
    bisheng = _require_live_bisheng()
    tl_inc = _require_tl_templates()

    sb_uninst = run_stage_b_ascendc_compile(_TRACKB_UNINSTANTIATED_TU)
    assert sb_uninst.passed is False
    assert "uninstantiated_template" in sb_uninst.error_message

    with tempfile.TemporaryDirectory(prefix="tf_uninst_check_") as tmp_dir:
        src = Path(tmp_dir) / "uninst.asc"
        obj = Path(tmp_dir) / "uninst.o"
        src.write_text(_TRACKB_UNINSTANTIATED_TU, encoding="utf-8")
        inc_flags = [f"-I{tl_inc}"] + [f"-I{d}" for d in _detect_cann_include_dirs()]
        cmd = [
            bisheng,
            "--npu-arch=dav-3510",
            "--cce-aicore-only",
            "--cce-res-usage",
            "-O2",
            "-std=c++20",
            *inc_flags,
            "-c",
            str(src),
            "-o",
            str(obj),
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        assert proc.returncode == 0, proc.stderr
        assert obj.is_file()
        ok, err = _verify_object_has_instantiated_kernel(obj, bisheng_bin=bisheng)
        assert ok is False
        assert "uninstantiated_template" in err

    broken_inst = _TRACKB_INSTANTIATED_VEC_TU.replace(
        "vf_scale_add<TILE_M, TILE_N>(bx, by, ELEMS >> 6);",
        "NONEXISTENT_FN_12345(bx, by);",
    )
    sb_broken = run_stage_b_ascendc_compile(broken_inst)
    assert sb_broken.passed is False
    assert "NONEXISTENT_FN_12345" in sb_broken.error_message


def test_rejects_plain_python_and_bypasses_removed() -> None:
    """Verify plain Python callables, enable_device_compile=False, and guards are absent."""
    plain_py = """
import torch

def kernel(x: torch.Tensor) -> torch.Tensor:
    return x + 1.0
"""
    sa = run_stage_a_static(plain_py)
    assert sa.passed is False
    assert sa.ledger is None
    assert "stage_a_non_kernel_callable" in sa.error_message

    sb_py = run_stage_b_compile(plain_py)
    assert sb_py.passed is False

    with pytest.raises(ValueError, match="enable_device_compile=False"):
        run_stage_b_ascendc_compile(
            "extern \"C\" __global__ __aicore__ __vector__ void k() {}",
            enable_device_compile=False,
        )

    with pytest.raises(ValueError, match="enable_device_compile=False"):
        run_stage_b_compile(plain_py, enable_device_compile=False)

    gf_src = inspect.getsource(gf)
    assert "202002L" not in gf_src
    assert "_build_custom_kernel_ledger" not in gf_src


def test_ascendc_static_ledger_folds_constexpr_and_l1_cube_waterlines() -> None:
    """Verify _build_ascendc_static_ledger folds C++20 constexprs into UB/L1 waterlines."""
    cube_tu = """
constexpr int CUBE_TILE_M = 64;
constexpr int CUBE_TILE_N = 128;
constexpr int CUBE_TILE_K = 64;
constexpr int A_STAGE = CUBE_TILE_M * CUBE_TILE_K;
constexpr int B_STAGE = CUBE_TILE_K * CUBE_TILE_N;
constexpr int C_UB_BYTES = (CUBE_TILE_M / 2) * CUBE_TILE_N * 4;
__cbuf__ half* a_l1 = (__cbuf__ half*)0;
__cbuf__ half* b_l1 = (__cbuf__ half*)(2 * A_STAGE * 2);
__cbuf__ half* b_l1_s1 = b_l1 + B_STAGE;
__ubuf__ float* c_ub0 = (__ubuf__ float*)0;
__ubuf__ float* c_ub1 = (__ubuf__ float*)(C_UB_BYTES);
__simd_vf__ void epilogue() {}
void run() {
    asc_copy_gm2l1_nd2nz();
    asc_mmad();
    asc_copy_l0c2ub();
    simd_inst::vadd();
    asc_copy_ub2gm_align();
}
"""
    ledger = _build_ascendc_static_ledger(cube_tu)
    assert ledger.is_legal is True
    assert ledger.steady_state_ii == max(ledger.pipe_cycles.values())
    assert "L1" in ledger.waterlines
    assert ledger.waterlines["L1"].explicit_bytes == 2 * 4096 * 2 + 2 * 8192 * 2
    assert ledger.waterlines["UB"].explicit_bytes == 2 * 16384
    assert ledger.pipe_cycles["AIC.PIPE_M"] > 0
    assert ledger.pipe_cycles["AIC.PIPE_FIX"] > 0
    assert ledger.pipe_cycles["AIV0.PIPE_MTE2"] > 0
    assert ledger.pipe_cycles["AIV0.PIPE_MTE3"] > 0
    assert ledger.pipe_cycles["AIV0.PIPE_V"] > 0
