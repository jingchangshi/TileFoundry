"""Compilation canary verifying Track-B predicate and SIMT primitives on bisheng."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    _detect_cann_include_dirs,
    _detect_tilelang_include_dir,
)

_CANARY_ASC_SOURCE = """\
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


def test_bisheng_dav3510_compiles_trackb_predicate_and_simt_canary() -> None:
    """Compile verified mask, broadcast, vadd, and SIMT threadIdx primitives on bisheng."""
    bisheng_bin = shutil.which("bisheng")
    if bisheng_bin is None:
        pytest.skip("bisheng compiler not present on PATH")
    tl_inc = _detect_tilelang_include_dir()
    if not tl_inc:
        pytest.skip("tl_templates include directory not available")
    cann_incs = _detect_cann_include_dirs(bisheng_bin=bisheng_bin)
    if not cann_incs:
        pytest.skip("CANN include directories not available")

    with tempfile.TemporaryDirectory(prefix="tf_pred_canary_") as tmp_dir:
        src_path = Path(tmp_dir) / "canary.asc"
        obj_path = Path(tmp_dir) / "canary.o"
        src_path.write_text(_CANARY_ASC_SOURCE, encoding="utf-8")
        cmd = [
            bisheng_bin,
            "--npu-arch=dav-3510",
            "--cce-aicore-only",
            "-std=c++20",
            f"-I{tl_inc}",
            *[f"-I{inc}" for inc in cann_incs],
            "-c",
            str(src_path),
            "-o",
            str(obj_path),
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        assert proc.returncode == 0, f"bisheng failed:\n{proc.stderr}\n{proc.stdout}"
        assert obj_path.is_file()
