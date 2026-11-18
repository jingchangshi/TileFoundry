"""End-to-end tests for MECE 7-Facet Lattice Top Element (ConstraintRelaxationMask.top()) across all 5 layers."""

from __future__ import annotations

import os
from pathlib import Path
import shutil

import pytest

from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    GateFunnel,
    StageBResult,
    run_stage_b_ascendc_compile,
    synthesize_pybind_extension_wrapper,
)
from tilefoundry_ascend.ascend.cost_model import BishengResourceSummary
from tilefoundry_ascend.ascend.sil import (
    ConstraintRelaxationMask,
    TopologyAxes,
    verify_source_against_facets,
)
from tilefoundry_ascend.ascend.templates import (
    SHELL_AIV_SIMD_STREAM,
    SHELL_CV_RING_PIPELINE,
    ComposableComputeSlotSpec,
    materialize_top_translation_unit,
)

def _has_real_bisheng() -> bool:
    if shutil.which("bisheng") is not None:
        return True
    for env_key in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH"):
        raw = os.environ.get(env_key, "").strip()
        if raw and (Path(raw).expanduser() / "bin" / "bisheng").is_file():
            return True
    return False


# ---------------------------------------------------------------------------
# Layer 1 & 2: Self-Contained Translation-Unit Materialization
# ---------------------------------------------------------------------------


def test_materialize_top_tu_from_slot_shell_and_lowered_tilelang() -> None:
    """Materialize standalone AscendC TUs with zero locked shell wrappers from slot, shell, and lowered PrimFunc."""
    slot = ComposableComputeSlotSpec(
        op_name="custom_1c2v_fused",
        freedom_mask=ConstraintRelaxationMask.bottom(),
        coordination_shell=SHELL_CV_RING_PIPELINE,
        slot_name="ring_step",
        compute_body_cpp=(
            "if ASCEND_IS_AIC {\n"
            "    CrossCoreSetFlag<0x2, PIPE_FIX>(0);\n"
            "}\n"
            "if ASCEND_IS_AIV {\n"
            "    CrossCoreWaitFlag(0);\n"
            "    y[0] = x[0] + 1.0f;\n"
            "}"
        ),
    )
    tu_slot = materialize_top_translation_unit(
        slot,
        host_signature="void launch_custom_1c2v_fused(void* stream, void* x, void* y)",
    )
    assert '#include "kernel_operator.h"' in tu_slot
    assert 'extern "C" __global__ __aicore__' in tu_slot
    assert "__mix__(1, 2)" in tu_slot
    assert "launch_custom_1c2v_fused" in tu_slot
    assert "CUBE_TILE_M" not in tu_slot
    assert "Synthesized Compute Slot" not in tu_slot
    assert "unrestricted_top=True" in tu_slot

    # Materialized 1C2V TU must pass SILVerifier at ConstraintRelaxationMask.top()
    rep_top = verify_source_against_facets(tu_slot, ConstraintRelaxationMask.top())
    assert rep_top.passed is True, rep_top.errors

    # Materializing from a canonical TopologyAxes shell removes locked contract wrappers
    tu_shell = materialize_top_translation_unit(
        SHELL_AIV_SIMD_STREAM,
        policy={"row_pack_factor": 4, "ub_256kb_unlock": True},
        host_signature={"name": "launch_simd_stream", "args": [("float*", "x"), ("float*", "y")]},
    )
    assert '#include "kernel_operator.h"' in tu_shell
    assert 'extern "C" __global__ __aicore__' in tu_shell
    assert "launch_simd_stream" in tu_shell
    assert "Synthesized Compute Slot" not in tu_shell

    # Materializing from a lowered Track-A TileLang PrimFunc kernel_source
    import tilelang
    from tilefoundry_ascend.agent.orchestrator.gate_funnel import _resolve_prim_func_from_dsl

    dsl = """
import tilelang.ascend.language as T

@T.prim_func
def vec_copy_add(A: T.Tensor((512,), "float32"), B: T.Tensor((512,), "float32"), C: T.Tensor((512,), "float32")):
    with T.Kernel(1):
        a_ub = T.alloc_shared((512,), "float32")
        b_ub = T.alloc_shared((512,), "float32")
        c_ub = T.alloc_shared((512,), "float32")
        T.copy(A, a_ub)
        T.copy(B, b_ub)
        with T.SimdVF():
            mask = T.simd.pset(32)
            for i in range(8):
                va = T.simd.vld(a_ub[i * 64])
                vb = T.simd.vld(b_ub[i * 64])
                T.simd.vsts(c_ub[i * 64], T.simd.vadd(va, vb, mask), mask)
        T.copy(c_ub, C)
"""
    pf = _resolve_prim_func_from_dsl(dsl)
    art = tilelang.lower(pf, target="ascend", enable_device_compile=False)
    assert art.kernel_source
    tu_tl = materialize_top_translation_unit(
        "vec_copy_add",
        lowered_kernel_cpp=art.kernel_source,
        host_signature="void launch_vec_copy_add(void* stream, void* A, void* B, void* C)",
    )
    assert '#include "kernel_operator.h"' in tu_tl
    assert 'extern "C" __global__ __aicore__' in tu_tl
    assert "vec_copy_add_kernel" in tu_tl
    assert "launch_vec_copy_add" in tu_tl
    assert verify_source_against_facets(tu_tl, ConstraintRelaxationMask.top()).passed is True


# ---------------------------------------------------------------------------
# Layer 3: Soundness vs Completeness of Static CFG Flag Check at top()
# ---------------------------------------------------------------------------


def test_cfg_flag_balance_1c2v_core_roles_and_symbolic_flags() -> None:
    """Verify 1C2V complementary ASCEND_IS_AIC/ASCEND_IS_AIV pairing and symbolic flag warnings at top()."""
    m_top = ConstraintRelaxationMask.top()
    m_sync = ConstraintRelaxationMask.bottom().unlock("pipeline_and_sync_graph")

    # 1. Complementary if (ASCEND_IS_AIC) / if (ASCEND_IS_AIV) sequential blocks
    tu_1c2v_seq = """
    void Kernel1C2V() {
        if (ASCEND_IS_AIC) {
            CrossCoreSetFlag<0x2, PIPE_FIX>(0);
        }
        if (ASCEND_IS_AIV) {
            CrossCoreWaitFlag(0);
        }
    }
    """
    rep_seq = verify_source_against_facets(tu_1c2v_seq, m_top)
    assert rep_seq.passed is True, rep_seq.errors
    assert rep_seq.errors == ()
    assert rep_seq.warnings == ()

    # 2. Complementary if (ASCEND_IS_AIC) { SetFlag(0); } else { WaitFlag(0); }
    tu_1c2v_else = """
    void Kernel1C2VElse() {
        if (ASCEND_IS_AIC) {
            CrossCoreSetFlag<0x2, PIPE_FIX>(0);
        } else {
            CrossCoreWaitFlag(0);
        }
    }
    """
    rep_else = verify_source_against_facets(tu_1c2v_else, m_top)
    assert rep_else.passed is True, rep_else.errors
    assert rep_else.warnings == ()

    # 3. Complementary if ASCEND_IS_AIC ... else if ASCEND_IS_AIV ... inside loop
    tu_1c2v_loop = """
    void Kernel1C2VLoop(int n) {
        for (int k = 0; k < n; ++k) {
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<0x2, PIPE_FIX>(0);
                CrossCoreWaitFlag(1);
            } else if ASCEND_IS_AIV {
                CrossCoreWaitFlag(0);
                CrossCoreSetFlag<0x2, PIPE_V>(1);
            }
        }
    }
    """
    rep_loop = verify_source_against_facets(tu_1c2v_loop, m_top)
    assert rep_loop.passed is True, rep_loop.errors
    assert rep_loop.warnings == ()

    # 4. Literal integer mismatch across 1C2V branches MUST still fail with errors
    tu_1c2v_bad_id = """
    void Kernel1C2VBadId() {
        if (ASCEND_IS_AIC) {
            CrossCoreSetFlag<0x2, PIPE_FIX>(0);
        }
        if (ASCEND_IS_AIV) {
            CrossCoreWaitFlag(1);
        }
    }
    """
    rep_bad_id = verify_source_against_facets(tu_1c2v_bad_id, m_top)
    assert rep_bad_id.passed is False
    assert len(rep_bad_id.errors) >= 1

    # 5. Literal flag ID > max_flag_depth MUST still fail with errors
    tu_over_depth = """
    void KernelOverDepth() {
        if (ASCEND_IS_AIC) {
            CrossCoreSetFlag<0x2, PIPE_FIX>(16);
        }
        if (ASCEND_IS_AIV) {
            CrossCoreWaitFlag(16);
        }
    }
    """
    assert verify_source_against_facets(tu_over_depth, m_top, max_flag_depth=15).passed is False

    # 6. Arbitrary non-core-role branches remain strictly rejected even at top()
    tu_non_role = """
    void KernelArbitrary(bool a, bool b) {
        if (a) { CrossCoreSetFlag<0x2, PIPE_FIX>(0); }
        if (b) { CrossCoreWaitFlag(0); }
    }
    """
    assert verify_source_against_facets(tu_non_role, m_top).passed is False

    # 7. Dynamic runtime symbolic expressions at top() emit warnings instead of false-positive errors
    tu_symbolic_1c2v = """
    void KernelDynamicFlag(int sid, int parity_flag) {
        if (ASCEND_IS_AIC) {
            CrossCoreSetFlag<0x2, PIPE_FIX>(sid ^ 1);
        }
        if (ASCEND_IS_AIV) {
            CrossCoreWaitFlag(parity_flag);
        }
    }
    """
    rep_sym_top = verify_source_against_facets(tu_symbolic_1c2v, m_top)
    assert rep_sym_top.passed is True, rep_sym_top.errors
    assert rep_sym_top.errors == ()
    assert len(rep_sym_top.warnings) >= 1

    # Helper-function dynamic symbolic expressions at top() also emit warnings
    tu_symbolic_helpers = """
    void ProduceStep(int sid) {
        CrossCoreSetFlag<0x2, PIPE_FIX>(sid ^ 1);
    }
    void ConsumeStep(int parity_flag) {
        CrossCoreWaitFlag(parity_flag);
    }
    """
    rep_helpers_top = verify_source_against_facets(tu_symbolic_helpers, m_top)
    assert rep_helpers_top.passed is True
    assert len(rep_helpers_top.warnings) >= 1

    # Non-top mask still strictly rejects unproved symbolic asymmetry as errors
    rep_sym_nontop = verify_source_against_facets(tu_symbolic_1c2v, m_sync)
    assert rep_sym_nontop.passed is False
    assert len(rep_sym_nontop.errors) >= 1


# ---------------------------------------------------------------------------
# Layer 4 & 5: Real bisheng Stage-B Compilation, GateFunnel Dispatch & Host ABI
# ---------------------------------------------------------------------------


def test_stage_b_ascendc_real_bisheng_compile_and_gate_funnel_dispatch() -> None:
    """Compile materialized top() AscendC TU with real bisheng and verify GateFunnel dispatch."""
    if not _has_real_bisheng():
        pytest.skip("bisheng compiler not present on disk")

    slot = ComposableComputeSlotSpec(
        op_name="real_bisheng_1c2v",
        freedom_mask=ConstraintRelaxationMask.top(),
        coordination_shell=TopologyAxes("coupled_1c2v", "bidirectional_cv_ring", "affine_simd"),
        compute_body_cpp=(
            "if ASCEND_IS_AIC {\n"
            "    CrossCoreSetFlag<0x2, PIPE_FIX>(0);\n"
            "}\n"
            "if ASCEND_IS_AIV {\n"
            "    CrossCoreWaitFlag(0);\n"
            "    y[0] = x[0] + 1.0f;\n"
            "}"
        ),
    )
    tu_cpp = materialize_top_translation_unit(
        slot,
        host_signature="void launch_real_bisheng_1c2v(void* stream, void* x, void* y)",
    )

    sb_ok = run_stage_b_ascendc_compile(tu_cpp)
    assert sb_ok.passed is True, sb_ok.error_message
    assert isinstance(sb_ok.compiler_summary, BishengResourceSummary)
    assert sb_ok.elapsed_sec >= 0.0

    # GateFunnel with backend_kind="ascendc" automatically routes Stage B to bisheng
    funnel = GateFunnel(skip_stage_b=False)
    ev_ok = funnel.evaluate(
        tu_cpp,
        backend_kind="ascendc",
        freedom_mask=ConstraintRelaxationMask.top(),
    )
    assert ev_ok.passed is True, ev_ok.error_message
    assert ev_ok.stage_a.passed is True
    assert ev_ok.stage_b is not None and ev_ok.stage_b.passed is True

    # Broken C++ fails cleanly at Stage B with real bisheng diagnostics
    bad_cpp = """
#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ __vector__ void broken_k(__gm__ float* x) {
    undeclared_symbol_xyz = 42;
}
"""
    ev_bad = funnel.evaluate(
        bad_cpp,
        backend_kind="ascendc_template",
        freedom_mask=ConstraintRelaxationMask.top(),
    )
    assert ev_bad.passed is False
    assert ev_bad.failed_stage == "stage_b"
    assert ev_bad.stage_b is not None and ev_bad.stage_b.passed is False
    assert "undeclared_symbol_xyz" in ev_bad.error_message


def test_stage_b_ascendc_graceful_no_bisheng_and_pybind_abi_wrapper() -> None:
    """Verify enable_device_compile=False fallback and torch_npu pybind extension ABI helper."""
    sb_disabled = run_stage_b_ascendc_compile(
        "extern \"C\" __global__ __aicore__ void k() {}",
        bisheng_bin="/nonexistent/bin/bisheng",
        enable_device_compile=False,
    )
    assert sb_disabled.passed is True
    assert sb_disabled.compiler_summary == BishengResourceSummary()

    sb_missing = run_stage_b_ascendc_compile(
        "extern \"C\" __global__ __aicore__ void k() {}",
        bisheng_bin="/nonexistent/bin/bisheng",
        enable_device_compile=True,
    )
    assert sb_missing.passed is False
    assert "bisheng_not_found" in sb_missing.error_message

    wrapper = synthesize_pybind_extension_wrapper(
        "rms_norm_top",
        "extern \"C\" void launch_rms_norm_top(aclrtStream stream, void* x, void* gamma, void* out)",
        ["x", "gamma", "out"],
    )
    assert "c10_npu::getCurrentNPUStream" in wrapper
    assert "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)" in wrapper
    assert 'm.def("rms_norm_top"' in wrapper
    assert "launch_rms_norm_top(npu_stream.stream(), x.data_ptr(), gamma.data_ptr(), out.data_ptr());" in wrapper
    assert "return out;" in wrapper
