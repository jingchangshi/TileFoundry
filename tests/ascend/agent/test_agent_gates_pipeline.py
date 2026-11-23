"""Integration tests for GateFunnel Stage A/B custom kernels, --cce-res-usage, and HAP."""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from tilefoundry_ascend.agent.campaign import marathon, surrogate_cannbench
from tilefoundry_ascend.agent.campaign.anchor_specs import CannbenchOpsRegistry, RegimeTargetSpec
from tilefoundry_ascend.agent.campaign.campaign_runner import _extract_trackb_cpp_source
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import run_l1_compile_check
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    GateFunnel,
    StageAResult,
    StageBResult,
    StageCResult,
    _bounded_compile_diagnostic,
    _check_custom_kernel_redlines,
    _parallel_forbidden_math_calls,
    _parse_cce_launch_block_dim,
    _resolve_cpp_consts,
    check_npu_entrypoint_purity,
    check_repair_monotonicity,
    evaluate_ascendc_candidate,
    parse_cce_res_usage,
    resolve_trackb_compile_flags,
    run_stage_a_static,
    run_stage_b_ascendc_compile,
    run_stage_b_compile,
)
from tilefoundry_ascend.ascend.sil.pytorch_frontend import PyTorchSemanticAnalyzer
from tilefoundry_ascend.ascend.target import PURE_SIMDVF_UBUF_FLAGS
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    compute_eq3_hap_objective,
    compute_static_dynamic_residual,
    extract_structural_signature,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger
from tilefoundry_ascend.ascend.sil.schema import CasePhysicsRatios, ConstraintRelaxationMask

REPO_ROOT = Path(__file__).resolve().parents[3]

_PYTHON_KERNEL_MODULE = """
import torch

def kernel(x: torch.Tensor, gamma: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    xf = x.to(torch.float32)
    rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    return (xf * rms * gamma.to(torch.float32)).to(x.dtype)
"""

_ILLEGAL_ASCENDC_WRAPPER = """
ASCENDC_CODE = '''
// Illegal dav-3510 datapath: L1->GM and GM->L0A
'''
def kernel(x):
    return x
"""


_PRIM_FUNC_KERNEL_MODULE = """
import tilelang.ascend.language as T

@T.prim_func
def main(X: T.Tensor((1, 64), "float16"), Y: T.Tensor((1, 64), "float16")):
    with T.Kernel(1):
        x_ub = T.alloc_shared((1, 64), "float16")
        T.copy(X, x_ub)
        T.copy(x_ub, Y)
"""


def _parallel_math_module(expr: str, *, simt_vf: bool = False) -> str:
    """Render a minimal Track-A kernel whose Parallel body evaluates one scalar-math expression."""
    loop = f"        for i in T.Parallel(64):\n            x_ub[0, i] = {expr}\n"
    if simt_vf:
        loop = "        with T.SimtVF(threads=128):\n" + "".join(
            f"    {line}\n" for line in loop.splitlines()
        )
    return (
        "import tilelang.ascend.language as T\n"
        "\n"
        "@T.prim_func\n"
        'def main(X: T.Tensor((1, 64), "float16"), Y: T.Tensor((1, 64), "float16")):\n'
        "    with T.Kernel(1):\n"
        '        x_ub = T.alloc_shared((1, 64), "float16")\n'
        "        T.copy(X, x_ub)\n"
        f"{loop}"
        "        T.copy(x_ub, Y)\n"
    )


_CLEAN_EVALUATED_CPP = (
    "#include <tl_templates/ascend/common.h>\n"
    'extern "C" __global__ __vector__ void kernel_active(__gm__ float* x) {\n'
    "  x[0] = Reg::Exp(x[0]);\n"
    "}\n"
)

_LEGACY_SET_FLAG_CPP = (
    "#include <tl_templates/ascend/common.h>\n"
    'extern "C" __global__ __vector__ void kernel_active(__gm__ float* x) {\n'
    "  set_flag(PIPE_V, PIPE_MTE3, 0);\n"
    "  x[0] = 1.0f;\n"
    "}\n"
)

_LAUNCHED_UB_STRIDED_CPP = (
    "#include <tl_templates/ascend/common.h>\n"
    'extern "C" __global__ __vector__ void kernel_active(__gm__ float* x) {\n'
    "  __ubuf__ float* x_ub = (__ubuf__ float*)x;\n"
    "  x_ub[0] = *(x_ub + r * 256);\n"
    "}\n"
    'extern "C" int launch(void* stream) {\n'
    "  kernel_active<<<56, nullptr, stream>>>();\n"
    "  return 0;\n"
    "}\n"
)


def test_gate_funnel_python_kernel_module_and_cce_res_usage_parsing() -> None:
    """Verify Stage A/B reject plain Python callables and parse --cce-res-usage."""
    sa_py = run_stage_a_static(_PYTHON_KERNEL_MODULE)
    assert sa_py.passed is False
    assert sa_py.backend_kind == "python_kernel"
    assert "stage_a_non_kernel_callable" in sa_py.error_message

    sb_py = run_stage_b_compile(_PYTHON_KERNEL_MODULE)
    assert sb_py.passed is False

    sa_bad = run_stage_a_static(_ILLEGAL_ASCENDC_WRAPPER)
    assert sa_bad.passed is False
    assert "forbidden_dav3510_dma_path" in sa_bad.error_message

    spill, vloop = parse_cce_res_usage(
        "warning: Register spill: 128 bytes\nremark: VLOOPv2 miss: inner branch\n"
    )
    assert spill == 128
    assert vloop is False

    spill_clean, vloop_clean = parse_cce_res_usage("info: stack spill: 0 bytes\nVLOOPv2 hit\n")
    assert spill_clean == 0
    assert vloop_clean is True

    funnel = GateFunnel(
        skip_stage_b=False,
        compile_runner=lambda _src: StageBResult(passed=True, spill_bytes=0, vloopv2_hit=True),
        device_runner=lambda _src, st_a: StageCResult(
            passed=True,
            accuracy_passed=True,
            measured_latency_us=5.0,
            residual=compute_static_dynamic_residual(
                st_a.ledger,
                {
                    "dur_us": 5.0,
                    "spill_bytes": 256,
                    "bank_conflict_ratio": 0.25,
                    "l2_hit_rate": 0.60,
                },
            ),
            telemetry={"dur_us": 5.0, "spill_bytes": 256},
        ),
    )
    ev = funnel.evaluate(_PRIM_FUNC_KERNEL_MODULE)
    assert ev.passed is True
    assert ev.stage_c is not None and ev.stage_c.residual is not None
    assert ev.stage_c.residual.spill_bytes == 256
    assert ev.stage_c.residual.bank_conflict_ratio == 0.25
    assert ev.stage_c.residual.l2_hit_rate == 0.60

    clean_score = compute_eq3_hap_objective(
        5.0, ref_perf_us=15.0, t_hw_us=2.5, spill_bytes=0, bank_conflict_ratio=0.0, l2_hit_rate=1.0
    )
    penalized_score = compute_eq3_hap_objective(
        5.0, ref_perf_us=15.0, t_hw_us=2.5, residual=ev.stage_c.residual
    )
    assert 0.0 < penalized_score < clean_score

    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 50},
        waterlines={},
    )
    res = compute_static_dynamic_residual(ledger, {"dur_us": 5.0, "spill_bytes": 64, "l2_hit_rate": 0.5})
    sig = extract_structural_signature(
        _PYTHON_KERNEL_MODULE,
        residual=res,
        stage_b=StageBResult(passed=True, spill_bytes=64, vloopv2_hit=False),
    )
    assert "hw:register_spill" in sig
    assert "hw:low_l2_hit" in sig
    assert "hw:vloopv2_miss" in sig


def test_bounded_compile_diagnostic_keeps_primary_error_and_source_tail() -> None:
    primary = "/tmp/candidate_dsl.py:42:7: error: use of undeclared identifier 'tile_buf'"
    source_tail = "Source:\n" + "".join(
        f"    {index}: T.copy(X[{index}], x_ub)\n" for index in range(120)
    )
    output = (
        "note: lowering candidate kernel\n"
        f"{primary}\n"
        "note: CCE codegen failed\n"
        "owner: bisheng\n"
        f"{source_tail}"
    )
    assert len(output) > 1500

    diagnostic = _bounded_compile_diagnostic(output)
    assert len(diagnostic) <= 1500
    assert diagnostic.startswith(primary)
    assert "[... compiler output truncated ...]" in diagnostic
    assert source_tail.rstrip()[-1200:] in diagnostic
    assert "T.copy(X[119], x_ub)" in diagnostic

    short = "kernel.cce:7:1: error: expected ';'"
    assert _bounded_compile_diagnostic(short) == short

    tail_with_error = "Source:\n" + "x" * 2000 + f"\n{primary}"
    assert _bounded_compile_diagnostic(tail_with_error) == tail_with_error[-1500:]

    no_error = "Source:\n" + "y" * 2000
    assert _bounded_compile_diagnostic(no_error) == no_error[-1500:]

    bracket_error = "[ERROR] bisheng crashed\n" + "z" * 2000
    assert _bounded_compile_diagnostic(bracket_error).startswith("[ERROR] bisheng crashed")


def test_bounded_compile_diagnostic_keeps_context_for_fatal_and_tvm_errors() -> None:
    """Surface Fatal, TVMError, and fatal error headers with six adjacent context lines."""
    for header in (
        "Fatal: bisheng backend crashed",
        "TVMError: Check failed: lowered",
        "fatal error: 'simd_inst.h' file not found",
    ):
        lines = ["noise " + "n" * 20] * 300
        lines.append(header)
        lines.extend(f"ctx{index} " + "c" * 20 for index in range(12))
        lines.extend("tail " + "t" * 30 for _ in range(60))
        diagnostic = _bounded_compile_diagnostic("\n".join(lines))
        assert len(diagnostic) <= 1500
        assert diagnostic.startswith(header)
        for index in range(6):
            assert f"ctx{index} " in diagnostic
        assert "ctx6 " not in diagnostic
        assert "[... compiler output truncated ...]" in diagnostic


def test_stage_b_compile_results_keep_untruncated_full_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Archive raw compiler output while keeping the bounded summary in error_message."""
    primary = "/tmp/kernel.cce:9:3: error: use of undeclared identifier 'tile_buf'"
    diagnostic = f"{primary}\n" + "".join(f"Source: line {index}\n" for index in range(200))
    completed = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=diagnostic)

    def _fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return completed

    monkeypatch.setattr(
        "tilefoundry_ascend.agent.orchestrator.gate_funnel.subprocess.run", _fake_run
    )
    track_a = run_stage_b_compile("def kernel():\n    pass\n", python_bin=sys.executable)
    assert track_a.passed is False
    assert track_a.full_diagnostic == diagnostic
    assert primary in track_a.error_message
    assert "[... compiler output truncated ...]" in track_a.error_message
    assert len(track_a.error_message) <= len("stage_b_compile_failed (rc=1): ") + 1500

    bisheng = tmp_path / "bisheng"
    bisheng.write_text("#!/bin/sh\n", encoding="utf-8")
    include_dir = tmp_path / "include"
    include_dir.mkdir()
    track_b = run_stage_b_ascendc_compile(
        'extern "C" __global__ void kernel_active() {}\n',
        bisheng_bin=bisheng,
        extra_include_dirs=(include_dir,),
    )
    assert track_b.passed is False
    assert track_b.full_diagnostic == diagnostic
    assert primary in track_b.error_message
    assert "[... compiler output truncated ...]" in track_b.error_message
    assert (
        len(track_b.error_message)
        <= len("stage_b_ascendc_compile_failed (rc=1): ") + 1500
    )


def test_run_stage_a_static_rejects_scalar_math_in_plain_parallel_loops() -> None:
    """Reject cast and transcendental scalar math directly inside T.Parallel before lowering."""
    exp_module = _parallel_math_module("T.exp(x_ub[0, i])")
    cast_module = _parallel_math_module('T.Cast("float16", x_ub[0, i])')
    mixed_module = _parallel_math_module("T.log(T.exp(x_ub[0, i]))")
    simt_module = _parallel_math_module("T.exp(x_ub[0, i])", simt_vf=True)

    assert _parallel_forbidden_math_calls(ast.parse(_PRIM_FUNC_KERNEL_MODULE)) == []
    assert _parallel_forbidden_math_calls(ast.parse(exp_module)) == ["exp"]
    assert _parallel_forbidden_math_calls(ast.parse(cast_module)) == ["Cast"]
    assert _parallel_forbidden_math_calls(ast.parse(mixed_module)) == ["exp", "log"]
    assert _parallel_forbidden_math_calls(ast.parse(simt_module)) == []

    exp_result = run_stage_a_static(exp_module)
    assert exp_result.passed is False
    assert exp_result.prim_func is None
    assert exp_result.ledger is None
    assert exp_result.error_message == (
        "forbidden_tracka_parallel_scalar_math: T.Parallel body calls T.exp; move "
        "elementwise math under with T.SimdVF(): with T.simd.vcvt / T.simd.vexp, or "
        "under with T.SimtVF(threads=128): with T.cast"
    )

    cast_result = run_stage_a_static(cast_module)
    assert cast_result.passed is False
    assert "forbidden_tracka_parallel_scalar_math: T.Parallel body calls T.Cast;" in (
        cast_result.error_message
    )

    mixed_result = run_stage_a_static(mixed_module)
    assert mixed_result.passed is False
    assert "T.Parallel body calls T.exp, T.log;" in mixed_result.error_message

    simt_result = run_stage_a_static(simt_module)
    assert simt_result.passed is True


def test_evaluate_ascendc_candidate_rejects_forbidden_tokens_with_optional_baseline() -> None:
    """Reject absolute forbidden tokens while baseline comparison only rejects growth."""
    clean_result = evaluate_ascendc_candidate(_CLEAN_EVALUATED_CPP)
    assert clean_result.passed is True
    assert clean_result.error_message == ""

    bad_cpp = _CLEAN_EVALUATED_CPP.replace("Reg::Exp(x[0])", "AscendC::Exp(dst, x[0], mask)")
    bad_result = evaluate_ascendc_candidate(bad_cpp)
    assert bad_result.passed is False
    assert bad_result.error_message == "cce_forbidden_tokens: AscendC::"

    preserved_result = evaluate_ascendc_candidate(
        _LEGACY_SET_FLAG_CPP, baseline_cpp_source=_LEGACY_SET_FLAG_CPP
    )
    assert preserved_result.passed is True

    absolute_result = evaluate_ascendc_candidate(_LEGACY_SET_FLAG_CPP)
    assert absolute_result.passed is False
    assert absolute_result.error_message == "cce_forbidden_tokens: set_flag("

    grown_cpp = _LEGACY_SET_FLAG_CPP.replace(
        "  x[0] = 1.0f;\n", "  x[0] = 1.0f;\n  set_flag(PIPE_MTE3, PIPE_V, 0);\n"
    )
    grown_result = evaluate_ascendc_candidate(grown_cpp, baseline_cpp_source=_LEGACY_SET_FLAG_CPP)
    assert grown_result.passed is False
    assert grown_result.error_message == "cce_forbidden_tokens: set_flag("


def test_evaluate_ascendc_candidate_derives_wave_occupancy_and_ub_bank_risk() -> None:
    """Derive wave occupancy from <<<block_dim>>> and flag only 512-byte-aligned row strides."""
    launched = evaluate_ascendc_candidate(_LAUNCHED_UB_STRIDED_CPP)
    assert launched.passed is True
    assert launched.ledger is not None
    assert launched.ledger.grid_blocks == 56
    assert launched.ledger.wave_occupancy == 1.0
    assert launched.ledger.ub_bank_conflict_risk == 1.0

    narrow_stride = evaluate_ascendc_candidate(_LAUNCHED_UB_STRIDED_CPP.replace("r * 256", "r * 64"))
    assert narrow_stride.ledger is not None
    assert narrow_stride.ledger.grid_blocks == 56
    assert narrow_stride.ledger.ub_bank_conflict_risk == 0.0

    unlaunched_cpp = _LAUNCHED_UB_STRIDED_CPP.split('extern "C" int launch')[0]
    unlaunched = evaluate_ascendc_candidate(unlaunched_cpp)
    assert unlaunched.ledger is not None
    assert unlaunched.ledger.grid_blocks == 1
    assert unlaunched.ledger.wave_occupancy == 0.0

    ratios = CasePhysicsRatios(
        case_id=1,
        stride_ratio=1.0,
        contig_atom_ratio=1.0,
        is_atom_aligned=True,
        working_set_sram_ratio=0.5,
        working_set_vreg_ratio=0.5,
        wave_occupancy_ratio=1.0714,
        oi_ridge_ratio=1.0,
        inner_contiguous_bytes=64,
    )
    fallback = evaluate_ascendc_candidate(unlaunched_cpp, case_ratios=ratios)
    assert fallback.ledger is not None
    assert fallback.ledger.grid_blocks == 60
    assert fallback.ledger.wave_occupancy == 1.0714
    assert fallback.ledger.ub_bank_conflict_risk == 1.0


def test_softmax_rendered_trackb_tu_reports_launcher_wave_occupancy() -> None:
    """Read real Softmax launcher truth into grid, wave, and UB bank telemetry."""
    mod_path = REPO_ROOT / "examples" / "ascend950pr_ops" / "softmax" / "kernels.py"
    mod_src = mod_path.read_text(encoding="utf-8")
    mod_spec = importlib.util.spec_from_file_location("softmax_pipeline", mod_path)
    assert mod_spec is not None and mod_spec.loader is not None
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    for regime in CannbenchOpsRegistry.default().get_regime_specs("softmax"):
        extracted = _extract_trackb_cpp_source(mod_src, regime)
        assert extracted is not None
        raw_cpp = extracted[1]
        dims = dict(regime.representative_hir_dims or {})
        dims["trackb_src"] = raw_cpp
        instantiated = mod._render_trackb_tu(**dims)
        result = evaluate_ascendc_candidate(instantiated)
        assert result.ledger is not None
        assert (
            result.ledger.grid_blocks,
            result.ledger.wave_occupancy,
            result.ledger.ub_bank_conflict_risk,
        ) == (56, 1.0, 0.0)
        assert _check_custom_kernel_redlines(raw_cpp) == []


def test_run_l1_compile_check_keeps_primary_error_before_source_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / "env.sh"
    env_file.write_text("TILEFOUNDRY_NPU_TILELANG_SOURCE=/opt/tilelang\n", encoding="utf-8")
    primary = "/tmp/candidate_dsl.py:42:7: error: use of undeclared identifier 'tile_buf'"
    stderr = f"{primary}\n" + "".join(f"Source: line {index}\n" for index in range(200))
    completed = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=stderr)

    def _fake_run(
        argv: list[str],
        *,
        text: bool,
        capture_output: bool,
        cwd: str,
        env: dict[str, str],
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        return completed

    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", _fake_run)
    result = run_l1_compile_check(
        "def _build_exp_prim_func():\n    return None\n",
        builder_name="_build_exp_prim_func",
        builder_kwargs={},
        env_file=env_file,
    )

    assert result.passed is False
    assert result.error_message.startswith("l1_compile_failed (rc=1): ")
    assert primary in result.error_message
    assert "Source: line 199" in result.error_message
    assert len(result.error_message) <= len("l1_compile_failed (rc=1): ") + 1500


def test_resolve_trackb_compile_flags_and_kernel_threads_ast_check() -> None:
    """Strip SIMD-VF UBUF flags on __simt_vf__ and reject threads= only on T.Kernel."""
    simt_cpp = (
        "#include <tl_templates/ascend/common.h>\n"
        "__simt_vf__ void vf(__gm__ float* x) {}\n"
        'extern "C" __global__ __vector__ void k(__gm__ float* x) {}\n'
    )
    simt_flags = resolve_trackb_compile_flags(
        simt_cpp,
        [*PURE_SIMDVF_UBUF_FLAGS, "-O2", "-DKEEP_FLAG=1"],
    )
    assert simt_flags == ("-O2", "-DKEEP_FLAG=1")

    simd_cpp = (
        "#include <tl_templates/ascend/common.h>\n"
        "// commented __simt_vf__ token must not strip flags\n"
        "__simd_vf__ void vf(__ubuf__ float* x) {}\n"
    )
    simd_flags = resolve_trackb_compile_flags(simd_cpp, ["-DKEEP_FLAG=1"])
    for flag in PURE_SIMDVF_UBUF_FLAGS:
        assert flag in simd_flags
    assert "-DKEEP_FLAG=1" in simd_flags

    bad_kernel_threads = (
        "import tilelang.ascend.language as T\n\n"
        "@T.prim_func\n"
        'def main(X: T.Tensor((1, 64), "float16")):\n'
        "    with T.Kernel(1, threads=128) as cid:\n"
        "        pass\n"
    )
    res_bad = run_stage_a_static(bad_kernel_threads)
    assert res_bad.passed is False
    assert "forbidden_kernel_threads_kwarg" in res_bad.error_message

    simt_ok = _parallel_math_module("T.exp(x_ub[0, i])", simt_vf=True)
    assert run_stage_a_static(simt_ok).passed is True


def test_require_kernel_cache_gate_funnel_roofline_and_engine_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _KERNEL_CACHE purity, GateFunnel roofline threading, and Cube/DMA gates."""
    uncached_mod = (
        "import tilelang\nimport tilelang.ascend.language as T\n\n"
        "@T.prim_func\n"
        'def _main(X: T.Tensor((1, 64), "float16")):\n    pass\n\n'
        "def op_fn(x):\n"
        '    return tilelang.compile(_main, target="ascend")(x)\n\n'
        "kernel = op_fn\n"
    )
    assert check_npu_entrypoint_purity(uncached_mod, "op_fn") == []
    assert any(
        "missing_kernel_cache" in err
        for err in check_npu_entrypoint_purity(
            uncached_mod, "op_fn", require_kernel_cache=True
        )
    )

    decoy_cached_mod = "_KERNEL_CACHE: dict[object, object] = {}\n" + uncached_mod
    assert any(
        "unpopulated_kernel_cache" in err
        for err in check_npu_entrypoint_purity(
            decoy_cached_mod, "op_fn", require_kernel_cache=True
        )
    )

    readonly_cached_mod = (
        "import tilelang\nimport tilelang.ascend.language as T\n\n"
        "_KERNEL_CACHE: dict[object, object] = {}\n\n"
        "@T.prim_func\n"
        'def _main(X: T.Tensor((1, 64), "float16")):\n    pass\n\n'
        "def op_fn(x):\n"
        "    k = _KERNEL_CACHE.get(x.dtype)\n"
        '    return (k or tilelang.compile(_main, target="ascend"))(x)\n\n'
        "kernel = op_fn\n"
    )
    assert any(
        "unpopulated_kernel_cache" in err
        for err in check_npu_entrypoint_purity(
            readonly_cached_mod, "op_fn", require_kernel_cache=True
        )
    )

    populated_mod = (
        "import tilelang\nimport tilelang.ascend.language as T\n\n"
        "_KERNEL_CACHE: dict[object, object] = {}\n\n"
        "@T.prim_func\n"
        'def _main(X: T.Tensor((1, 64), "float16")):\n    pass\n\n'
        "def op_fn(x):\n"
        "    k = _KERNEL_CACHE.get(x.dtype)\n"
        "    if k is None:\n"
        '        k = tilelang.compile(_main, target="ascend")\n'
        "        _KERNEL_CACHE[x.dtype] = k\n"
        "    return k(x)\n\n"
        "kernel = op_fn\n"
    )
    assert check_npu_entrypoint_purity(
        populated_mod, "op_fn", require_kernel_cache=True
    ) == []

    funnel = GateFunnel(
        skip_stage_b=False,
        compile_runner=lambda _src: StageBResult(passed=True, spill_bytes=0, vloopv2_hit=True),
    )
    ev = funnel.evaluate(
        _PRIM_FUNC_KERNEL_MODULE,
        roofline={"min_gm_bytes": 16384, "compute_floor_us": 2.0},
    )
    assert ev.passed is True
    assert ev.stage_c is not None and ev.stage_c.residual is not None
    assert ev.stage_c.residual.observation is not None
    assert ev.stage_c.residual.transfer_inflation_ratio > 0.0
    assert ev.stage_c.residual.observation.transfer_inflation_ratio > 0.0

    cube_req = run_stage_a_static(_PRIM_FUNC_KERNEL_MODULE, require_cube_engine=True)
    assert cube_req.passed is False
    assert "missing_cube_engine_primitives" in cube_req.error_message

    mx_cpp = (
        "#include <tl_templates/ascend/common.h>\n"
        "#include <tl_templates/ascend/gemm.h>\n"
        'extern "C" __global__ __aicore__ void mm(__gm__ half* a, __gm__ half* b) {\n'
        "    asc_init();\n"
        "    asc_copy_gm2l1_nd2nz();\n"
        "    asc_mmad_mx();\n"
        "}\n"
    )
    mx_res = evaluate_ascendc_candidate(
        mx_cpp,
        freedom_mask=ConstraintRelaxationMask.top(),
        require_hardware_dma=True,
        require_cube_engine=True,
    )
    assert mx_res.passed is True, mx_res.error_message

    hybrid_mod = (
        "import torch\n"
        "import tilelang\n"
        "import tilelang.ascend.language as T\n\n"
        "_KERNEL_CACHE: dict[object, object] = {}\n"
        'TB_CU = """\n'
        "#include <tl_templates/ascend/common.h>\n"
        'extern "C" __global__ __vector__ void k(__gm__ half* x) {\n'
        "    asc_init();\n"
        "    asc_copy_gm2ub_align(nullptr, x, 64);\n"
        "}\n"
        '"""\n'
        "def _render_trackb_tu(**dims: object) -> str:\n"
        '    return str(dims.get("trackb_src", ""))\n'
        "def _select_cfg(**kw: object) -> dict[str, object]:\n"
        '    return {"compile_flags": []}\n'
        "def _build_fn(**kw: object):\n"
        "    @T.prim_func\n"
        '    def main(X: T.Tensor((1, 64), "float16")):\n        pass\n'
        "    return main\n"
        "def demo(x: torch.Tensor) -> torch.Tensor:\n"
        '    k_fn = _KERNEL_CACHE.get("k")\n'
        "    if k_fn is None:\n"
        '        k_fn = tilelang.compile(_build_fn(), target="ascend")\n'
        '        _KERNEL_CACHE["k"] = k_fn\n'
        "    return k_fn(x)\n"
        "kernel = demo\n"
    )
    spec = RegimeTargetSpec(
        regime_tag="mm",
        builder_candidates=("_build_fn",),
        sample_builder_kwargs={},
        representative_hir_dims={"M": 64},
        config_selector_fn="_select_cfg",
        runtime_backend="hybrid_trackb",
        trackb_symbols=("TB_CU",),
    )
    vec_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=10,
        pipe_cycles={"AIV0.PIPE_MTE2": 10},
        waterlines={},
    )
    cube_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=10,
        pipe_cycles={"AIC.PIPE_M": 10, "AIV0.PIPE_MTE2": 10},
        waterlines={},
    )
    monkeypatch.setattr(
        marathon,
        "evaluate_ascendc_candidate",
        lambda *_a, **_kw: StageAResult(passed=True, prim_func=None, ledger=vec_ledger),
    )
    monkeypatch.setattr(
        marathon, "run_stage_b_ascendc_compile", lambda *_a, **_kw: StageBResult(passed=True)
    )
    monkeypatch.setattr(
        marathon, "run_l1_compile_check", lambda *_a, **_kw: StageBResult(passed=True)
    )
    monkeypatch.setattr(
        marathon,
        "run_stage_a_static",
        lambda *_a, **_kw: StageAResult(passed=True, prim_func=None, ledger=vec_ledger),
    )
    sem_mm = PyTorchSemanticAnalyzer().analyze_source(
        "import torch\n"
        "def demo(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:\n"
        "    return torch.matmul(a, b)\n",
        entry_name="demo",
    )
    ok_both_vec, err_both_vec = marathon._validate_stage0_hybrid_track_b(
        hybrid_mod,
        spec,
        entrypoint="demo",
        worktree=tmp_path,
        env_file=None,
        semantics=sem_mm,
    )
    assert ok_both_vec is False
    assert "missing_cube_engine_primitives" in err_both_vec

    monkeypatch.setattr(
        marathon,
        "run_stage_a_static",
        lambda *_a, **_kw: StageAResult(passed=True, prim_func=None, ledger=cube_ledger),
    )
    ok_ta_cube, err_ta_cube = marathon._validate_stage0_hybrid_track_b(
        hybrid_mod,
        spec,
        entrypoint="demo",
        worktree=tmp_path,
        env_file=None,
        semantics=sem_mm,
    )
    assert ok_ta_cube is True, err_ta_cube


def test_scoped_mixed_gm_store_multiline_define_and_launcher_min_caps() -> None:
    """Scope mixed GM stores per function, unfold multiline #define, and parse std::min caps."""
    sibling_safe = (
        'extern "C" __global__ __vector__ void k1(\n'
        "    __gm__ float* __restrict__ out, __ubuf__ float* u) {\n"
        "    asc_copy_ub2gm_align(out, u, 128);\n"
        "}\n"
        'extern "C" __global__ __vector__ void k2(__gm__ float* out) {\n'
        "    out[0] = 1.0f;\n"
        "}\n"
    )
    assert _check_custom_kernel_redlines(sibling_safe) == []

    stride_same_name = (
        'extern "C" __global__ __vector__ void k_stride(\n'
        "    __gm__ float* out, __gm__ float* idx, __ubuf__ float* u) {\n"
        "    asc_copy_ub2gm_align(out + 1 * 4, u, 128);\n"
        "    idx[0] = 2.0f;\n"
        "}\n"
    )
    assert _check_custom_kernel_redlines(stride_same_name) == []

    auto_alias_mixed = (
        'extern "C" __global__ __vector__ void k_auto(\n'
        "    __gm__ float* __restrict__ out, __ubuf__ float* u) {\n"
        "    auto* p = out + 64;\n"
        "    asc_copy_ub2gm_align(out, u, 128);\n"
        "    p[0] = 0.0f;\n"
        "}\n"
        "void unrelated_flush(__gm__ float* q) { dcci(q); }\n"
    )
    assert any(
        "mixed_scalar_and_mte3_gm_store" in err
        for err in _check_custom_kernel_redlines(auto_alias_mixed)
    )

    callee_flushed = (
        "void helper_flush(__gm__ float* q) { dcci(q); }\n"
        'extern "C" __global__ __vector__ void k_ok(\n'
        "    __gm__ float* __restrict__ out, __ubuf__ float* u) {\n"
        "    auto* p = out + 64;\n"
        "    asc_copy_ub2gm_align(out, u, 128);\n"
        "    p[0] = 0.0f;\n"
        "    helper_flush(out);\n"
        "}\n"
    )
    assert _check_custom_kernel_redlines(callee_flushed) == []

    macro_cpp = (
        "#define BASE_W 32\n"
        "#define WIDE_W \\\n"
        "    (BASE_W + 16)\n"
        'extern "C" __global__ __vector__ void k(__gm__ half* x) {}\n'
        "void launch(__gm__ half* x, int rows, void* stream) {\n"
        "    const auto blocks = std::min(WIDE_W, rows);\n"
        "    k<<<blocks, nullptr, stream>>>(x);\n"
        "}\n"
    )
    consts = _resolve_cpp_consts(macro_cpp)
    assert consts["WIDE_W"] == 48
    assert _parse_cce_launch_block_dim(macro_cpp, consts) == 48


def test_l0_kernel_cache_key_pollution_audit() -> None:
    """Reject float hyperparameters and pointer/id values in _KERNEL_CACHE keys."""
    def _wrap_cache_mod(key_stmt: str) -> str:
        return (
            "import ctypes\n"
            "import tilelang\n"
            "import tilelang.ascend.language as T\n\n"
            "_KERNEL_CACHE: dict[object, object] = {}\n\n"
            "@T.prim_func\n"
            'def _main(X: T.Tensor((1, 64), "float16")):\n'
            "    pass\n\n"
            "def op_fn(x, eps: float = 1e-5, scale: float = 0.125):\n"
            f"    {key_stmt}\n"
            "    k = _KERNEL_CACHE.get(key)\n"
            "    if k is None:\n"
            '        k = tilelang.compile(_main, target="ascend")\n'
            "        _KERNEL_CACHE[key] = k\n"
            "    return k(x)\n\n"
            "kernel = op_fn\n"
        )

    clean_src = _wrap_cache_mod('key = (x.shape[-1], str(x.dtype), "aligned")')
    assert check_npu_entrypoint_purity(clean_src, "op_fn", require_kernel_cache=True) == []

    for bad_stmt in (
        "key = (x.shape[-1], str(x.dtype), eps)",
        "key = (x.shape[-1], float(scale))",
        "key = (x.shape[-1], 1e-6)",
        "key = (id(x), str(x.dtype))",
        "key = (x.data_ptr(), str(x.dtype))",
        "key = (ctypes.addressof(x), str(x.dtype))",
        "key = (x.storage_offset(), str(x.dtype))",
    ):
        bad_src = _wrap_cache_mod(bad_stmt)
        errs = check_npu_entrypoint_purity(bad_src, "op_fn", require_kernel_cache=True)
        assert any("polluted_kernel_cache_key" in err for err in errs), (bad_stmt, errs)

    direct_get_src = (
        "import tilelang\n"
        "import tilelang.ascend.language as T\n\n"
        "_KERNEL_CACHE: dict[object, object] = {}\n\n"
        "@T.prim_func\n"
        'def _main(X: T.Tensor((1, 64), "float16")):\n'
        "    pass\n\n"
        "def op_fn(x, sm_scale: float = 0.5):\n"
        "    k = _KERNEL_CACHE.get((x.shape[-1], sm_scale))\n"
        "    if k is None:\n"
        '        k = tilelang.compile(_main, target="ascend")\n'
        "        _KERNEL_CACHE[(x.shape[-1], sm_scale)] = k\n"
        "    return k(x)\n\n"
        "kernel = op_fn\n"
    )
    direct_errs = check_npu_entrypoint_purity(direct_get_src, "op_fn", require_kernel_cache=True)
    assert any("polluted_kernel_cache_key" in err for err in direct_errs), direct_errs


def test_check_repair_monotonicity_tracka_and_trackb() -> None:
    """Reject repairs dropping T.Pipelined or vector/cube primitives while allowing valid fixes."""
    parent_tracka = (
        "import tilelang.ascend.language as T\n\n"
        "@T.prim_func\n"
        'def main(X: T.Tensor((128, 64), "float16"), Y: T.Tensor((128, 64), "float16")):\n'
        "    with T.Kernel(1):\n"
        '        a_ub = T.alloc_shared((64, 64), "float16")\n'
        "        for ko in T.Pipelined(2, num_stages=2):\n"
        "            T.copy(X[ko * 64 : (ko + 1) * 64, :], a_ub)\n"
        "            with T.SimdVF():\n"
        "                mask = T.simd.pset(32)\n"
        "                v = T.simd.vld(a_ub[0])\n"
        "                T.simd.vsts(a_ub[0], T.simd.vexp(v, mask), mask)\n"
        "            T.copy(a_ub, Y[ko * 64 : (ko + 1) * 64, :])\n"
    )
    valid_halved = parent_tracka.replace("(64, 64)", "(32, 64)").replace(
        "num_stages=2", "num_stages=1"
    )
    assert check_repair_monotonicity(parent_tracka, valid_halved) == []

    dropped_pipe = parent_tracka.replace("T.Pipelined(2, num_stages=2)", "T.serial(2)")
    pipe_errs = check_repair_monotonicity(parent_tracka, dropped_pipe)
    assert any("monotonicity_regression_pipeline_dropped" in err for err in pipe_errs), pipe_errs

    zero_pipe = parent_tracka.replace("num_stages=2", "num_stages=0")
    zero_errs = check_repair_monotonicity(parent_tracka, zero_pipe)
    assert any("monotonicity_regression_pipeline_disabled" in err for err in zero_errs), zero_errs

    scalar_regressed = (
        "import tilelang.ascend.language as T\n\n"
        "@T.prim_func\n"
        'def main(X: T.Tensor((128, 64), "float16"), Y: T.Tensor((128, 64), "float16")):\n'
        "    with T.Kernel(1):\n"
        '        a_ub = T.alloc_shared((64, 64), "float16")\n'
        "        for ko in T.Pipelined(2, num_stages=2):\n"
        "            T.copy(X[ko * 64 : (ko + 1) * 64, :], a_ub)\n"
        "            for i in T.serial(64):\n"
        "                a_ub[0, i] = a_ub[0, i]\n"
        "            T.copy(a_ub, Y[ko * 64 : (ko + 1) * 64, :])\n"
    )
    scalar_errs = check_repair_monotonicity(parent_tracka, scalar_regressed)
    assert any("monotonicity_regression_scalar_fallback" in err for err in scalar_errs), scalar_errs

    parent_trackb = (
        "__simd_vf__ void vf_step(__ubuf__ float* a) {\n"
        "    simd_inst::vexp(a, a, 64);\n"
        "}\n"
    )
    valid_trackb = (
        "__simd_vf__ void vf_step(__ubuf__ float* a) {\n"
        "    simd_inst::vadd(a, a, a, 64);\n"
        "}\n"
    )
    bad_trackb = "void vf_step(__ubuf__ float* a) { for (int i = 0; i < 64; ++i) a[i] += 1.0f; }\n"
    assert check_repair_monotonicity(parent_trackb, valid_trackb, backend_kind="ascendc_trackb") == []
    tb_errs = check_repair_monotonicity(parent_trackb, bad_trackb, backend_kind="ascendc_trackb")
    assert any("monotonicity_regression_trackb_scalar_fallback" in err for err in tb_errs), tb_errs
