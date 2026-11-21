"""Integration tests for GateFunnel Stage A/B custom kernels, --cce-res-usage, and HAP."""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from tilefoundry_ascend.agent.campaign import surrogate_cannbench
from tilefoundry_ascend.agent.campaign.anchor_specs import CannbenchOpsRegistry
from tilefoundry_ascend.agent.campaign.campaign_runner import _extract_trackb_cpp_source
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import run_l1_compile_check
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    GateFunnel,
    StageBResult,
    StageCResult,
    _bounded_compile_diagnostic,
    _check_custom_kernel_redlines,
    _parallel_forbidden_math_calls,
    evaluate_ascendc_candidate,
    parse_cce_res_usage,
    run_stage_a_static,
    run_stage_b_ascendc_compile,
    run_stage_b_compile,
)
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    compute_eq3_hap_objective,
    compute_static_dynamic_residual,
    extract_structural_signature,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger
from tilefoundry_ascend.ascend.sil.schema import CasePhysicsRatios

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
        asc_devkit_include=include_dir,
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
    assert fallback.ledger.grid_blocks == 1
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
