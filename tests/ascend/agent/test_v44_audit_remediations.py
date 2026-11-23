"""Regression tests for Track-B contracts, NaN diagnostics, co-edit splicing, and gates."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from textwrap import dedent
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tests.ascend.benchmarks import l3_contract
from tests.ascend.benchmarks.cannbench_contract import (
    _format_nonfinite_diagnostic,
    _wrap_comparator_with_nonfinite_diag,
)
from tilefoundry_ascend.agent.campaign.anchor_specs import (
    AnchorCaseSpec,
    RegimeTargetSpec,
    build_semantic_regime_specs,
)
from tilefoundry_ascend.agent.campaign.campaign_runner import (
    CampaignConfig,
    PerfTargetCampaignRunner,
    patch_regime_into_module,
)
from tilefoundry_ascend.agent.campaign.seed_synthesizer import (
    OracleAccuracyResult,
    _render_signature,
    _seed_parameter_specs,
    prepare_operator_seed_or_takeover,
    select_fallback_track_from_semantics,
    select_initial_track_from_semantics,
)
from tilefoundry_ascend.agent.campaign import surrogate_cannbench
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    _NPU_COMPARE_SHIM_SOURCE,
    AnchorEvalMetrics,
    CandidateSurrogateResult,
    run_l0_l1_round_filter,
)
from tilefoundry_ascend.agent.kb import (
    SymptomFamily,
    build_predicate_context,
    load_seed_rules,
    rank_symptom_families,
)
from tilefoundry_ascend.agent.orchestrator import llm_client
from tilefoundry_ascend.agent.orchestrator.e2e_evolution import run_e2e_evolution
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    GateFunnel,
    StageAResult,
    StageBResult,
    StageCResult,
    _bounded_compile_diagnostic,
    _build_ascendc_static_ledger,
    _check_custom_kernel_redlines,
    _parse_cce_launch_block_dim,
    _resolve_cpp_consts,
    evaluate_ascendc_candidate,
    run_stage_b_ascendc_compile,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import (
    GenerationRequest,
    KernelLLMBackend,
    MockLLMBackend,
    StreamWatchdogConfig,
    create_llm_backend_from_env,
)
from tilefoundry_ascend.agent.orchestrator.residual_beam import compute_static_dynamic_residual
from tilefoundry_ascend.ascend.cost_model.bisheng_res_parser import BishengResourceSummary
from tilefoundry_ascend.ascend.cost_model.experience_lake import ExperienceLake
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import (
    HirBlueprintPacket,
    HirRooflineSummary,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger
from tilefoundry_ascend.ascend.sil.pytorch_frontend import PyTorchSemanticAnalyzer
from tilefoundry_ascend.ascend.sil.schema import ConstraintRelaxationMask
from tilefoundry_ascend.ascend.target import PURE_SIMDVF_UBUF_FLAGS
from tilefoundry_ascend.ascend.target.dialect_surface import (
    CCE_CPP20_SURFACE,
    TILELANG_DSL_SURFACE,
)

_RMS_GOLDEN = dedent(
    """
    import torch

    def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        var = torch.mean(x.float() * x.float(), dim=-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + eps)).to(x.dtype) * weight
    """
)


class _ScriptedStage0Backend(KernelLLMBackend):
    def __init__(self, replies: Sequence[str]) -> None:
        self._replies = list(replies)
        self.requests: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> str:
        self.requests.append(request)
        if not self._replies:
            raise RuntimeError("exhausted scripted replies")
        return self._replies.pop(0)


def test_cce_surface_arities_non_global_launch_and_mixed_store_redlines() -> None:
    """Validate aligned DMA arities, CUDA/subblock bans, chevron targets, and MTE3/PIPE_S mix."""
    forbidden_src = (
        "__shared__ float smem[64];\n"
        "int s = get_subblocknum();\n"
        "int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "__syncthreads();\n"
    )
    hits = CCE_CPP20_SURFACE.count_forbidden_tokens(forbidden_src)
    for token in (
        "__shared__",
        "get_subblocknum",
        "blockIdx",
        "blockDim",
        "threadIdx",
        "__syncthreads",
    ):
        assert hits[token] >= 1

    valid_arities_src = dedent(
        """
        #include <tl_templates/ascend/common.h>
        extern "C" __global__ __vector__ void vec_k(__gm__ half* x, __gm__ half* y) {
            asc_init();
            __ubuf__ half* ux = (__ubuf__ half*)0;
            __ubuf__ half* uy = (__ubuf__ half*)4096;
            asc_copy_gm2ub_align(ux, x, 256);
            asc_copy_gm2ub_align((__ubuf__ uint8_t*)ux, (__gm__ uint8_t*)x, 1, 256, 0, 0, 0, 0);
            asc_copy_gm2ub_align((__ubuf__ uint8_t*)ux, (__gm__ uint8_t*)x, 1, 256, 0, 0, 0, 0, 256, 256);
            asc_copy_ub2gm_align(y, uy, 256);
            asc_copy_ub2gm_align((__gm__ uint8_t*)y, (__ubuf__ uint8_t*)uy, 1, 256, 4, 256, 256);
            asc_copy_ub2gm_align((__gm__ uint8_t*)y, (__ubuf__ uint8_t*)uy, 1, 256, 0, 0, 0, 0);
        }
        """
    )
    assert _check_custom_kernel_redlines(valid_arities_src) == []

    bad_gm2ub = "__global__ void k(__gm__ half* x, __ubuf__ half* u) { asc_copy_gm2ub_align(u, x, 1, 256); }"
    assert any("expected 3/8/10 args" in err for err in _check_custom_kernel_redlines(bad_gm2ub))

    bad_ub2gm = "__global__ void k(__gm__ half* y, __ubuf__ half* u) { asc_copy_ub2gm_align(y, u, 1, 256, 0); }"
    assert any("expected 3/7/8 args" in err for err in _check_custom_kernel_redlines(bad_ub2gm))

    non_global_launch = dedent(
        """
        __aicore__ void helper_k(__gm__ half* y) {}
        void launch(__gm__ half* y, void* stream) {
            helper_k<<<8, nullptr, stream>>>(y);
        }
        """
    )
    assert any(
        "non_global_chevron_launch" in err
        for err in _check_custom_kernel_redlines(non_global_launch)
    )

    valid_global_launch = dedent(
        """
        extern "C" __global__ __vector__ void kernel_active(__gm__ half* y) {}
        void launch(__gm__ half* y, void* stream) {
            kernel_active<<<8, nullptr, stream>>>(y);
        }
        """
    )
    assert _check_custom_kernel_redlines(valid_global_launch) == []

    mixed_store = dedent(
        """
        extern "C" __global__ __vector__ void mixed_k(__gm__ float* out, __ubuf__ float* u) {
            __gm__ float* alias_out = out;
            asc_copy_ub2gm_align(out, u, 128);
            alias_out[0] = 0.0f;
        }
        """
    )
    assert any(
        "mixed_scalar_and_mte3_gm_store" in err
        for err in _check_custom_kernel_redlines(mixed_store)
    )

    uncalled_flush = mixed_store + "\nvoid flush_fn(__gm__ float* p) { dcci(p); }\n"
    assert any(
        "mixed_scalar_and_mte3_gm_store" in err
        for err in _check_custom_kernel_redlines(uncalled_flush)
    )

    flushed_store = dedent(
        """
        void flush_fn(__gm__ float* p) { dcci(p); }
        extern "C" __global__ __vector__ void mixed_k(__gm__ float* out, __ubuf__ float* u) {
            __gm__ float* alias_out = out;
            asc_copy_ub2gm_align(out, u, 128);
            alias_out[0] = 0.0f;
            flush_fn(out);
        }
        """
    )
    assert _check_custom_kernel_redlines(flushed_store) == []
    assert any("reshape/view as 3D" in note for note in TILELANG_DSL_SURFACE.prompt_notes)


def test_nonfinite_comparator_diagnostic_on_zero_error_nan_and_inf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Surface NaN position and Inf sign/position coordinates when metrics report 0.0 diff."""
    actual_nan = torch.tensor([[1.0, float("nan")], [2.0, 3.0]], dtype=torch.float32)
    golden_ok = torch.tensor([[1.0, 0.0], [2.0, 3.0]], dtype=torch.float32)
    nan_diag = _format_nonfinite_diagnostic(actual_nan, golden_ok)
    assert "NaN position mismatch" in nan_diag
    assert "actual_nan=1/4" in nan_diag
    assert "first_coord=[0, 1]" in nan_diag

    actual_inf = torch.tensor([float("inf"), 1.0], dtype=torch.float32)
    golden_ninf = torch.tensor([float("-inf"), 1.0], dtype=torch.float32)
    inf_diag = _format_nonfinite_diagnostic(actual_inf, golden_ninf)
    assert "Inf mismatch" in inf_diag
    assert "first_coord=[0]" in inf_diag

    def _fake_compare(**kwargs: object) -> object:
        _ = kwargs
        return SimpleNamespace(passed=False, message="MERE=0.000000e+00, MARE=0.000000e+00")

    wrapped = _wrap_comparator_with_nonfinite_diag(_fake_compare)
    res = wrapped(case_id="c1", actual=actual_nan, golden=golden_ok)
    assert res.passed is False
    assert "NaN position mismatch" in res.message
    assert "first_coord=[0, 1]" in res.message

    pkg_mod = ModuleType("kernel_eval")
    utils_mod = ModuleType("kernel_eval.utils")
    cmp_mod = ModuleType("kernel_eval.utils.compare")

    def _stub_compare_tensors(
        actual: object,
        golden: object,
        *,
        dtype: str = "float32",
        native_output: object = None,
        custom_thresholds: object = None,
    ) -> object:
        _ = actual, golden, dtype, native_output, custom_thresholds
        return SimpleNamespace(passed=False, error_msg="", mere=0.0, mare=0.0, max_diff=0.0)

    setattr(cmp_mod, "compare_tensors", _stub_compare_tensors)
    setattr(utils_mod, "compare", cmp_mod)
    setattr(pkg_mod, "utils", utils_mod)
    monkeypatch.setitem(sys.modules, "kernel_eval", pkg_mod)
    monkeypatch.setitem(sys.modules, "kernel_eval.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "kernel_eval.utils.compare", cmp_mod)

    rendered_shim = _NPU_COMPARE_SHIM_SOURCE.format(cann_repo="/tmp/unused_cannbench")
    namespace: dict[str, object] = {}
    exec(compile(rendered_shim, "<npu_compare_shim>", "exec"), namespace)
    shim_compare = namespace["compare_case_result"]
    assert callable(shim_compare)
    shim_res = shim_compare(case_id="c1", actual=actual_nan, golden=golden_ok)
    assert shim_res.passed is False
    assert "NaN position mismatch" in shim_res.message
    assert "first_coord=[0, 1]" in shim_res.message


def test_patch_regime_coedits_python_builder_render_tu_and_single_cpp_kernel() -> None:
    """Co-edit Python selector, _render_trackb_tu, and active C++ kernel while keeping siblings."""
    cases = [
        {"case_id": 1, "input_shape": [[1, 128]], "dtype": ["float16"]},
        {"case_id": 2, "input_shape": [[64, 4096]], "dtype": ["float16"]},
    ]
    analyzer = PyTorchSemanticAnalyzer()
    semantics = analyzer.analyze_source(_RMS_GOLDEN, entry_name="rms_norm")
    ratios = tuple(analyzer.compute_case_ratios(semantics, c) for c in cases)
    specs = build_semantic_regime_specs(
        operator_name="level2/rms_norm",
        entrypoint="rms_norm",
        cases=cases,
        proto={"inputs": [{"name": "x"}, {"name": "weight"}], "outputs": [{"name": "y"}]},
        base_axes=semantics.base_axes,
        has_reduction=semantics.has_reduction,
        case_ratios=ratios,
        runtime_backend="ascendc_trackb",
        trackb_symbols=("OP_TRACKB_CU",),
    )
    assert len(specs) == 2
    assert "_render_trackb_tu" in specs[0].editable_symbols
    assert "_render_trackb_tu" in specs[1].editable_symbols

    r0, r1 = specs[0], specs[1]
    base_module = dedent(
        f"""
        OP_TRACKB_CU = r\"\"\"
        #include <tl_templates/ascend/common.h>
        extern "C" __global__ __vector__ void kernel_{r0.regime_tag}(__gm__ half* x) {{
            asc_init();
        }}
        extern "C" __global__ __vector__ void kernel_{r1.regime_tag}(__gm__ half* x) {{
            asc_init();
        }}
        \"\"\"

        def _render_trackb_tu(REGIME: str = "{r0.regime_tag}", TILE_D: int = 128, **extra: object) -> str:
            return f"#define TILE_D {{TILE_D}}\\n" + OP_TRACKB_CU

        def {r0.config_selector_fn}(M: int = 1, D: int = 128, **extra: object) -> dict[str, object]:
            return {{"REGIME": "{r0.regime_tag}", "TILE_D": 128}}

        def {r1.config_selector_fn}(M: int = 64, D: int = 4096, **extra: object) -> dict[str, object]:
            return {{"REGIME": "{r1.regime_tag}", "TILE_D": 256}}

        def _build_rms_norm_{r0.regime_tag}_prim_func(**kw: object) -> object:
            return None

        def _build_rms_norm_{r1.regime_tag}_prim_func(**kw: object) -> object:
            return None
        """
    ).strip()

    coedit_snippet = dedent(
        f"""
        def {r0.config_selector_fn}(M: int = 1, D: int = 128, **extra: object) -> dict[str, object]:
            return {{"REGIME": "{r0.regime_tag}", "TILE_D": 64}}

        def _render_trackb_tu(REGIME: str = "{r0.regime_tag}", TILE_D: int = 64, **extra: object) -> str:
            return f"#define TILE_D {{TILE_D}}\\n#define ACTIVE 1\\n" + OP_TRACKB_CU

        <<<<<<< SEARCH
        extern "C" __global__ __vector__ void kernel_{r0.regime_tag}(__gm__ half* x) {{
            asc_init();
        }}
        =======
        extern "C" __global__ __vector__ void kernel_{r0.regime_tag}(__gm__ half* x) {{
            asc_init();
            asc_copy_gm2ub_align(nullptr, x, 64);
        }}
        >>>>>>> REPLACE
        """
    ).strip()

    patched = patch_regime_into_module(
        base_module,
        r0,
        coedit_snippet,
        operator_entrypoint="rms_norm",
        sibling_specs=(r1,),
        active_kernel=f"kernel_{r0.regime_tag}",
    )
    assert "#define ACTIVE 1" in patched
    assert '{"REGIME":' in patched and '"TILE_D": 64' in patched
    assert "asc_copy_gm2ub_align(nullptr, x, 64);" in patched
    assert f"kernel_{r1.regime_tag}" in patched
    assert '"TILE_D": 256' in patched

    partial_cpp = dedent(
        f"""
        extern "C" __global__ __vector__ void kernel_{r0.regime_tag}(__gm__ half* x) {{
            asc_init();
            asc_copy_gm2ub_align(nullptr, x, 128);
        }}
        """
    ).strip()
    spliced = patch_regime_into_module(
        patched,
        r0,
        partial_cpp,
        operator_entrypoint="rms_norm",
        sibling_specs=(r1,),
        active_kernel=f"kernel_{r0.regime_tag}",
    )
    assert "asc_copy_gm2ub_align(nullptr, x, 128);" in spliced
    assert f"kernel_{r1.regime_tag}" in spliced

    broken_render_tu = dedent(
        f"""
        def _render_trackb_tu(REGIME: str = "{r0.regime_tag}", **extra: object) -> str:
            raise ValueError("broken sibling")
        """
    ).strip()
    with pytest.raises(ValueError, match="sibling"):
        patch_regime_into_module(
            spliced,
            r0,
            broken_render_tu,
            operator_entrypoint="rms_norm",
            sibling_specs=(r1,),
            active_kernel=f"kernel_{r0.regime_tag}",
        )


def test_physical_ledger_to_observation_5d_forwards_roofline_inflation() -> None:
    """Forward Roofline min_gm_bytes through to_observation_5d into transfer_inflation_ratio."""
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=120,
        pipe_cycles={"AIV0.PIPE_V": 80, "AIV0.PIPE_MTE2": 120},
        waterlines={},
        gm_roundtrip_bytes=16384,
        grid_blocks=4,
    )
    obs = ledger.to_observation_5d(
        roofline={"min_gm_bytes": 16384, "compute_floor_us": 2.0},
    )
    assert obs.offchip_dma.transfer_inflation_ratio == pytest.approx(5.0)
    assert obs.transfer_inflation_ratio == pytest.approx(5.0)


def test_cpp_const_define_folding_launcher_cap_and_case_ratios_grid_blocks() -> None:
    """Resolve #define macros and launcher core caps; avoid fabricated grid_blocks=1 or pipes."""
    macro_src = dedent(
        """
        #define RMS_CORES 48
        #define BLK_W 64
        #define DOUBLE_W (BLK_W * 2)
        #define IGNORE_FN(x) ((x) + 1)
        extern "C" __global__ __vector__ void rms_k(__gm__ half* x) {
            asc_init();
            asc_copy_gm2ub_align(nullptr, x, DOUBLE_W);
        }
        void launch_rms(__gm__ half* x, int rows, void* stream) {
            int num_cores = std::min(rows, RMS_CORES);
            rms_k<<<num_cores, nullptr, stream>>>(x);
        }
        """
    )
    consts = _resolve_cpp_consts(macro_src)
    assert consts["RMS_CORES"] == 48
    assert consts["BLK_W"] == 64
    assert consts["DOUBLE_W"] == 128
    assert "IGNORE_FN" not in consts
    assert _parse_cce_launch_block_dim(macro_src, consts) == 48

    unlaunched_vec = dedent(
        """
        #include <tl_templates/ascend/common.h>
        extern "C" __global__ __vector__ void vec_only(__gm__ half* x, __gm__ half* y) {
            asc_init();
            __ubuf__ half* ub = (__ubuf__ half*)0;
            asc_copy_gm2ub_align(ub, x, 128);
            simd_inst::vadd();
            asc_copy_ub2gm_align(y, ub, 128);
        }
        """
    )
    analyzer = PyTorchSemanticAnalyzer()
    sem = analyzer.analyze_source(_RMS_GOLDEN, entry_name="rms_norm")
    ratios = analyzer.compute_case_ratios(
        sem,
        {"case_id": 1, "input_shape": [[28, 128]], "dtype": ["float16"]},
    )
    vec_res = evaluate_ascendc_candidate(
        unlaunched_vec,
        freedom_mask=ConstraintRelaxationMask.top(),
        case_ratios=ratios,
    )
    assert vec_res.passed is True
    assert vec_res.ledger is not None
    expected_blocks = max(1, int(round(float(ratios.wave_occupancy_ratio) * 56)))
    assert vec_res.ledger.grid_blocks == expected_blocks
    assert vec_res.ledger.grid_blocks != 1
    assert "AIC.PIPE_M" not in vec_res.ledger.pipe_cycles
    assert vec_res.ledger.pipe_cycles["AIV0.PIPE_MTE2"] > 0

    scalar_only_src = dedent(
        """
        #include <tl_templates/ascend/common.h>
        extern "C" __global__ __vector__ void scalar_rms(__gm__ float* x, __gm__ float* y) {
            asc_init();
            for (int i = 0; i < 128; ++i) {
                y[i] = x[i];
            }
        }
        """
    )
    scalar_ledger = _build_ascendc_static_ledger(scalar_only_src, case_ratios=ratios)
    assert scalar_ledger.pipe_cycles == {}
    scalar_obs = scalar_ledger.to_observation_5d()
    assert scalar_obs.compute_axis_residuals().compute_issue_gap == pytest.approx(1.0)
    ranked = rank_symptom_families(scalar_obs)
    assert ranked[0][0] == SymptomFamily.SCALAR_LOOP_TRAP

    rules = {r.rule_id: r for r in load_seed_rules()}
    ctx = build_predicate_context(scalar_ledger)
    assert rules["R34_SCALAR_LOOP_TO_VECTOR_DMA_PIPELINE"].matches(ctx) is True
    assert rules["R01_DUAL_COPY_1C2V_SPLIT"].matches(ctx) is False
    assert rules["R13_SPLIT_KV_WAVE_RECOVERY"].matches(ctx) is False


def test_stage0_tracka_first_pivot_and_positive_hardware_engine_gates() -> None:
    """Prefer Track-A first, auto-pivot after static budget, and enforce DMA/Cube gates."""
    analyzer = PyTorchSemanticAnalyzer()
    sem = analyzer.analyze_source(_RMS_GOLDEN, entry_name="rms_norm")
    cases = [
        {"case_id": 1, "input_shape": [[1, 128]], "dtype": ["float16"]},
        {"case_id": 2, "input_shape": [[128, 4096]], "dtype": ["float16"]},
    ]
    ratios = tuple(analyzer.compute_case_ratios(sem, c) for c in cases)
    assert select_initial_track_from_semantics(sem, ratios) == "tilelang_dsl"
    assert select_fallback_track_from_semantics(sem, ratios) in (
        "cce_trackb",
        "hybrid_trackb",
    )

    proto = {"inputs": [{"name": "x"}, {"name": "weight"}], "outputs": [{"name": "y"}]}
    regimes = build_semantic_regime_specs(
        operator_name="level2/rms_norm",
        entrypoint="rms_norm",
        cases=cases,
        proto=proto,
        base_axes=sem.base_axes,
        has_reduction=sem.has_reduction,
        case_ratios=ratios,
        runtime_backend="tilelang_dsl",
    )
    converged_params = dict(_seed_parameter_specs(regimes))
    assert "total_elements" not in converged_params
    assert "**extra_dims: object" in _render_signature(
        "_select_rms_norm_config",
        list(converged_params.items()),
        "dict[str, object]",
    )

    good_trackb_mod = dedent(
        """
        RMS_NORM_TRACKB_CU = r\"\"\"
        #include <tl_templates/ascend/common.h>
        extern "C" __global__ __vector__ void k(__gm__ half* x) {
            asc_init();
            asc_copy_gm2ub_align(nullptr, x, 128);
        }
        \"\"\"
        def _select_rms_norm_config(**extra_dims: object) -> dict[str, object]:
            return {}
        def _build_rms_norm_prim_func(**extra_dims: object) -> object:
            return None
        def _get_kernel(**extra_dims: object) -> object:
            return None
        def _render_trackb_tu(**dims: object) -> str:
            return RMS_NORM_TRACKB_CU
        def rms_norm(x, weight, eps=1e-6):
            return x
        kernel = rms_norm
        """
    ).strip()
    bad_tracka_mod = "def rms_norm(x, weight, eps=1e-6):\n    return x\n"
    backend = _ScriptedStage0Backend([bad_tracka_mod, bad_tracka_mod, good_trackb_mod])
    seen_backends: list[str] = []

    def _validator_factory(
        specs: tuple[RegimeTargetSpec, ...],
        *,
        semantics: object | None = None,
    ) -> Callable[[str, str], tuple[bool, str]]:
        _ = semantics
        rt = specs[0].runtime_backend

        def _check(src: str, tag: str) -> tuple[bool, str]:
            _ = src, tag
            seen_backends.append(rt)
            if rt == "tilelang_dsl":
                return False, "L1 compile error on Track-A"
            return True, ""

        return _check

    def _oracle_ok(src: str) -> OracleAccuracyResult:
        digest = hashlib.sha256(src.encode("utf-8")).hexdigest()
        return OracleAccuracyResult(
            passed=True,
            candidate_sha256=digest,
            l3_run_id="r0",
            l3_receipt_path="/tmp/r0.json",
            l3_state="passed",
            elapsed_sec=0.1,
        )

    outcome = asyncio.run(
        prepare_operator_seed_or_takeover(
            operator_name="level2/rms_norm",
            entrypoint="rms_norm",
            pytorch_golden_source=_RMS_GOLDEN,
            proto=proto,
            cases=cases,
            legacy_regime_specs=None,
            llm_backend=backend,
            l0_l1_validator_factory=_validator_factory,
            oracle_accuracy_validator=_oracle_ok,
            force_cold_start=True,
            max_seed_attempts=4,
        )
    )
    assert outcome.l0_l1_passed is True
    assert outcome.attempts_used == 3
    assert seen_backends[:2] == ["tilelang_dsl", "tilelang_dsl"]
    assert seen_backends[2] in ("ascendc_trackb", "hybrid_trackb")
    assert "CCE C++20" in backend.requests[2].system_prompt

    scalar_src = dedent(
        """
        #include <tl_templates/ascend/common.h>
        extern "C" __global__ __vector__ void s_only(__gm__ float* x, __gm__ float* y) {
            asc_init();
            for (int i = 0; i < 64; ++i) {
                y[i] = x[i];
            }
        }
        """
    )
    dma_gate = evaluate_ascendc_candidate(scalar_src, require_hardware_dma=True)
    assert dma_gate.passed is False
    assert "missing_hardware_dma_staging" in dma_gate.error_message

    cube_gate = evaluate_ascendc_candidate(
        scalar_src,
        require_hardware_dma=False,
        require_cube_engine=True,
    )
    assert cube_gate.passed is False
    assert "missing_cube_engine_primitives" in cube_gate.error_message


def test_agy_cli_503_exponential_backoff_and_fallback_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry agy_503 capacity errors with configured schedule and switch to fallback model."""
    default_watchdog = StreamWatchdogConfig()
    assert default_watchdog.max_retries == 1
    assert default_watchdog.capacity_503_max_retries is None

    jitter_ranges: list[tuple[float, float]] = []
    monkeypatch.setattr(
        llm_client.random,
        "uniform",
        lambda low, high: (jitter_ranges.append((low, high)) or 0.0),
    )

    state_file = tmp_path / "models_seen.json"
    fake_agy = tmp_path / "agy"
    fake_agy.write_text(
        dedent(
            f"""\
            #!{sys.executable}
            import json
            import sys
            from pathlib import Path

            args = sys.argv[1:]
            model = ""
            for idx, arg in enumerate(args):
                if arg == "--model" and idx + 1 < len(args):
                    model = args[idx + 1]
                    break
            state_path = Path({str(state_file)!r})
            seen = json.loads(state_path.read_text()) if state_path.is_file() else []
            seen.append(model)
            state_path.write_text(json.dumps(seen))
            if len(seen) < 3:
                sys.stderr.write("UNAVAILABLE (code 503): No capacity available\\n")
                sys.exit(1)
            sys.stdout.write(json.dumps({{"event": "init", "conversation_id": "c3"}}) + "\\n")
            sys.stdout.write(
                json.dumps(
                    {{
                        "event": "result",
                        "result": {{
                            "status": "SUCCESS",
                            "response": "def ok():\\n    return 1\\n",
                        }},
                    }}
                )
                + "\\n"
            )
            """
        ),
        encoding="utf-8",
    )
    fake_agy.chmod(fake_agy.stat().st_mode | stat.S_IXUSR)

    monkeypatch.setenv("TILEFOUNDRY_NPU_LLM_BACKEND", "agy_cli")
    monkeypatch.setenv("TILEFOUNDRY_NPU_AGY_BIN", str(fake_agy))
    monkeypatch.setenv("TILEFOUNDRY_NPU_LLM_MAX_503_RETRIES", "2")
    monkeypatch.setenv("TILEFOUNDRY_NPU_LLM_503_BACKOFF_SEC", "0.01,0.02")
    monkeypatch.setenv("TILEFOUNDRY_NPU_LLM_FALLBACK_MODEL", "gemini-4-hydrogen-medium")
    monkeypatch.delenv("TILEFOUNDRY_NPU_LLM_EXPLOIT_MODEL", raising=False)

    backend = create_llm_backend_from_env(
        overrides={
            "TILEFOUNDRY_NPU_LLM_EXPLOIT_MODEL": "gemini-4-hydrogen-high",
        }
    )
    reply = asyncio.run(
        backend.generate(GenerationRequest("exploit_rule", "system", "prompt", effort="high"))
    )
    assert "return 1" in reply
    seen_models = json.loads(state_file.read_text(encoding="utf-8"))
    assert seen_models == [
        "gemini-4-hydrogen-high",
        "gemini-4-hydrogen-high",
        "gemini-4-hydrogen-medium",
    ]
    assert jitter_ranges == [(0.0, 3.0), (0.0, 3.0)]


def test_p0_1_scalar_loop_trap_l1_residual_and_runner_wiring() -> None:
    """Detect L1+L2 SCALAR_LOOP_TRAP via StaticDynamicResidual and causal rule evaluation."""
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=120,
        pipe_cycles={"AIV0.PIPE_V": 120, "AIV0.PIPE_MTE2": 64},
        waterlines={},
        gm_roundtrip_bytes=16384,
        grid_blocks=16,
    )
    l1_scalar = BishengResourceSummary(vec_regs=0, scalar_regs=16, software_loops=1)
    roofline = HirRooflineSummary(
        anchor_dims={"M": 128, "D": 128},
        compute_flops=1000000,
        min_gm_bytes=16384,
        arithmetic_intensity=61.0,
        compute_floor_us=2.0,
        memory_floor_us=4.0,
        ideal_roofline_us=4.0,
        dominant_bound="memory",
        cube_flops=0,
        vector_flops=1000000,
    )
    res = compute_static_dynamic_residual(
        ledger,
        {"dur_us": 20.0, "aiv_vec_ratio": 0.005, "aic_mac_ratio": 0.0},
        compiler_summary=l1_scalar,
        roofline=roofline,
    )
    assert res.has_l1_compiler_summary is True
    ctx = build_predicate_context(ledger, residual=res, roofline=roofline)
    assert ctx["has_l1_compiler_summary"] is True
    assert ctx["l1_vec_regs"] == 0
    ranked = rank_symptom_families(ctx)
    assert ranked[0][0] == SymptomFamily.SCALAR_LOOP_TRAP
    rules = {r.rule_id: r for r in load_seed_rules()}
    assert rules["R34_SCALAR_LOOP_TO_VECTOR_DMA_PIPELINE"].matches(ctx) is True

    post_rule = rules["R34_SCALAR_LOOP_TO_VECTOR_DMA_PIPELINE"]
    l1_vec = BishengResourceSummary(vec_regs=32, scalar_regs=8, vloopv2_loops=1)
    cand_res = compute_static_dynamic_residual(
        ledger,
        {"dur_us": 5.0, "aiv_vec_ratio": 0.45, "aic_mac_ratio": 0.0},
        compiler_summary=l1_vec,
        roofline=roofline,
    )
    parent_res = compute_static_dynamic_residual(
        ledger,
        {"dur_us": 20.0, "aiv_vec_ratio": 0.005, "aic_mac_ratio": 0.0},
        compiler_summary=l1_scalar,
        roofline=roofline,
    )
    realized, gain = PerfTargetCampaignRunner._evaluate_causal_rule_outcome(
        rule=post_rule,
        res_all_passed=True,
        rolled_back=False,
        ledger=ledger,
        features=None,
        cand_residual=cand_res,
        cand_lat_us=5.0,
        cand_obj_score=70.0,
        parent_ledger=ledger,
        parent_residual=parent_res,
        parent_lat_us=20.0,
        parent_obj_score=50.0,
        has_cand_msprof=True,
        cand_compiler_summary=l1_vec,
        roofline=roofline,
    )
    assert realized is True
    assert gain == pytest.approx(0.75)


def test_p0_2_trackb_compile_flags_strip_ubuf_on_simt_vf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strip PURE_SIMDVF_UBUF_FLAGS on __simt_vf__ TUs while keeping them on SimdVF TUs."""
    mod_path = tmp_path / "trackb_simt.py"
    mod_path.write_text("def k(x):\n    return x\n", encoding="utf-8")
    spec = RegimeTargetSpec(
        regime_tag="indirect",
        builder_candidates=(),
        sample_builder_kwargs={},
        runtime_backend="ascendc_trackb",
        trackb_symbols=("_TU",),
        editable_symbols=("_TU", "_select_cfg"),
        config_selector_fn="_select_cfg",
    )
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="level2/gather", max_rounds=1),
        target_module_path=mod_path,
        anchors=(),
        regime_specs=(spec,),
        regime_blueprints={
            "indirect": HirBlueprintPacket(
                entry_name="gather",
                roofline=HirRooflineSummary(
                    anchor_dims={"M": 64, "D": 64},
                    compute_flops=1000,
                    min_gm_bytes=1024,
                    arithmetic_intensity=1.0,
                    compute_floor_us=1.0,
                    memory_floor_us=1.0,
                    ideal_roofline_us=1.0,
                    dominant_bound="memory",
                ),
            )
        },
        l2_batch_evaluator=lambda _c, _a: {},
    )
    seen_flags: list[tuple[str, ...]] = []

    def _fake_compile(cpp_source: str, **kwargs: object) -> StageBResult:
        _ = cpp_source
        raw_cf = kwargs.get("compile_flags") or ()
        seen_flags.append(tuple(str(f) for f in raw_cf) if isinstance(raw_cf, Sequence) else ())
        return StageBResult(passed=True)

    monkeypatch.setattr(
        "tilefoundry_ascend.agent.campaign.campaign_runner.run_stage_b_ascendc_compile",
        _fake_compile,
    )
    simt_mod = dedent(
        """
        _TU = r'''
        #include <tl_templates/ascend/common.h>
        __simt_vf__ __aicore__ void gather_vf(__gm__ float* out) {}
        extern "C" __global__ __vector__ void k(__gm__ float* out) { asc_init(); }
        '''
        def _select_cfg(**kw: object) -> dict[str, object]:
            return {"compile_flags": ["--cce-disable-vf-stack-reserved-ubuf", "-O2"]}
        """
    )
    simd_mod = dedent(
        """
        _TU = r'''
        #include <tl_templates/ascend/common.h>
        extern "C" __global__ __vector__ void k(__gm__ float* out) { asc_init(); }
        '''
        def _select_cfg(**kw: object) -> dict[str, object]:
            return {"compile_flags": ["-O2"]}
        """
    )
    assert runner._run_l1(simt_mod, spec).passed is True
    assert runner._run_l1(simd_mod, spec).passed is True
    assert len(seen_flags) == 2
    for flag in PURE_SIMDVF_UBUF_FLAGS:
        assert flag not in seen_flags[0]
        assert flag in seen_flags[1]
    assert "-O2" in seen_flags[0]
    assert "-O2" in seen_flags[1]


def test_p1_1_experience_lake_and_e2e_evolution_persist_roofline(tmp_path: Path) -> None:
    """Persist min_gm_bytes, compute_floor_us, and transfer_inflation_ratio into ExperienceLake."""
    mod_path = tmp_path / "op_mod.py"
    mod_path.write_text("def make_kernel():\n    return None\n", encoding="utf-8")
    lake_dir = tmp_path / "lake"
    spec = RegimeTargetSpec(
        regime_tag="r0",
        builder_candidates=("make_kernel",),
        sample_builder_kwargs={},
    )
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=100,
        pipe_cycles={"AIV0.PIPE_V": 100},
        waterlines={},
        gm_roundtrip_bytes=4096,
        grid_blocks=2,
    )
    roofline = HirRooflineSummary(
        anchor_dims={"M": 64, "D": 64},
        compute_flops=1000000,
        min_gm_bytes=8192,
        arithmetic_intensity=122.0,
        compute_floor_us=1.5,
        memory_floor_us=3.0,
        ideal_roofline_us=3.0,
        dominant_bound="memory",
        cube_flops=0,
        vector_flops=1000000,
    )
    bp = HirBlueprintPacket(
        entry_name="op",
        roofline=roofline,
    )
    anchor = AnchorCaseSpec("level2/rms_norm", 1, "r0", 20.0, 3.0)
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="level2/rms_norm", max_rounds=1),
        target_module_path=mod_path,
        anchors=(anchor,),
        regime_specs=(spec,),
        regime_blueprints={"r0": bp},
        experience_lake_dir=lake_dir,
        l2_batch_evaluator=lambda _c, _a: {},
    )
    surrogate_res = CandidateSurrogateResult(
        candidate_id="c1",
        operator_name="level2/rms_norm",
        all_passed=True,
        avg_speedup=2.0,
        min_case_speedup=2.0,
        avg_roofline_eta=0.3,
        surrogate_score=60.0,
        anchors=(
            AnchorEvalMetrics(
                case_num=1,
                regime_tag="r0",
                accuracy_passed=True,
                cand_us=10.0,
                ref_perf_us=20.0,
                t_hw_us=3.0,
                speedup=2.0,
                roofline_eta=0.3,
                eq3_score=50.0,
            ),
        ),
    )
    runner._record_surrogate_observations(
        source="def make_kernel():\n    return 1\n",
        result=surrogate_res,
        active_anchors=(anchor,),
        ledger=ledger,
        regime_tag="r0",
    )
    lake = ExperienceLake(lake_dir)
    _, observations = lake.load_records_by_regime()
    assert len(observations) == 1
    assert observations[0].min_gm_bytes == pytest.approx(8192.0)
    assert observations[0].compute_floor_us == pytest.approx(1.5)
    assert observations[0].transfer_inflation_ratio == pytest.approx(2.0)

    dsl_src = dedent(
        """
        import tilelang.ascend.language as T

        def make_kernel():
            @T.prim_func
            def main(
                X: T.Tensor((128, 128), "bfloat16"),
                W: T.Tensor((128, 128), "bfloat16"),
                Y: T.Tensor((128, 128), "float32"),
            ):
                with T.Kernel(1):
                    x_l1 = T.alloc_l1((128, 64), "bfloat16")
                    w_l1 = T.alloc_l1((128, 64), "bfloat16")
                    acc = T.alloc_l0c((128, 128), "float32")
                    T.copy(X[0:128, 0:64], x_l1)
                    T.copy(W[0:128, 0:64], w_l1)
                    T.gemm(x_l1, w_l1, acc, transpose_B=True, clear_accum=True)
                    T.copy(acc, Y[0:128, 0:128])
            return main
        """
    )
    ar_golden = dedent(
        """
        import torch

        def ar_roofline(h, x, w):
            return torch.matmul(x + h, w)
        """
    )
    funnel = GateFunnel(
        compile_runner=lambda _dsl: StageBResult(passed=True),
        device_runner=lambda _dsl, sa: StageCResult(
            passed=True,
            accuracy_passed=True,
            measured_latency_us=25.0,
            residual=compute_static_dynamic_residual(sa.ledger, {"dur_us": 25.0}),
            telemetry={"dur_us": 25.0},
        ),
    )
    traj = asyncio.run(
        run_e2e_evolution(
            problem_name="ar_roofline",
            hir_source=ar_golden,
            shape_contract="make_kernel()",
            backend=MockLLMBackend(responses={}, default_response=dsl_src),
            funnel=funnel,
            initial_dsl_source=dsl_src,
            anchor_dims={"M": 128, "N": 128, "K": 128},
            num_rounds=1,
            k_rule_workers=1,
        )
    )
    assert traj.hir_blueprint_summary["roofline"]["min_gm_bytes"] > 0


def test_p1_2_hybrid_and_tracka_cube_dma_aggregate_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enforce Cube/DMA invariants uniformly across Track-A and aggregate hybrid_trackb."""
    mod_path = tmp_path / "hyb_matmul.py"
    mod_path.write_text("def make_kernel():\n    return None\n", encoding="utf-8")
    hyb_spec = RegimeTargetSpec(
        regime_tag="r0",
        builder_candidates=("_build_aux",),
        sample_builder_kwargs={},
        runtime_backend="hybrid_trackb",
        trackb_symbols=("_TU",),
        editable_symbols=("_build_aux", "_TU"),
    )
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="level4/gqa", max_rounds=1),
        target_module_path=mod_path,
        anchors=(),
        regime_specs=(hyb_spec,),
        regime_blueprints={
            "r0": HirBlueprintPacket(
                entry_name="gqa",
                roofline=HirRooflineSummary(
                    anchor_dims={"M": 64, "D": 64},
                    compute_flops=1000,
                    min_gm_bytes=1024,
                    arithmetic_intensity=1.0,
                    compute_floor_us=1.0,
                    memory_floor_us=1.0,
                    ideal_roofline_us=1.0,
                    dominant_bound="memory",
                ),
            )
        },
        l2_batch_evaluator=lambda _c, _a: {},
    )
    matmul_sem = PyTorchSemanticAnalyzer().analyze_source(
        "import torch\ndef gqa(q, k):\n    return q @ k.transpose(-1, -2)\n",
        entry_name="gqa",
    )
    monkeypatch.setattr(runner, "_resolve_operator_semantics_for_gates", lambda: matmul_sem)
    monkeypatch.setattr(runner, "_resolve_combine_gate_spec", lambda *_a: None)
    monkeypatch.setattr(
        runner,
        "_resolve_trackb_freedom_mask",
        lambda _spec: ConstraintRelaxationMask.top(),
    )

    vec_ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=20,
        pipe_cycles={"AIV0.PIPE_V": 20, "AIV0.PIPE_MTE2": 16},
        waterlines={},
    )
    seen_stage_a_reqs: list[tuple[bool, bool]] = []

    def _fake_stage_a(*_args: object, **kwargs: object) -> StageAResult:
        seen_stage_a_reqs.append(
            (
                bool(kwargs.get("require_hardware_dma", False)),
                bool(kwargs.get("require_cube_engine", False)),
            )
        )
        return StageAResult(passed=True, prim_func=None, ledger=vec_ledger)

    monkeypatch.setattr(
        "tilefoundry_ascend.agent.campaign.campaign_runner.run_stage_a_static",
        _fake_stage_a,
    )

    no_cube_hyb = dedent(
        """
        _TU = r'''
        #include <tl_templates/ascend/common.h>
        extern "C" __global__ __vector__ void vec_k(__gm__ half* x, __gm__ half* y) {
            asc_init();
            __ubuf__ half* u = (__ubuf__ half*)0;
            asc_copy_gm2ub_align(u, x, 128);
            asc_copy_ub2gm_align(y, u, 128);
        }
        '''
        def _build_aux():
            def inner():
                pass
            inner.__prim_func__ = True
            return inner
        """
    )
    cube_hyb = no_cube_hyb.replace(
        "asc_copy_ub2gm_align(y, u, 128);",
        "asc_mmad(nullptr, nullptr, nullptr, 16, 16, 16, true);\n    asc_copy_ub2gm_align(y, u, 128);",
    )
    no_cube_res = runner._default_l0_checker(no_cube_hyb, hyb_spec)
    assert no_cube_res.passed is False
    assert "missing_cube_engine_primitives" in no_cube_res.error_message

    cube_res = runner._default_l0_checker(cube_hyb, hyb_spec)
    assert cube_res.passed is True

    tracka_spec = RegimeTargetSpec(
        regime_tag="r0",
        builder_candidates=("_build_aux",),
        sample_builder_kwargs={},
        runtime_backend="tilelang_dsl",
    )
    runner._default_l0_checker(no_cube_hyb, tracka_spec)
    assert seen_stage_a_reqs[-1] == (True, True)


def test_p1_4_and_p1_5_renderer_required_and_partial_tu_exclusive_helper() -> None:
    """Require non-empty _render_trackb_tu output and allow spliced exclusive helpers."""
    spec = RegimeTargetSpec(
        regime_tag="r0",
        builder_candidates=("_build_r0",),
        sample_builder_kwargs={},
        runtime_backend="ascendc_trackb",
        trackb_symbols=("_TU",),
        editable_symbols=("_select_r0", "_render_trackb_tu", "_TU"),
        config_selector_fn="_select_r0",
    )
    sibling = RegimeTargetSpec(
        regime_tag="r1",
        builder_candidates=("_build_r1",),
        sample_builder_kwargs={},
        runtime_backend="ascendc_trackb",
        trackb_symbols=("_TU",),
        editable_symbols=("_select_r1", "_render_trackb_tu", "_TU"),
        config_selector_fn="_select_r1",
    )
    base_mod = dedent(
        """
        _TU = r'''
        #include <tl_templates/ascend/common.h>
        extern "C" __global__ __vector__ void kernel_r0(__gm__ float* x) {
            asc_init();
            x[0] = 1.0f;
        }
        extern "C" __global__ __vector__ void kernel_r1(__gm__ float* x) {
            asc_init();
            x[0] = 2.0f;
        }
        '''
        def _select_r0(**kw: object) -> dict[str, object]:
            return {"TILE": 64}
        def _select_r1(**kw: object) -> dict[str, object]:
            return {"TILE": 128}
        def _render_trackb_tu(**dims: object) -> str:
            return _TU
        def _build_r0(**kw: object) -> object:
            return None
        def _build_r1(**kw: object) -> object:
            return None
        """
    ).strip()

    empty_render = dedent(
        """
        def _render_trackb_tu(**dims: object) -> str:
            return ""
        """
    ).strip()
    with pytest.raises(ValueError, match="empty C\\+\\+ translation unit"):
        patch_regime_into_module(
            base_mod,
            spec,
            empty_render,
            sibling_specs=(sibling,),
            active_kernel="kernel_r0",
        )

    partial_with_helper = dedent(
        """
        _TU = r'''
        __attribute__((always_inline)) inline void scale_helper(__gm__ float* x) {
            x[0] = 9.0f;
        }
        extern "C" __global__ __vector__ void kernel_r0(__gm__ float* x) {
            asc_init();
            scale_helper(x);
        }
        '''
        """
    ).strip()
    patched = patch_regime_into_module(
        base_mod,
        spec,
        partial_with_helper,
        sibling_specs=(sibling,),
        active_kernel="kernel_r0",
    )
    assert "void scale_helper(" in patched
    assert "kernel_r1" in patched

    dead_helper = dedent(
        """
        _TU = r'''
        __attribute__((always_inline)) inline void unused_helper(__gm__ float* x) {
            x[0] = 9.0f;
        }
        extern "C" __global__ __vector__ void kernel_r0(__gm__ float* x) {
            asc_init();
            x[0] = 5.0f;
        }
        '''
        """
    ).strip()
    with pytest.raises(ValueError, match="outside the editable scope"):
        patch_regime_into_module(
            base_mod,
            spec,
            dead_helper,
            sibling_specs=(sibling,),
            active_kernel="kernel_r0",
        )


def test_p1_6_cost_prescreen_l1_backfill_rescues_top_k_failures() -> None:
    """Backfill L1 Stage-B failures sequentially from sorted_passers[top_k:]."""
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=20,
        pipe_cycles={"AIV0.PIPE_V": 20},
        waterlines={},
    )
    cands = [
        ("c1_bad_l1", "s1"),
        ("c2_ok", "s2"),
        ("c3_backfill_ok", "s3"),
        ("c4_never_compiled", "s4"),
    ]
    compiled_ids: list[str] = []

    def _l1(src: str) -> StageBResult:
        compiled_ids.append(src)
        if src == "s1":
            return StageBResult(passed=False, error_message="bisheng: spill")
        return StageBResult(passed=True)

    checks = asyncio.run(
        run_l0_l1_round_filter(
            cands,
            patch_fn=lambda snip: snip,
            l0_checker=lambda _src: StageAResult(passed=True, prim_func=None, ledger=ledger),
            l1_compiler=_l1,
            stage_a_cost_ranker=lambda items: [
                ("c1_bad_l1", 1.0),
                ("c2_ok", 2.0),
                ("c3_backfill_ok", 3.0),
                ("c4_never_compiled", 4.0),
            ],
            cost_prescreen_top_k=2,
        )
    )
    by_id = {c.candidate_id: c for c in checks}
    assert by_id["c1_bad_l1"].passed is False
    assert by_id["c1_bad_l1"].failed_tier == "stage_b"
    assert by_id["c2_ok"].passed is True
    assert by_id["c3_backfill_ok"].passed is True
    assert by_id["c4_never_compiled"].passed is False
    assert by_id["c4_never_compiled"].failed_tier == "cost_prescreen"
    assert compiled_ids == ["s1", "s2", "s3"]


def test_p1_r1_kernel_cache_bloat_classified_as_candidate_failure(tmp_path: Path) -> None:
    """Classify _KERNEL_CACHE bloat as a candidate comparison failure, never infrastructure."""
    contract = l3_contract.TargetContract(
        operator="rms_norm",
        task="level2/rms_norm",
        task_dir=tmp_path,
        nodes=(
            l3_contract.NodeContract(
                name=l3_contract.GENERIC_NODE,
                required_cases=("0", "1"),
            ),
        ),
        case_ids=("0", "1"),
    )
    evidence_path = tmp_path / "l3_evidence.json"
    recorder = l3_contract.EvidenceRecorder(evidence_path, contract)
    detail = "volatile _KERNEL_CACHE keys bloated cache to 12 entries across 20 cases (limit=10)"
    with pytest.raises(l3_contract.ClassifiedTestFailure):
        with recorder.node(l3_contract.GENERIC_NODE):
            recorder.case_started(l3_contract.GENERIC_NODE, "0")
            recorder.case_finished(l3_contract.GENERIC_NODE, "0", "passed")
            recorder.case_started(l3_contract.GENERIC_NODE, "1")
            recorder.case_finished(l3_contract.GENERIC_NODE, "1", "passed")
            failure = l3_contract.comparison_failure(
                node=l3_contract.GENERIC_NODE,
                case_id="1",
                role="candidate",
                comparator="kernel_cache_contract",
                references=("regime_budget",),
                comparisons=({"output": 0, "passed": False, "message": detail},),
                detail=detail,
            )
            recorder.case_finished(
                l3_contract.GENERIC_NODE,
                "1",
                "failed",
                detail,
                failure=failure,
            )
            raise l3_contract.ClassifiedTestFailure(detail)
    recorder.finalize(interrupted=False)
    payload = recorder.payload()
    failures = payload["failures"]
    assert isinstance(failures, list) and len(failures) == 1
    assert failures[0]["kind"] == l3_contract.FAILURE_KIND_CANDIDATE
    assert failures[0]["category"] == "numeric"
    assert failures[0]["comparator"] == "kernel_cache_contract"
    coverage = payload["coverage"]
    assert isinstance(coverage, dict)
    assert (
        surrogate_cannbench._l3_failure_kind(
            observation={"failureCategory": "numeric"},
            evidence=payload,
            coverage=coverage,
        )
        == surrogate_cannbench.FAILURE_KIND_CANDIDATE
    )


def test_v441_surfaces_cover_simd_gemm_and_sync_contracts() -> None:
    """Verify TileLang DSL and CCE C++20 surfaces include SIMD, GEMM NT, and event_t APIs."""
    tl_rendered = TILELANG_DSL_SURFACE.render_prompt_reference()
    for token in (
        "T.simd.vdupv",
        "T.simd.alloc_var",
        "T.simd.alloc_local",
        "T.simd.vmula",
        "T.simd.vsel",
        "T.simd.vcmps",
        "T.simd.vci",
        "ONEPT_B32",
        "BRC_B32",
        "transpose_B=True",
        "T.copy(src, dst, transpose=True)",
    ):
        assert token in tl_rendered

    cce_rendered = CCE_CPP20_SURFACE.render_prompt_reference()
    for token in (
        "EVENT_ID0",
        "ascend_gemm_l1",
        "asc_unit_flag_mode::DISABLE",
        "simd_inst::vdupv",
    ):
        assert token in cce_rendered
    positive_lines = [
        line
        for line in cce_rendered.splitlines()
        if "AscendC" in line and "Strictly Forbidden" not in line and "Do NOT emit" not in line
    ]
    assert positive_lines == []


def test_v441_bounded_compile_diagnostic_strips_deprecation_and_bt_noise(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Filter deprecation cascades and TVM backtrace frames, and silence bisheng warnings."""
    deprecated_block = (
        "/tmp/k.asc:22:5: warning: 'asc_mmad' is deprecated [-Wdeprecated-declarations]\n"
        "    asc_mmad(c, a, b, 16, 16, 16, 0, true, false, true);\n"
        "    ^\n"
        "/inc/cube_compute.h:629:1: note: 'asc_mmad' has been explicitly marked deprecated here\n"
        "ASC_DEPRECATED(9.2.0)\n"
        "^\n"
        "/inc/macro.h:15:7: note: expanded from macro 'ASC_DEPRECATED'\n"
        "    [[deprecated]]\n"
        "      ^\n"
    )
    real_err = "/tmp/k.asc:88:9: error: no matching function for call to 'asc_sync_notify'"
    noisy = (deprecated_block * 12) + f"{real_err}\n    asc_sync_notify(PIPE_MTE2, PIPE_V, 1);\n"
    diag = _bounded_compile_diagnostic(noisy)
    assert real_err in diag
    assert "marked deprecated" not in diag

    tvm_noisy = (
        "InternalError: Check failed: (pb->value != 0) is false: Divide by zero\n"
        "Stack trace:\n"
        + "".join(f"  [bt] ({i}) /lib/libtvm.so+0x{i:x} frame_{'x' * 60}\n" for i in range(40))
    )
    tvm_diag = _bounded_compile_diagnostic(tvm_noisy)
    assert "Check failed: (pb->value != 0)" in tvm_diag
    assert "[bt] (0)" not in tvm_diag

    seen_cmds: list[list[str]] = []

    def _fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        seen_cmds.append(list(cmd))
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="err")

    monkeypatch.setattr("tilefoundry_ascend.agent.orchestrator.gate_funnel.subprocess.run", _fake_run)
    fake_bisheng = tmp_path / "bisheng"
    fake_bisheng.write_text("", encoding="utf-8")
    res = run_stage_b_ascendc_compile(
        '#include <tl_templates/ascend/common.h>\n__global__ void k() {}',
        bisheng_bin=fake_bisheng,
        cann_root=tmp_path,
    )
    assert res.passed is False
    assert seen_cmds and "-Wno-deprecated-declarations" in seen_cmds[0]
