"""Unit tests for scoped C++20 slicing, patching, and Track-A-to-Track-B escalation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tilefoundry_ascend.agent.specs.anchor_specs import (
    CannbenchOpsRegistry,
    RegimeTargetSpec,
)
from tilefoundry_ascend.agent.campaign.campaign_runner import (
    CampaignConfig,
    PerfTargetCampaignRunner,
    _extract_trackb_cpp_source,
    _is_real_tir_primfunc,
    extract_regime_snippet,
    patch_regime_into_module,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    AnchorEvalMetrics,
    CandidateSurrogateResult,
    run_l0_l1_round_filter,
)
from tilefoundry_ascend.agent.kb.schema import AscendCausalRule, load_seed_rules
from tilefoundry_ascend.agent.orchestrator.cpp_slice_protocol import (
    _split_cpp_units,
    apply_scoped_trackb_patch,
    build_trackb_context_view,
    escalate_primfunc_to_trackb_tu,
    extract_cpp_diagnostic_lines,
    extract_reachable_trackb_source,
    parse_scoped_search_replace_blocks,
    resolve_instantiated_trackb_kernel,
    select_active_trackb_kernel,
    splice_cpp_units_into_tu,
)
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    StageAResult,
    StageBResult,
    _resolve_agent_gates_bin,
    _resolve_prim_func_from_dsl,
    build_stage_b_driver_source,
    evaluate_ascendc_candidate,
)
from tilefoundry_ascend.ascend.sil.schema import CasePhysicsRatios, ConstraintRelaxationMask
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    BeamCandidate,
    compute_static_dynamic_residual,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger
from tilefoundry_ascend.ascend.templates.registry import _format_host_launcher_stub

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_select_active_trackb_kernel_prefers_tag_then_launcher_call() -> None:
    """Resolve the active __global__ kernel by explicit name, regime tag, or launcher call."""
    cpp = (
        "#include <tl_templates/ascend/common.h>\n"
        'extern "C" __global__ __vector__ void helper_init_kernel(__gm__ float* x) {\n'
        "    x[0] = 0.0f;\n"
        "}\n"
        'extern "C" __global__ __vector__ void mla_kv_reuse_c1c1c2(__gm__ float* q) {\n'
        "    q[0] = 1.0f;\n"
        "}\n"
        'extern "C" __global__ __vector__ void mla_fast_combine(__gm__ float* out) {\n'
        "    out[0] = 2.0f;\n"
        "}\n"
    )
    module_src = (
        f'_SRC = r"""\n{cpp}\n"""\n'
        'LAUNCHER = r"""\n'
        'extern "C" int launch(void* stream, void* q) {\n'
        "    mla_kv_reuse_c1c1c2<32><<<28, nullptr, stream>>>(q);\n"
        "    return 0;\n"
        "}\n"
        '"""\n'
    )
    assert select_active_trackb_kernel(cpp, active_kernel="mla_fast_combine") == "mla_fast_combine"
    assert select_active_trackb_kernel(cpp, active_kernel="nonexistent") is None
    assert (
        select_active_trackb_kernel(cpp, regime_tag="split_kv_combine", module_source=module_src)
        == "mla_fast_combine"
    )
    assert (
        select_active_trackb_kernel(cpp, regime_tag="decode_main", module_source=module_src)
        == "mla_kv_reuse_c1c1c2"
    )


@pytest.mark.parametrize("op_name", ["softmax", "mla", "grouped_matmul"])
def test_build_trackb_context_view_compacts_real_ops_under_15kb(op_name: str) -> None:
    """Compact >15KB real Track-B C++ sources under 15KB while keeping globals and launcher."""
    mod_path = REPO_ROOT / "examples" / "ascend950pr_ops" / op_name / "kernels.py"
    mod_src = mod_path.read_text(encoding="utf-8")
    specs = CannbenchOpsRegistry.default().get_regime_specs(op_name)
    spec = next(s for s in specs if s.trackb_symbols)
    extracted = _extract_trackb_cpp_source(mod_src, spec)
    assert extracted is not None
    sym_name, cpp_src = extracted
    assert len(cpp_src.encode("utf-8")) > 15360

    active = select_active_trackb_kernel(
        cpp_src,
        regime_tag=spec.regime_tag,
        module_source=mod_src,
    )
    assert active is not None

    view = build_trackb_context_view(
        cpp_src,
        active_kernel=active,
        regime_tag=spec.regime_tag,
        module_source=mod_src,
        trackb_symbol=sym_name,
        max_bytes=14800,
    )
    assert "MIDDLE BODY OMITTED" not in view
    assert "LINES OMITTED" not in view
    assert active in view
    assert "LAUNCHER BINDING PREVIEW" in view
    assert 'extern "C"' in view


def test_apply_scoped_trackb_patch_exact_multiblock_and_error_guards() -> None:
    """Verify exact single/multi-block patching and deterministic errors on invalid hunks."""
    cpp = (
        "#include <tl_templates/ascend/common.h>\n"
        "struct Params { int n; };\n"
        'extern "C" __global__ __vector__ void kernel_active(__gm__ float* x) {\n'
        "    int step = 1;\n"
        "    x[0] = x[0] + (float)step;\n"
        "    int tail = 2;\n"
        "    x[1] = x[1] + (float)tail;\n"
        "}\n"
        'extern "C" __global__ __vector__ void kernel_sibling(__gm__ float* y) {\n'
        "    int keep = 7;\n"
        "    y[0] = (float)keep;\n"
        "}\n"
    )
    multi_patch = (
        "<<<<<<< SEARCH\n"
        "    int step = 1;\n"
        "=======\n"
        "    int step = 4;\n"
        ">>>>>>> REPLACE\n\n"
        "<<<<<<< SEARCH\n"
        "    int tail = 2;\n"
        "=======\n"
        "    int tail = 8;\n"
        ">>>>>>> REPLACE\n"
    )
    updated, count = apply_scoped_trackb_patch(cpp, multi_patch, active_kernel="kernel_active")
    assert count == 2
    assert "int step = 4;" in updated
    assert "int tail = 8;" in updated
    assert "int keep = 7;" in updated

    missing_patch = (
        "<<<<<<< SEARCH\n"
        "    int missing_var = 99;\n"
        "=======\n"
        "    int missing_var = 100;\n"
        ">>>>>>> REPLACE\n"
    )
    with pytest.raises(ValueError, match="did not match"):
        apply_scoped_trackb_patch(cpp, missing_patch, active_kernel="kernel_active")

    dup_cpp = cpp + "\n// dup line: x[0] = x[0] + (float)step;\n"
    ambiguous_patch = (
        "<<<<<<< SEARCH\n"
        "x[0] = x[0] + (float)step;\n"
        "=======\n"
        "x[0] = x[0] + 2.0f;\n"
        ">>>>>>> REPLACE\n"
    )
    with pytest.raises(ValueError, match="Ambiguous SEARCH block"):
        apply_scoped_trackb_patch(dup_cpp, ambiguous_patch, active_kernel="kernel_active")

    sibling_patch = (
        "<<<<<<< SEARCH\n"
        "    int keep = 7;\n"
        "=======\n"
        "    int keep = 9;\n"
        ">>>>>>> REPLACE\n"
    )
    with pytest.raises(ValueError, match="read-only code outside active kernel"):
        apply_scoped_trackb_patch(cpp, sibling_patch, active_kernel="kernel_active")


_SCOPED_HELPER_CPP = (
    "#include <tl_templates/ascend/common.h>\n"
    "template <typename InT>\n"
    "__simd_vf__ inline void vf_helper_exclusive(__ubuf__ InT* x) {\n"
    "  int step = 1;\n"
    "  x[0] = step;\n"
    "}\n"
    "template <typename InT>\n"
    "__simd_vf__ inline void vf_helper_twin(__ubuf__ InT* x) {\n"
    "    int step = 1;\n"
    "    x[0] = step;\n"
    "}\n"
    "template <typename InT>\n"
    "__simd_vf__ inline void vf_helper_shared(__ubuf__ InT* x) {\n"
    "  int keep = 2;\n"
    "  x[0] = keep;\n"
    "}\n"
    'extern "C" __global__ __vector__ void kernel_active(__gm__ float* x) {\n'
    "  vf_helper_exclusive(x);\n"
    "  vf_helper_shared(x);\n"
    "}\n"
    'extern "C" __global__ __vector__ void kernel_sibling(__gm__ float* y) {\n'
    "  vf_helper_shared(y);\n"
    "}\n"
)


def _hunk(search: str, replace: str) -> str:
    """Render one SEARCH/REPLACE hunk in the scoped Track-B patch format."""
    return f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


def test_apply_scoped_trackb_patch_scopes_exclusive_helpers_and_normalized_matches() -> None:
    """Allow exclusive helpers and normalized hunks while shared helpers stay read-only."""
    exclusive_hunk = _hunk(
        "  int step = 1;\n  x[0] = step;", "  int step = 3;\n  x[0] = step;"
    )
    updated, applied = apply_scoped_trackb_patch(
        _SCOPED_HELPER_CPP, exclusive_hunk, active_kernel="kernel_active"
    )
    assert applied == 1
    assert "  int step = 3;" in updated

    for search, replace in (
        ("  int keep = 2;", "  int keep = 4;"),
        ("  vf_helper_shared(y);", "  vf_helper_shared(x);"),
    ):
        with pytest.raises(ValueError, match="read-only code outside active kernel"):
            apply_scoped_trackb_patch(
                _SCOPED_HELPER_CPP, _hunk(search, replace), active_kernel="kernel_active"
            )

    with pytest.raises(ValueError, match="read-only code outside active kernel"):
        apply_scoped_trackb_patch(
            _SCOPED_HELPER_CPP,
            exclusive_hunk,
            active_kernel="kernel_active",
            allow_exclusive_helpers=False,
        )

    normalized, normalized_count = apply_scoped_trackb_patch(
        _SCOPED_HELPER_CPP,
        _hunk(
            "vf_helper_exclusive(x);\nvf_helper_shared(x);",
            "vf_helper_exclusive(x);\nvf_helper_shared(x);\nx[0] = x[0] + 1.0f;",
        ),
        active_kernel="kernel_active",
    )
    assert normalized_count == 1
    assert "  vf_helper_shared(x);\n  x[0] = x[0] + 1.0f;\n}" in normalized

    with pytest.raises(ValueError, match="Ambiguous SEARCH block matched 2 locations"):
        apply_scoped_trackb_patch(
            _SCOPED_HELPER_CPP,
            _hunk("int step = 1;\nx[0] = step;", "int step = 9;\nx[0] = step * 2;"),
            active_kernel="kernel_active",
        )

    with pytest.raises(ValueError, match="read-only code outside active kernel"):
        apply_scoped_trackb_patch(
            _SCOPED_HELPER_CPP,
            _hunk("int keep = 2;\nx[0] = keep;", "int keep = 6;\nx[0] = keep;"),
            active_kernel="kernel_active",
        )


def test_splice_cpp_units_into_tu_replaces_only_scoped_bare_units() -> None:
    """Replace same-name kernel and helper definitions while sibling kernels stay untouched."""
    candidate_kernel = (
        'extern "C" __global__ __vector__ void kernel_active(__gm__ float* x) {\n'
        "  vf_helper_exclusive(x);\n"
        "  vf_helper_shared(x);\n"
        "  x[0] = x[0] + 1.0f;\n"
        "}\n"
    )
    updated, count = splice_cpp_units_into_tu(
        _SCOPED_HELPER_CPP, candidate_kernel, active_kernel="kernel_active"
    )
    assert count == 1
    assert "x[0] = x[0] + 1.0f;" in updated
    assert "vf_helper_shared(y);" in updated

    candidate_helper = (
        "__simd_vf__ inline void vf_helper_exclusive(__ubuf__ InT* x) {\n"
        "  x[0] = 42;\n"
        "}\n"
    )
    updated_helper, helper_count = splice_cpp_units_into_tu(
        _SCOPED_HELPER_CPP, candidate_helper, active_kernel="kernel_active"
    )
    assert helper_count == 1
    assert "x[0] = 42;" in updated_helper
    assert "int keep = 2;" in updated_helper

    sibling = (
        'extern "C" __global__ __vector__ void kernel_sibling(__gm__ float* y) {\n'
        "  y[0] = 1.0f;\n"
        "}\n"
    )
    with pytest.raises(ValueError, match="outside the editable scope"):
        splice_cpp_units_into_tu(_SCOPED_HELPER_CPP, sibling, active_kernel="kernel_active")
    with pytest.raises(ValueError, match="defined more than once"):
        splice_cpp_units_into_tu(
            _SCOPED_HELPER_CPP, candidate_helper + candidate_helper, active_kernel="kernel_active"
        )
    with pytest.raises(ValueError, match="not found in C\\+\\+ source"):
        splice_cpp_units_into_tu(
            _SCOPED_HELPER_CPP,
            "__simd_vf__ inline void vf_helper_absent(__ubuf__ InT* x) {\n  x[0] = 0;\n}\n",
        )
    with pytest.raises(ValueError, match="does not contain any bare C\\+\\+ kernel or helper"):
        splice_cpp_units_into_tu(_SCOPED_HELPER_CPP, "int x = 1;\n", active_kernel="kernel_active")
    with pytest.raises(ValueError, match="Active kernel 'missing_kernel' not found"):
        splice_cpp_units_into_tu(_SCOPED_HELPER_CPP, candidate_helper, active_kernel="missing_kernel")


def test_resolve_instantiated_trackb_kernel_reads_launcher_truth_per_regime() -> None:
    """Resolve each Softmax regime's active kernel from the instantiated launcher tail."""
    mod_path = REPO_ROOT / "examples" / "ascend950pr_ops" / "softmax" / "kernels.py"
    mod_src = mod_path.read_text(encoding="utf-8")
    mod_spec = importlib.util.spec_from_file_location("softmax_trackb_protocol", mod_path)
    assert mod_spec is not None and mod_spec.loader is not None
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    render = mod._render_trackb_tu
    expected = {
        "reduce_tail": "softmax_last_axis_kernel",
        "reduce_tail_subatom": "softmax_last_axis_kernel",
        "reduce_strided_twopass": "softmax_inner_2pass_kernel",
        "reduce_strided": "softmax_inner_1pass_kernel",
    }
    for regime in CannbenchOpsRegistry.default().get_regime_specs("softmax"):
        extracted = _extract_trackb_cpp_source(mod_src, regime)
        assert extracted is not None
        raw_cpp = extracted[1]
        dims = dict(regime.representative_hir_dims or {})
        dims["trackb_src"] = raw_cpp
        instantiated = render(**dims)
        assert resolve_instantiated_trackb_kernel(instantiated, raw_cpp) == expected[regime.regime_tag]

    raw_cpp = (
        'extern "C" __global__ __vector__ void present_kernel(__gm__ float* x) {\n'
        "  x[0] = 1.0f;\n"
        "}\n"
    )
    absent_launcher = (
        'extern "C" int launch() {\n'
        "  absent_kernel<<<56, nullptr, stream>>>();\n"
        "  return 0;\n"
        "}\n"
    )
    assert resolve_instantiated_trackb_kernel(absent_launcher, raw_cpp) is None
    assert resolve_instantiated_trackb_kernel("", raw_cpp) is None


def test_resolve_instantiated_trackb_kernel_requires_one_distinct_launched_kernel() -> None:
    """Mask comments and literals, accept repeated launches, and reject ambiguity."""
    raw_cpp = (
        "// softmax_last_axis_kernel<<<1, nullptr, stream>>>();\n"
        'extern "C" __global__ __vector__ void softmax_last_axis_kernel(__gm__ float* x) {\n'
        "  x[0] = 1.0f;\n"
        "}\n"
        'extern "C" __global__ __vector__ void softmax_inner_kernel(__gm__ float* x) {\n'
        "  x[1] = 1.0f;\n"
        "}\n"
    )
    repeated = (
        raw_cpp
        + 'extern "C" int launch(void* stream) {\n'
        "  // softmax_inner_kernel<<<4, nullptr, stream>>>();\n"
        '  const char* doc = "softmax_inner_kernel<<<4, nullptr, stream>>>();";\n'
        "  softmax_last_axis_kernel<<<56, nullptr, stream>>>();\n"
        "  softmax_last_axis_kernel<<<28, nullptr, stream>>>();\n"
        "  return 0;\n"
        "}\n"
    )
    assert resolve_instantiated_trackb_kernel(repeated, raw_cpp) == "softmax_last_axis_kernel"

    ambiguous = repeated.replace(
        "  softmax_last_axis_kernel<<<28, nullptr, stream>>>();\n",
        "  softmax_inner_kernel<<<28, nullptr, stream>>>();\n",
    )
    assert resolve_instantiated_trackb_kernel(ambiguous, raw_cpp) is None

    only_commented = (
        raw_cpp
        + 'extern "C" int launch(void* stream) {\n'
        "  // softmax_last_axis_kernel<<<56, nullptr, stream>>>();\n"
        "  return 0;\n"
        "}\n"
    )
    assert resolve_instantiated_trackb_kernel(only_commented, raw_cpp) is None


def test_patch_regime_into_module_search_replace_and_python_fallback() -> None:
    """Splice scoped SEARCH/REPLACE hunks into Track-B constants and keep Python fallback."""
    mod_path = REPO_ROOT / "examples" / "ascend950pr_ops" / "grouped_matmul" / "kernels.py"
    gmm_src = mod_path.read_text(encoding="utf-8")
    spec = CannbenchOpsRegistry.default().get_regime_specs("grouped_matmul")[0]
    extracted = _extract_trackb_cpp_source(gmm_src, spec)
    assert extracted is not None
    sym_name, orig_cpp = extracted
    active = select_active_trackb_kernel(
        orig_cpp,
        regime_tag=spec.regime_tag,
        module_source=gmm_src,
    )
    assert active is not None

    view = build_trackb_context_view(
        orig_cpp,
        active_kernel=active,
        regime_tag=spec.regime_tag,
        module_source=gmm_src,
        trackb_symbol=sym_name,
    )
    marker = "// [READONLY_SIBLING_DECLARATION]"
    assert marker in view

    target_line = (
        "  __ubuf__ BiasT* bias_raw_ub0 = "
        "(__ubuf__ BiasT*)(2 * C_UB_BYTES + 2 * OUT_UB_BYTES);"
    )
    assert orig_cpp.count(target_line) == 1
    sr_patch = (
        "```cpp\n"
        "<<<<<<< SEARCH\n"
        f"{target_line}\n"
        "=======\n"
        f"{target_line} // tuned_m2_marker\n"
        ">>>>>>> REPLACE\n"
        "```\n"
    )
    patched_sr = patch_regime_into_module(gmm_src, spec, sr_patch)
    post_extracted = _extract_trackb_cpp_source(patched_sr, spec)
    assert post_extracted is not None
    assert "// tuned_m2_marker" in post_extracted[1]
    assert "def _select_grouped_matmul_config(" in patched_sr

    full_python = (
        f'{sym_name} = r"""\n'
        + orig_cpp.replace(target_line, f"{target_line} // full_py_marker")
        + '\n"""\n'
    )
    patched_py = patch_regime_into_module(gmm_src, spec, full_python)
    post_py = _extract_trackb_cpp_source(patched_py, spec)
    assert post_py is not None
    assert "// full_py_marker" in post_py[1]


def test_exploit_and_creative_requests_compact_trackb_snippet_under_15kb() -> None:
    """Keep Track-B snippet literals under 15KB in exploit and creative LLM requests."""
    mod_path = REPO_ROOT / "examples" / "ascend950pr_ops" / "grouped_matmul" / "kernels.py"
    gmm_src = mod_path.read_text(encoding="utf-8")
    spec = CannbenchOpsRegistry.default().get_regime_specs("grouped_matmul")[0]
    func_name, snippet = extract_regime_snippet(gmm_src, spec)
    assert len(snippet.encode("utf-8")) > 15360

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="level3/grouped_matmul", max_rounds=1),
        target_module_path=mod_path,
    )
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 100},
        waterlines={},
    )
    surrogate_res = CandidateSurrogateResult(
        candidate_id="base",
        operator_name="level3/grouped_matmul",
        all_passed=True,
        avg_speedup=1.0,
        min_case_speedup=1.0,
        avg_roofline_eta=0.2,
        surrogate_score=50.0,
        anchors=(
            AnchorEvalMetrics(
                case_num=1,
                regime_tag=spec.regime_tag,
                accuracy_passed=True,
                cand_us=10.0,
                ref_perf_us=10.0,
                t_hw_us=2.0,
                speedup=1.0,
                roofline_eta=0.2,
                eq3_score=50.0,
            ),
        ),
    )
    rule = load_seed_rules()[0]
    residual = compute_static_dynamic_residual(ledger, {"dur_us": 10.0})

    exp_req = runner._build_exploit_regime_request(
        snippet_source=snippet,
        func_name=func_name,
        regime_spec=spec,
        rule=rule,
        best_result=surrogate_res,
        parent_ledger=ledger,
        parent_residual=residual,
        incumbent_source=gmm_src,
    )
    cre_req = runner._build_creative_regime_request(
        snippet_source=snippet,
        func_name=func_name,
        regime_spec=spec,
        best_result=surrogate_res,
        effort="high",
        parent_ledger=ledger,
        parent_residual=residual,
        incumbent_source=gmm_src,
    )
    for req in (exp_req, cre_req):
        marker = "### TARGET FUNCTION SNIPPET\n```python\n"
        assert marker in req.user_prompt
        snippet_block = req.user_prompt.split(marker, 1)[1].rsplit("\n```\n", 1)[0]
        assert len(snippet_block.encode("utf-8")) < 51200
        assert '_GMM_TRACKB_SRC = r"""' in snippet_block
        assert "def _select_grouped_matmul_config(" in snippet_block


def test_escalate_primfunc_to_trackb_tu_and_tir_guard() -> None:
    """Lower a real PrimFunc to tl_templates C++20 without #if __cplusplus guard."""
    assert _is_real_tir_primfunc(("v0", 64, 64)) is False
    assert _is_real_tir_primfunc(None) is False

    dsl = """
import tilelang.ascend.language as T

@T.prim_func
def vec_copy_add(
    A: T.Tensor((512,), "float32"),
    B: T.Tensor((512,), "float32"),
    C: T.Tensor((512,), "float32"),
):
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
    assert _is_real_tir_primfunc(pf) is True

    tu_cpp = escalate_primfunc_to_trackb_tu(pf)
    assert "tl_templates/ascend/" in tu_cpp
    assert "#if __cplusplus < 202002L" not in tu_cpp


def test_format_host_launcher_stub_all_branches_and_edge_cases() -> None:
    """Emit real <<<1, nullptr, stream>>> launcher calls across all 3 signature branches."""
    stub_map = _format_host_launcher_stub(
        {"name": "launch_op", "args": [("float*", "x"), "int* y = nullptr", "float arr[4]"]},
        "op_kernel",
    )
    assert "#ifndef __CCE_AICORE__" in stub_map
    assert 'extern "C" inline void launch_op(' in stub_map
    assert "op_kernel<<<1, nullptr, stream>>>(x, y, arr);" in stub_map

    stub_void = _format_host_launcher_stub("void launch_void(void)", "void_kernel")
    assert 'extern "C" void launch_void(void)' in stub_void
    assert "void_kernel<<<1, nullptr, stream>>>();" in stub_void

    stub_empty = _format_host_launcher_stub("void launch_empty()", "empty_kernel")
    assert "empty_kernel<<<1, nullptr, stream>>>();" in stub_empty

    stub_complex = _format_host_launcher_stub(
        "void launch_complex(void* stream, const float* &a, float b[8], int n = 16)",
        "complex_kernel",
    )
    assert "complex_kernel<<<1, nullptr, stream>>>(a, b, n);" in stub_complex

    stub_bare = _format_host_launcher_stub("launch_bare", "bare_kernel")
    assert 'extern "C" inline void launch_bare(void* stream, void* x, void* y)' in stub_bare
    assert "bare_kernel<<<1, nullptr, stream>>>(x, y);" in stub_bare


def test_m4_hub_glue_kb_backend_rendering_and_syntax_fail_closed(tmp_path: Path) -> None:
    """Verify hub-glue Track-B rule rendering and fail-closed SyntaxError pruning."""
    spec = RegimeTargetSpec(
        regime_tag="main",
        builder_candidates=("_build_demo",),
        sample_builder_kwargs={},
        representative_hir_dims={"total_elements": 16384, "M": 16, "D": 64},
        editable_symbols=("_DEMO_TRACKB_SRC", "_select_demo"),
        runtime_backend="ascendc_trackb",
        trackb_symbols=("_DEMO_TRACKB_SRC",),
    )
    mod_file = tmp_path / "demo_mod.py"
    mod_file.write_text(
        '_DEMO_TRACKB_SRC = r"""\n__global__ __vector__ void k() {}\n"""\n'
        "def _select_demo():\n    return {}\n",
        encoding="utf-8",
    )
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="level2/softmax", max_rounds=1),
        target_module_path=mod_file,
        regime_specs=[spec],
    )
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 100},
        waterlines={},
    )
    surrogate_res = CandidateSurrogateResult(
        candidate_id="base",
        operator_name="demo",
        all_passed=True,
        avg_speedup=1.0,
        min_case_speedup=1.0,
        avg_roofline_eta=0.2,
        surrogate_score=50.0,
        anchors=(
            AnchorEvalMetrics(
                case_num=1,
                regime_tag="main",
                accuracy_passed=True,
                cand_us=10.0,
                ref_perf_us=10.0,
                t_hw_us=2.0,
                speedup=1.0,
                roofline_eta=0.2,
                eq3_score=50.0,
            ),
        ),
    )
    dual_rule = AscendCausalRule(
        rule_id="R_M4_TEST",
        regime_guard="True",
        precondition="True",
        transform_template="TILELANG_TEMPLATE_MARKER",
        expected_postcondition="True",
        source_provenance="test",
        compatible_backends=("tilelang_dsl", "ascendc_trackb"),
        cpp20_transform_template="CPP20_TRACKB_TEMPLATE_MARKER",
    )
    req = runner._build_exploit_regime_request(
        snippet_source=mod_file.read_text(encoding="utf-8"),
        func_name="_select_demo",
        regime_spec=spec,
        rule=dual_rule,
        best_result=surrogate_res,
        parent_ledger=ledger,
        parent_residual=compute_static_dynamic_residual(ledger, {"dur_us": 10.0}),
    )
    assert "CPP20_TRACKB_TEMPLATE_MARKER" in req.user_prompt
    assert "TILELANG_TEMPLATE_MARKER" not in req.user_prompt

    with pytest.raises(SyntaxError):
        runner._prune_stale_sibling_stepping_stones(spec, "def broken(:")


def test_prune_stale_sibling_stepping_stones_handles_bare_fenced_trackb_cpp(
    tmp_path: Path,
) -> None:
    """Prune sibling stepping stones touching shared Track-B constants without SyntaxError."""
    promoted_spec = RegimeTargetSpec(
        regime_tag="primary",
        builder_candidates=("_build_primary",),
        sample_builder_kwargs={},
        representative_hir_dims={"total_elements": 16384, "M": 16, "D": 64},
        editable_symbols=("_DEMO_TRACKB_SRC", "_select_primary"),
        runtime_backend="hybrid_trackb",
        trackb_symbols=("_DEMO_TRACKB_SRC",),
    )
    sibling_spec = RegimeTargetSpec(
        regime_tag="sibling",
        builder_candidates=("_build_sibling",),
        sample_builder_kwargs={},
        representative_hir_dims={"total_elements": 16384, "M": 16, "D": 64},
        editable_symbols=("_DEMO_TRACKB_SRC", "_select_sibling"),
        runtime_backend="hybrid_trackb",
        trackb_symbols=("_DEMO_TRACKB_SRC",),
    )
    mod_file = tmp_path / "demo_sibling_mod.py"
    mod_file.write_text(
        '_DEMO_TRACKB_SRC = r"""\n__global__ __vector__ void kernel_active(float* x) {}\n"""\n'
        "def _select_primary():\n    return {}\n"
        "def _select_sibling():\n    return {}\n",
        encoding="utf-8",
    )
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="level2/softmax", max_rounds=1),
        target_module_path=mod_file,
        regime_specs=[promoted_spec, sibling_spec],
    )
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 100},
        waterlines={},
    )
    residual = compute_static_dynamic_residual(ledger, {"dur_us": 10.0})
    fenced_cpp = "```cpp\n__global__ __vector__ void kernel_active(float* x) {\n}\n```"
    py_sibling = "def _select_sibling():\n    return {'tile_m': 8}\n"
    runner._beams["sibling"].stepping_stone_beam = [
        BeamCandidate(
            candidate_id="stone_cpp",
            round_idx=1,
            worker_kind="explore_creative",
            rule_id=None,
            dsl_source=fenced_cpp,
            track="unconstrained",
            ledger=ledger,
            residual=residual,
            latency_us=12.0,
            accuracy_passed=True,
        ),
        BeamCandidate(
            candidate_id="stone_py",
            round_idx=1,
            worker_kind="explore_creative",
            rule_id=None,
            dsl_source=py_sibling,
            track="unconstrained",
            ledger=ledger,
            residual=residual,
            latency_us=11.0,
            accuracy_passed=True,
        ),
    ]

    runner._prune_stale_sibling_stepping_stones(promoted_spec, fenced_cpp)
    assert [
        stone.candidate_id for stone in runner._beams["sibling"].stepping_stone_beam
    ] == ["stone_py"]

    with pytest.raises(SyntaxError):
        runner._prune_stale_sibling_stepping_stones(promoted_spec, "def broken(:")


def test_template_arg_callee_closure_view_patch_and_splice() -> None:
    """Discover template-launched VF helpers and transitive __simt_callee__ helpers."""
    cpp = (
        "#include <tl_templates/ascend/common.h>\n"
        "__simt_callee__ inline float inner_simt_helper(float v) {\n"
        "    return v + 1.0f;\n"
        "}\n"
        "__simt_vf__ void vf_worker(__gm__ float* x) {\n"
        "    x[0] = inner_simt_helper(x[0]);\n"
        "}\n"
        "__simt_callee__ inline float shared_simt_helper(float v) {\n"
        "    return v * 2.0f;\n"
        "}\n"
        "__simt_vf__ void vf_sibling_only(__gm__ float* y) {\n"
        "    y[0] = shared_simt_helper(y[0]);\n"
        "}\n"
        'extern "C" __global__ __vector__ void kernel_active(__gm__ float* x) {\n'
        "    asc_vf_call<vf_worker>(dim3(64, 1, 1), x);\n"
        "    x[1] = shared_simt_helper(x[1]);\n"
        "}\n"
        'extern "C" __global__ __vector__ void kernel_sibling(__gm__ float* y) {\n'
        "    async_invoke<vf_sibling_only>(y);\n"
        "}\n"
    )
    view = build_trackb_context_view(
        cpp,
        active_kernel="kernel_active",
        regime_tag="stream",
        max_bytes=14800,
    )
    assert "vf_worker(__gm__ float* x)" in view
    assert "inner_simt_helper(float v)" in view
    assert "shared_simt_helper(float v)" in view
    assert "/* [READ-ONLY HELPER] */ __simt_vf__ void vf_sibling_only" in view

    patch_worker_and_callee = (
        _hunk("    x[0] = inner_simt_helper(x[0]);", "    x[0] = inner_simt_helper(x[0]) + 3.0f;")
        + "\n"
        + _hunk("    return v + 1.0f;", "    return v + 4.0f;")
    )
    updated, applied = apply_scoped_trackb_patch(
        cpp,
        patch_worker_and_callee,
        active_kernel="kernel_active",
    )
    assert applied == 2
    assert "inner_simt_helper(x[0]) + 3.0f" in updated
    assert "return v + 4.0f;" in updated

    with pytest.raises(ValueError, match="read-only code outside active kernel"):
        apply_scoped_trackb_patch(
            cpp,
            _hunk("    return v * 2.0f;", "    return v * 5.0f;"),
            active_kernel="kernel_active",
        )
    with pytest.raises(ValueError, match="read-only code outside active kernel"):
        apply_scoped_trackb_patch(
            cpp,
            _hunk("    y[0] = shared_simt_helper(y[0]);", "    y[0] = 0.0f;"),
            active_kernel="kernel_active",
        )

    splice_with_new_helper = (
        "__simt_callee__ inline float brand_new_helper(float v) {\n"
        "    return v + 9.0f;\n"
        "}\n"
        'extern "C" __global__ __vector__ void kernel_active(__gm__ float* x) {\n'
        "    asc_vf_call<vf_worker>(dim3(64, 1, 1), x);\n"
        "    x[1] = brand_new_helper(x[1]);\n"
        "}\n"
    )
    spliced, count = splice_cpp_units_into_tu(
        cpp,
        splice_with_new_helper,
        active_kernel="kernel_active",
    )
    assert count == 2
    assert "brand_new_helper" in spliced
    assert spliced.index("brand_new_helper") < spliced.index("kernel_active(__gm__ float* x)")

    with pytest.raises(ValueError, match="outside the editable scope"):
        splice_cpp_units_into_tu(
            cpp,
            "__simt_callee__ inline float unreferenced_helper(float v) {\n    return v;\n}\n",
            active_kernel="kernel_active",
        )
    with pytest.raises(ValueError, match="outside the editable scope"):
        splice_cpp_units_into_tu(
            cpp,
            "__simt_vf__ void vf_sibling_only(__gm__ float* y) {\n    y[0] = 0.0f;\n}\n",
            active_kernel="kernel_active",
        )


def test_split_and_splice_recognize_static_inline_aicore_and_bare_helpers() -> None:
    """Parse static inline, static __aicore__, and bare multi-line helpers in splicing."""
    cpp = (
        "#include <tl_templates/ascend/common.h>\n"
        "static inline float helper_si(float v) {\n"
        "    return v + 1.0f;\n"
        "}\n"
        "static __aicore__ void helper_sa(__gm__ float* p) {\n"
        "    p[0] = helper_si(p[0]);\n"
        "}\n"
        "float\n"
        "bare_multiline_helper(\n"
        "    float a,\n"
        "    float b\n"
        ") {\n"
        "    return a + b;\n"
        "}\n"
        'extern "C" __global__ __vector__ void kernel_active(__gm__ float* x) {\n'
        "    helper_sa(x);\n"
        "    x[1] = bare_multiline_helper(x[0], 1.0f);\n"
        "}\n"
        'extern "C" __global__ __vector__ void kernel_sibling(__gm__ float* y) {\n'
        "    y[0] = 2.0f;\n"
        "}\n"
    )
    _, units = _split_cpp_units(cpp)
    helper_names = [u.name for u in units if u.kind == "helper"]
    assert helper_names == ["helper_si", "helper_sa", "bare_multiline_helper"]

    cand = (
        "static inline float brand_new_si(float v) {\n"
        "    return v * 3.0f;\n"
        "}\n"
        "static __aicore__ void brand_new_sa(__gm__ float* p) {\n"
        "    p[2] = brand_new_si(p[1]);\n"
        "}\n"
        'extern "C" __global__ __vector__ void kernel_active(__gm__ float* x) {\n'
        "    helper_sa(x);\n"
        "    brand_new_sa(x);\n"
        "    x[1] = bare_multiline_helper(x[0], 1.0f);\n"
        "}\n"
    )
    spliced, count = splice_cpp_units_into_tu(cpp, cand, active_kernel="kernel_active")
    assert count == 3
    assert "brand_new_si" in spliced
    assert "brand_new_sa" in spliced
    assert "kernel_sibling" in spliced


def test_extract_reachable_trackb_source_and_evaluate_dead_cube_and_single_lane_simt() -> None:
    """Resolve launched kernel, prune dead cube/helpers, and reject single-lane SIMT traps."""
    cpp = (
        "#include <tl_templates/ascend/common.h>\n"
        "struct TileCfg { int m; };\n"
        "enum PipeKind { P_VEC = 0 };\n"
        "using ScalarT = float;\n"
        "typedef int IndexT;\n"
        "constexpr int kTile = 64;\n"
        "__aicore__ inline void dead_aicore_helper(__gm__ float* p) {\n"
        "    asc_mmad(p, p, p, 16, 16, 16, asc_unit_flag_mode::DISABLE, true, false, true);\n"
        "}\n"
        "__simd_vf__ inline void dead_simdvf_helper(__ubuf__ float* u) {\n"
        "    u[0] = 0.0f;\n"
        "}\n"
        'extern "C" __global__ __aicore__ void gqa_cube_mmad(__gm__ float* q, __gm__ float* out) {\n'
        "    dead_aicore_helper(q);\n"
        "    out[0] = q[0];\n"
        "}\n"
        "__simt_callee__ inline float live_simt_helper(float v) {\n"
        "    return v + 1.0f;\n"
        "}\n"
        "__simt_vf__ __launch_bounds__(128) void live_vf(__gm__ float* q, __gm__ float* out) {\n"
        "    out[0] = live_simt_helper(q[0]);\n"
        "}\n"
        'extern "C" __global__ __vector__ void gqa_kernel(__gm__ float* q, __gm__ float* out) {\n'
        "    asc_vf_call<live_vf>(dim3(128, 1, 1), q, out);\n"
        "}\n"
        'extern "C" int launch_gqa(void* stream, void* q, void* out) {\n'
        "    gqa_kernel<<<28, nullptr, stream>>>((__gm__ float*)q, (__gm__ float*)out);\n"
        "    return 0;\n"
        "}\n"
    )
    module_src = (
        f'CPP_SRC = r"""\n{cpp}\n"""\n'
        "def run_op(lib, stream, q, out):\n"
        "    return lib.launch_gqa(stream, q, out)\n"
    )
    assert select_active_trackb_kernel(cpp, module_source=module_src) == "gqa_kernel"
    entry, reachable = extract_reachable_trackb_source(cpp, module_source=module_src)
    assert entry == "gqa_kernel"
    for kept in ("struct TileCfg", "enum PipeKind", "using ScalarT", "typedef int IndexT", "kTile"):
        assert kept in reachable
    assert "live_vf" in reachable
    assert "live_simt_helper" in reachable
    assert "gqa_cube_mmad" not in reachable
    assert "dead_aicore_helper" not in reachable
    assert "dead_simdvf_helper" not in reachable
    assert "asc_mmad" not in reachable

    cube_res = evaluate_ascendc_candidate(
        cpp,
        module_source=module_src,
        require_cube_engine=True,
    )
    assert cube_res.passed is False
    assert "missing_cube_engine_primitives" in cube_res.error_message

    simt_1_bounds = (
        "#include <tl_templates/ascend/common.h>\n"
        "__simt_vf__ __launch_bounds__(1) void bad_vf(__gm__ float* out) {\n"
        "    out[0] = 1.0f;\n"
        "}\n"
        'extern "C" __global__ __vector__ void bad_kernel(__gm__ float* out) {\n'
        "    asc_vf_call<bad_vf>(dim3(128, 1, 1), out);\n"
        "}\n"
    )
    res_1_bounds = evaluate_ascendc_candidate(simt_1_bounds)
    assert res_1_bounds.passed is False
    assert "degenerated_single_lane_simt" in res_1_bounds.error_message

    simt_1_dim3 = simt_1_bounds.replace("__launch_bounds__(1)", "__launch_bounds__(128)").replace(
        "dim3(128, 1, 1)", "dim3(1, 1, 1)"
    )
    res_1_dim3 = evaluate_ascendc_candidate(simt_1_dim3)
    assert res_1_dim3.passed is False
    assert "degenerated_single_lane_simt" in res_1_dim3.error_message

    empty_ratios = CasePhysicsRatios(
        case_id=1,
        stride_ratio=1.0,
        contig_atom_ratio=1.0,
        is_atom_aligned=True,
        working_set_sram_ratio=0.1,
        working_set_vreg_ratio=0.1,
        wave_occupancy_ratio=0.25,
        oi_ridge_ratio=1.0,
        inner_contiguous_bytes=64,
        has_empty_group=True,
    )
    assert (
        evaluate_ascendc_candidate(
            simt_1_bounds,
            freedom_mask=ConstraintRelaxationMask.top(),
            case_ratios=empty_ratios,
        ).passed
        is True
    )


def test_markdown_fence_stripping_and_agent_gates_relative_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strip markdown fences in Track-B extraction/splicing and resolve agent-gates cleanly."""
    fenced_patch = (
        "```cpp\n"
        "<<<<<<< SEARCH\n"
        "    int step = 1;\n"
        "=======\n"
        "```c++\n"
        "    int step = 4;\n"
        "```\n"
        ">>>>>>> REPLACE\n"
        "```\n"
    )
    hunks = parse_scoped_search_replace_blocks(fenced_patch)
    assert len(hunks) == 1
    assert "```" not in hunks[0][1]
    assert "int step = 4;" in hunks[0][1]

    base_tu = (
        "#include <tl_templates/ascend/common.h>\n"
        'extern "C" __global__ __vector__ void kernel_active(__gm__ float* x) {\n'
        "    x[0] = 1.0f;\n"
        "}\n"
    )
    fenced_cand = (
        "```cpp\n"
        'extern "C" __global__ __vector__ void kernel_active(__gm__ float* x) {\n'
        "```\n"
        "    x[0] = 2.0f;\n"
        "}\n"
        "```\n"
    )
    spliced, count = splice_cpp_units_into_tu(base_tu, fenced_cand, active_kernel="kernel_active")
    assert count == 1
    assert "x[0] = 2.0f;" in spliced
    assert "```" not in spliced

    ag_dir = tmp_path / "AgentGates"
    ag_bin = ag_dir / ".venv" / "bin" / "agent-gates"
    ag_bin.parent.mkdir(parents=True)
    ag_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    env_file = ag_dir / "configs" / "npu.env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text("", encoding="utf-8")

    monkeypatch.delenv("TILEFOUNDRY_NPU_AGENT_GATES_BIN", raising=False)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.setenv("TILEFOUNDRY_NPU_ENV_FILE", str(env_file))
    assert _resolve_agent_gates_bin() == str(ag_bin)


def test_build_trackb_context_view_preserves_diagnostic_focus_window_and_helper() -> None:
    """Preserve middle-line focus windows and focused helpers within max_bytes."""
    assert extract_cpp_diagnostic_lines(
        "kernel.cce:210:17: error: bad op\n"
        "<source>:45: fatal error: missing\n"
        "input:12:3: warning: unused\n"
        "failed at line 305\n"
    ) == (12, 45, 210, 305)

    header = ["#include <tl_templates/ascend/common.h>", ""]
    body = [
        'extern "C" __global__ __vector__ void big_focus_kernel(__gm__ float* x, __gm__ float* out) {'
    ]
    for idx in range(1, 410):
        body.append(
            f"    float v_{idx:03d} = x[{idx}] + 0.125f * (float){idx}; /* pad_{idx:03d}_payload */"
        )
    body.append("    out[0] = v_409;")
    body.append("}")
    cpp_lines = header + body
    cpp_source = "\n".join(cpp_lines) + "\n"
    assert len(cpp_source.encode("utf-8")) > 20480

    view = build_trackb_context_view(
        cpp_source,
        active_kernel="big_focus_kernel",
        max_bytes=12000,
        diagnostic_text="kernel.cce:210:17: error: invalid vector operand",
    )
    assert "LINES OMITTED" not in view
    assert "MIDDLE BODY OMITTED" not in view
    assert "extern \"C\" __global__ __vector__ void big_focus_kernel" in view
    assert "out[0] = v_409;" in view
    for lineno in range(1, len(cpp_lines) + 1):
        if cpp_lines[lineno - 1].strip():
            assert cpp_lines[lineno - 1] in view

    helper_lines = [
        "#include <tl_templates/ascend/common.h>",
        "__simd_vf__ inline void focused_helper(__ubuf__ float* u) {",
    ]
    for h_idx in range(1, 20):
        helper_lines.append(f"    u[{h_idx}] = u[{h_idx}] + 1.0f; /* helper_step_{h_idx:02d} */")
    helper_lines.append("}")
    focus_helper_line = 12
    assert "helper_step_" in helper_lines[focus_helper_line - 1]
    big_caller = [
        'extern "C" __global__ __vector__ void big_caller(__gm__ float* x, __ubuf__ float* u) {',
        "    focused_helper(u);",
    ]
    for idx in range(1, 360):
        big_caller.append(
            f"    float k_{idx:03d} = x[{idx}] * 2.0f + (float){idx}; /* caller_pad_{idx:03d} */"
        )
    big_caller.append("}")
    helper_cpp = "\n".join(helper_lines + big_caller) + "\n"
    assert len(helper_cpp.encode("utf-8")) > 16384

    helper_view = build_trackb_context_view(
        helper_cpp,
        active_kernel="big_caller",
        max_bytes=10000,
        diagnostic_text=f"kernel.cce:{focus_helper_line}:5: error: helper op mismatch",
    )
    assert "LINES OMITTED" not in helper_view
    assert "MIDDLE BODY OMITTED" not in helper_view
    assert "/* [HELPER SUMMARY] */" not in helper_view
    assert helper_lines[focus_helper_line - 1] in helper_view


def test_evaluate_ascendc_candidate_accepts_templated_ascend_gemm_l1() -> None:
    """Accept templated ascend_gemm_l1 calls under require_cube_engine and record AIC.PIPE_M."""
    cpp = (
        "#include <tl_templates/ascend/common.h>\n"
        'extern "C" __global__ __aicore__ void cube_shell_kernel(\n'
        "    __gm__ half* a, __gm__ half* b, __gm__ float* c\n"
        ") {\n"
        "    __cbuf__ half* l1_a = (__cbuf__ half*)0;\n"
        "    __cbuf__ half* l1_b = (__cbuf__ half*)32768;\n"
        "    __cc__ float* l0c = (__cc__ float*)0;\n"
        "    asc_copy_gm2l1(l1_a, a, 128, 32);\n"
        "    asc_copy_gm2l1(l1_b, b, 32, 64);\n"
        "    ascend_gemm_l1<128, 64, 128, 32, false, half, float>(l1_a, l1_b, l0c, true);\n"
        "    asc_copy_l0c2gm(c, l0c, 128, 64);\n"
        "}\n"
    )
    res = evaluate_ascendc_candidate(
        cpp,
        freedom_mask=ConstraintRelaxationMask.top(),
        require_cube_engine=True,
    )
    assert res.passed is True, res.error_message
    assert res.ledger is not None
    assert int(res.ledger.pipe_cycles.get("AIC.PIPE_M", 0)) > 0

    sb = StageBResult(passed=False, generated_cce="cce", lowered_tir="tir")
    assert sb.generated_cce == "cce"
    assert sb.lowered_tir == "tir"
    drv = build_stage_b_driver_source("/tmp/x.py", tir_dump_path="/tmp/lowered_tir.tir")
    assert "tir_dump_path" in drv
    assert "/tmp/lowered_tir.tir" in drv


def test_wired_p2_hotspot_and_repair_focus_preservation(tmp_path: Path) -> None:
    """Verify campaign_runner p2_diag and surrogate repair retain focused Track-B lines."""
    import asyncio  # noqa: PLC0415

    cpp_body = [
        "#include <tl_templates/ascend/common.h>",
        'extern "C" __global__ __vector__ void wired_focus_kernel(__gm__ float* x, __gm__ float* y) {',
    ]
    for idx in range(1, 900):
        cpp_body.append(
            f"    float r_{idx:03d} = x[{idx}] + 0.5f * (float){idx}; /* wire_payload_{idx:03d} */"
        )
    cpp_body.append("    y[0] = r_899;")
    cpp_body.append("}")
    cpp_text = "\n".join(cpp_body) + "\n"
    assert len(cpp_text.encode("utf-8")) > 49152

    mod_file = tmp_path / "wired_mod.py"
    mod_file.write_text(
        f'_WIRED_CPP = r"""\n{cpp_text}"""\n'
        "def _select_wired():\n    return {}\n",
        encoding="utf-8",
    )
    spec = RegimeTargetSpec(
        regime_tag="main",
        builder_candidates=("_select_wired",),
        sample_builder_kwargs={},
        representative_hir_dims={"total_elements": 4096, "M": 64, "D": 64},
        editable_symbols=("_WIRED_CPP", "_select_wired"),
        runtime_backend="ascendc_trackb",
        trackb_symbols=("_WIRED_CPP",),
    )
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="level2/softmax", max_rounds=1),
        target_module_path=mod_file,
        regime_specs=[spec],
    )
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 100},
        waterlines={},
    )
    surrogate_res = CandidateSurrogateResult(
        candidate_id="base",
        operator_name="level2/softmax",
        all_passed=True,
        avg_speedup=1.0,
        min_case_speedup=1.0,
        avg_roofline_eta=0.2,
        surrogate_score=50.0,
    )
    rule = AscendCausalRule(
        rule_id="R_WIRE_TEST",
        regime_guard="True",
        precondition="True",
        transform_template="NOP",
        expected_postcondition="True",
        source_provenance="test",
        compatible_backends=("ascendc_trackb",),
        cpp20_transform_template="Optimize middle loop.",
    )
    req = runner._build_exploit_regime_request(
        snippet_source=mod_file.read_text(encoding="utf-8"),
        func_name="_select_wired",
        regime_spec=spec,
        rule=rule,
        best_result=surrogate_res,
        parent_ledger=ledger,
        parent_residual=compute_static_dynamic_residual(ledger, {"dur_us": 10.0}),
        p2_diag={
            "p2_hotspots": [{"source_location": "kernel.cce:450", "instruction_count": 320}]
        },
    )
    assert "wire_payload_448" in req.user_prompt
    assert "LINES OMITTED OUTSIDE FOCUS WINDOW" not in req.user_prompt

    captured_prompts: list[str] = []

    class _RecordingRepairBackend:
        async def generate(self, request: object) -> str:
            captured_prompts.append(str(getattr(request, "user_prompt", "")))
            return cpp_text

    checks = asyncio.run(
        run_l0_l1_round_filter(
            [("cand_wire", cpp_text)],
            patch_fn=lambda src: src,
            l0_checker=lambda _src: StageAResult(passed=True, prim_func=None, ledger=ledger),
            l1_compiler=lambda _src: StageBResult(
                passed=False,
                error_message="kernel.cce:450:9: error: invalid operand",
            ),
            repair_backend=_RecordingRepairBackend(),
            max_syntax_repairs=1,
            backend_kind="ascendc_trackb",
        )
    )
    assert len(checks) == 1
    assert len(captured_prompts) == 1
    assert "wire_payload_448" in captured_prompts[0]
    assert "LINES OMITTED OUTSIDE FOCUS WINDOW" not in captured_prompts[0]
