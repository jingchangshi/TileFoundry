"""Unit tests for scoped C++20 slicing, patching, and Track-A-to-Track-B escalation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tilefoundry_ascend.agent.campaign.anchor_specs import (
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
)
from tilefoundry_ascend.agent.kb.schema import AscendCausalRule, load_seed_rules
from tilefoundry_ascend.agent.orchestrator.cpp_slice_protocol import (
    apply_scoped_trackb_patch,
    build_trackb_context_view,
    escalate_primfunc_to_trackb_tu,
    resolve_instantiated_trackb_kernel,
    select_active_trackb_kernel,
    splice_cpp_units_into_tu,
)
from tilefoundry_ascend.agent.orchestrator.gate_funnel import _resolve_prim_func_from_dsl
from tilefoundry_ascend.agent.orchestrator.residual_beam import compute_static_dynamic_residual
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
    assert len(view.encode("utf-8")) < 15360
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
    marker = "/* [READ-ONLY SIBLING KERNEL:"
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
        assert len(snippet_block.encode("utf-8")) < 15360
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
    """Verify M4 Track-B rule rendering and fail-closed SyntaxError pruning in campaign_runner."""
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
