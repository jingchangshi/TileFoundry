"""Tests for C++20 closure views, Host-Device ABI slices, scoped patching, and CoW guard."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from tilefoundry_ascend.agent import (
    BayesianRuleTracker,
    BeamCandidate,
)
from tilefoundry_ascend.agent.orchestrator import compute_static_dynamic_residual
from tilefoundry_ascend.agent.orchestrator.regime_slice_patcher import (
    DEFAULT_READONLY_HOST_EXPORTS,
    apply_ast_patch,
    extract_regime_snippet,
    patch_regime_into_module,
    resolve_regime_config,
    split_readonly_host_tail,
)
from tilefoundry_ascend.agent.gates.cpp_slice_protocol import (
    DEFAULT_PROTECTED_EXPORTS,
    TRACKB_REPLY_CONTRACT_PROMPT,
    AmbiguousPatchAnchorError,
    NoopIdenticalPatchError,
    PatchAnchorMissError,
    RepairClosureSlice,
    apply_scoped_trackb_patch,
    build_trackb_context_view,
    classify_stage_failure_for_repair,
    compact_creative_source_slice,
    extract_cpp_diagnostic_lines,
    extract_reachable_trackb_source,
    extract_repair_closure_slice,
    extract_scoped_trackb_kernel_view,
    extract_trackb_launcher_preview,
    format_trackb_reply_contract,
    parse_scoped_search_replace_blocks,
    render_active_symbol_closure_view,
    render_active_symbol_closure_with_line_map,
    resolve_instantiated_trackb_kernel,
    scan_launched_trackb_kernels,
    select_active_trackb_kernel,
    splice_cpp_units_by_symbol,
    splice_cpp_units_into_tu,
    split_tracka_editable_and_readonly_tail,
)
from tilefoundry_ascend.agent.orchestrator.portfolio_runner import (
    build_creative_request,
    build_exploit_request,
    build_syntax_repair_request,
)
from tilefoundry_ascend.agent.specs.anchor_specs import RegimeTargetSpec
from tilefoundry_ascend.ascend.passes import build_objective_physical_ledger

_SCOPED_CPP = (
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
    "__simd_vf__ inline void vf_exclusive(__ubuf__ float* u) {\n"
    "  int step = 1;\n"
    "  u[0] = step;\n"
    "}\n"
    'extern "C" __global__ __vector__ void kernel_active(__gm__ float* x) {\n'
    "    asc_vf_call<vf_worker>(dim3(64, 1, 1), x);\n"
    "    x[1] = shared_simt_helper(x[1]);\n"
    "}\n"
    'extern "C" __global__ __vector__ void kernel_sibling(__gm__ float* y) {\n'
    "    async_invoke<vf_sibling_only>(y);\n"
    "}\n"
)

_SHARED_HELPER_MODULE = '''import tilelang.ascend.language as T

_KERNEL_CACHE = {"kept": True}

def _stage_leaf(val: int) -> int:
    return val + 1

def _emit_pass1_compute(scale: int) -> int:
    return _stage_leaf(scale) * 2

def _exclusive_a_helper(x: int) -> int:
    return x + 3

def _select_regime_a_config(d_dim: int = 128) -> dict[str, object]:
    return {"tile": d_dim, "pass_configs": {"tl.disable_auto_sync": True}, "compile_flags": ["-O2"]}

def _build_regime_a(tile: int = 1) -> object:
    step = _emit_pass1_compute(tile) + _exclusive_a_helper(1)
    @T.prim_func
    def main(X: T.Tensor((128, 128), "bfloat16"), Y: T.Tensor((128, 128), "bfloat16")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((128, 128), "bfloat16")
            T.copy(X[0:128, 0:128], a_ub)
            T.copy(a_ub, Y[0:128, 0:128])
    return main, step

def _select_regime_b_config(d_dim: int = 64) -> dict[str, object]:
    return {"tile": d_dim}

def _build_regime_b(tile: int = 1) -> object:
    step = _emit_pass1_compute(tile)
    @T.prim_func
    def main(X: T.Tensor((64, 64), "bfloat16"), Y: T.Tensor((64, 64), "bfloat16")):
        with T.Kernel(1):
            b_ub = T.alloc_shared((64, 64), "bfloat16")
            T.copy(X[0:64, 0:64], b_ub)
            T.copy(b_ub, Y[0:64, 0:64])
    return main, step

def kernel(x: int) -> object:
    return _KERNEL_CACHE, _build_regime_a(), _build_regime_b(), x

ref_kernel = kernel
__all__ = ["kernel", "ref_kernel"]
'''


def _hunk(search: str, replace: str, *, window: str = "") -> str:
    hdr = f"<<<<<<< SEARCH [{window}]" if window else "<<<<<<< SEARCH"
    return f"{hdr}\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


def _ast_sha256(module_source: str, symbol_name: str) -> str:
    tree = ast.parse(module_source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == symbol_name:
            return hashlib.sha256(ast.dump(node).encode("utf-8")).hexdigest()
    raise AssertionError(f"Missing symbol {symbol_name}")


def _regime_specs() -> tuple[RegimeTargetSpec, RegimeTargetSpec]:
    return (
        RegimeTargetSpec(
            regime_tag="regime_a",
            builder_candidates=("_build_regime_a",),
            sample_builder_kwargs={"tile": 1},
            config_selector_fn="_select_regime_a_config",
            editable_symbols=("_select_regime_a_config", "_build_regime_a"),
            allowed_helper_prefix="_regime_a_",
        ),
        RegimeTargetSpec(
            regime_tag="regime_b",
            builder_candidates=("_build_regime_b",),
            sample_builder_kwargs={"tile": 1},
            config_selector_fn="_select_regime_b_config",
            editable_symbols=("_select_regime_b_config", "_build_regime_b"),
            allowed_helper_prefix="_regime_b_",
        ),
    )


def test_reachable_call_graph_closure_and_launcher_resolution() -> None:
    """Resolve launched kernel while masking comments/strings and pruning dead units."""
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
        'extern "C" __global__ __aicore__ void gqa_cube_mmad(__gm__ float* q) {\n'
        "    dead_aicore_helper(q);\n"
        "}\n"
        "__simt_callee__ inline float live_simt_helper(float v) { return v + 1.0f; }\n"
        "__simt_vf__ __launch_bounds__(128) void live_vf(__gm__ float* q, __gm__ float* out) {\n"
        "    out[0] = live_simt_helper(q[0]);\n"
        "}\n"
        'extern "C" __global__ __vector__ void gqa_kernel(__gm__ float* q, __gm__ float* out) {\n'
        "    asc_vf_call<live_vf>(dim3(128, 1, 1), q, out);\n"
        "}\n"
        'extern "C" int launch_gqa(void* stream, void* q, void* out) {\n'
        "    // gqa_cube_mmad<<<1, nullptr, stream>>>((__gm__ float*)q);\n"
        '    const char* doc = "gqa_cube_mmad<<<1, nullptr, stream>>>(q);";\n'
        "    gqa_kernel<<<28, nullptr, stream>>>((__gm__ float*)q, (__gm__ float*)out);\n"
        "    gqa_kernel<<<14, nullptr, stream>>>((__gm__ float*)q, (__gm__ float*)out);\n"
        "    return 0;\n"
        "}\n"
    )
    module_src = (
        f'CPP_SRC = r"""\n{cpp}"""\ndef run(lib, s, q, o): return lib.launch_gqa(s, q, o)\n'
    )
    launch_tu = (
        'extern "C" int launch_gqa(void* s, void* q, void* o) {\n'
        "    gqa_kernel<<<28, nullptr, s>>>((__gm__ float*)q, (__gm__ float*)o);\n"
        "    return 0;\n"
        "}\n"
    )
    assert select_active_trackb_kernel(cpp, module_source=module_src) == "gqa_kernel"
    assert resolve_instantiated_trackb_kernel(launch_tu, cpp) == "gqa_kernel"
    assert resolve_instantiated_trackb_kernel("", cpp) is None
    launched, primary, ordered = scan_launched_trackb_kernels(
        launch_tu, cpp, {"gqa_kernel", "gqa_cube_mmad"}
    )
    assert launched == {"gqa_kernel"}
    assert primary == {"gqa_kernel"}
    assert ordered == ("gqa_kernel",)

    ambig = cpp.replace(
        "    gqa_kernel<<<14, nullptr, stream>>>",
        "    gqa_cube_mmad<<<14, nullptr, stream>>>",
    )
    assert resolve_instantiated_trackb_kernel(ambig, cpp) is None

    entry, reachable = extract_reachable_trackb_source(cpp, module_source=module_src)
    assert entry == "gqa_kernel"
    for kept in ("struct TileCfg", "enum PipeKind", "using ScalarT", "typedef int IndexT", "kTile"):
        assert kept in reachable
    assert "live_vf" in reachable and "live_simt_helper" in reachable
    assert "void gqa_cube_mmad(" not in reachable
    assert "dead_aicore_helper" not in reachable
    assert "asc_mmad" not in reachable


def test_lossless_closure_line_map_and_context_view_over_50kb() -> None:
    """Render 100% lossless active closure with [L0001] anchors and single-line siblings."""
    header = [
        "#include <tl_templates/ascend/common.h>",
        "struct ActiveParams { int count; };",
        "__simd_vf__ inline float active_leaf(float v) { return v * 2.0f; }",
        "__simd_vf__ inline float active_mid(float v) { return active_leaf(v) + 0.5f; }",
        "__simd_vf__ inline float sibling_only(float v) { return v - 3.0f; }",
    ]
    active = ['extern "C" __global__ __vector__ void kernel_active(__gm__ float* x) {']
    active.extend(
        f"    float reg_{i:03d} = active_mid(x[{i}]) + 0.25f; /* pad_{i:03d}_payload */"
        for i in range(1, 780)
    )
    active.extend(["    x[0] = reg_779;", "}"])
    sibling = [
        'extern "C" __global__ __vector__ void kernel_sibling(__gm__ float* y) {',
        "    float sib_secret = sibling_only(y[0]);",
        "    y[0] = sib_secret;",
        "}",
    ]
    cpp = "\n".join(header + active + sibling) + "\n"
    assert len(cpp.encode("utf-8")) > 52_000

    view, line_map = render_active_symbol_closure_with_line_map(cpp, "kernel_active")
    assert view == render_active_symbol_closure_view(cpp, "kernel_active")
    assert "MIDDLE BODY OMITTED" not in view
    assert "LINES OMITTED" not in view
    assert "sib_secret" not in view
    assert "/* [READ-ONLY HELPER] */" in view
    assert (
        '// [READONLY_SIBLING_DECLARATION] extern "C" __global__ __vector__ '
        "void kernel_sibling(__gm__ float* y);"
    ) in view
    assert "float reg_400 = active_mid" in view
    assert "float reg_779 = active_mid" in view

    tu_lines = cpp.splitlines()
    rendered_lines = view.splitlines()
    for out_ln, tu_ln in line_map.items():
        prefix = f"[L{tu_ln:04d}] "
        assert rendered_lines[out_ln - 1].startswith(prefix)
        assert rendered_lines[out_ln - 1][len(prefix) :] == tu_lines[tu_ln - 1]

    assert extract_scoped_trackb_kernel_view is build_trackb_context_view
    scoped = build_trackb_context_view(cpp, active_kernel="kernel_active", max_bytes=4096)
    assert "float reg_400 = active_mid" in scoped
    assert "MIDDLE BODY OMITTED" not in scoped
    assert compact_creative_source_slice(cpp, active_kernel="kernel_active")

    assert extract_cpp_diagnostic_lines(
        "kernel.cce:210:17: error: bad\n<source>:45: fatal error: x\nline 305\n"
    ) == (45, 210, 305)
    assert classify_stage_failure_for_repair("UB_SRAM_OVERFLOW") is False
    assert classify_stage_failure_for_repair("NOOP_IDENTICAL") is False
    assert classify_stage_failure_for_repair("STAGE_B_COMPILE_FAILED") is True


def test_host_device_abi_slice_and_readonly_host_tail_isolation() -> None:
    """Separate device closure, sibling protos, and <read_only_host_tail> cleanly."""
    hybrid_mod = (
        "import tilelang.ascend.language as T\n\n"
        '_SRC = r"""\n'
        "#include <tl_templates/ascend/common.h>\n"
        "__simd_vf__ inline float used_vf(float v) { return v + 1.0f; }\n"
        "__simd_vf__ inline float unused_vf(float v) { return v - 1.0f; }\n"
        'extern "C" __global__ __vector__ void active_k(__gm__ float* x) {\n'
        "    x[0] = used_vf(x[0]);\n"
        "}\n"
        'extern "C" __global__ __vector__ void sib_k(__gm__ float* y) {\n'
        "    y[0] = unused_vf(y[0]);\n"
        "}\n"
        'extern "C" int launch_op(void* stream, void* x) {\n'
        "    active_k<<<28, nullptr, stream>>>((__gm__ float*)x);\n"
        "    return 0;\n"
        "}\n"
        '"""\n\n'
        "def _select_decode_config(d: int = 64):\n    return {'d': d}\n\n"
        "def _render_trackb_tu(**dims):\n    return dims['trackb_src']\n\n"
        "def _build_dead_prim_func():\n"
        "    @T.prim_func\n"
        "    def main():\n        pass\n"
        "    return main\n\n"
        "def kernel(x):\n    return x\n"
    )
    hybrid_slice = extract_repair_closure_slice(
        hybrid_mod,
        active_symbols=("active_k",),
        regime_tag="decode",
        backend_kind="hybrid_trackb",
    )
    assert isinstance(hybrid_slice, RepairClosureSlice)
    assert "active_k" in hybrid_slice.editable_device_source
    assert "used_vf" in hybrid_slice.editable_device_source
    assert "_build_dead_prim_func" not in hybrid_slice.editable_device_source
    assert any("sib_k" in p for p in hybrid_slice.sibling_prototypes)
    assert "def _render_trackb_tu" in hybrid_slice.readonly_host_abi_tail
    assert 'extern "C" int launch_op' in hybrid_slice.readonly_host_abi_tail
    assert "launch_op" in extract_trackb_launcher_preview(hybrid_mod)

    tracka_mod = (
        "import tilelang.ascend.language as T\n\n"
        "def _used_helper(x):\n    return x + 1\n\n"
        "def _dead_helper(y):\n    return y * 2\n\n"
        "def _select_decode_config(d: int):\n    return {'bm': 32}\n\n"
        "def _build_decode_func(bm: int = 32):\n"
        "    @T.prim_func\n"
        "    def main():\n        _used_helper(bm)\n"
        "    return main\n\n"
        "def _build_prefill_func(bm: int = 128):\n"
        "    @T.prim_func\n"
        "    def main():\n        _dead_helper(bm)\n"
        "    return main\n\n"
        "kernel = _build_decode_func\n"
        "__all__ = ['kernel']\n"
    )
    py_slice = extract_repair_closure_slice(
        tracka_mod,
        regime_tag="decode",
        backend_kind="tilelang_dsl",
    )
    assert "def _used_helper" in py_slice.editable_device_source
    assert "def _build_decode_func" in py_slice.editable_device_source
    assert "def _dead_helper" not in py_slice.editable_device_source
    assert "def _build_prefill_func(bm: int=128): ..." in py_slice.sibling_prototypes
    assert "kernel = _build_decode_func" in py_slice.readonly_host_abi_tail

    editable, tail = split_tracka_editable_and_readonly_tail(tracka_mod, DEFAULT_PROTECTED_EXPORTS)
    assert "def _build_decode_func" in editable
    assert "kernel =" not in editable
    assert tail.startswith("kernel = _build_decode_func")

    broken_edit, broken_tail = split_tracka_editable_and_readonly_tail(
        "def _build_op():\n    if True\n        pass\nkernel = _build_op\n"
    )
    assert "def _build_op" in broken_edit
    assert broken_tail == "kernel = _build_op"

    snip_edit, snip_tail = split_readonly_host_tail(
        "def _helper():\n    return 1\n\ndef kernel(x):\n    return x\n",
        DEFAULT_READONLY_HOST_EXPORTS,
    )
    assert "def _helper" in snip_edit and "kernel" not in snip_edit
    assert snip_tail.startswith("def kernel(x):")

    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=8,
        pipe_cycles={"AIV0.PIPE_V": 32},
        waterlines={},
    )
    parent = BeamCandidate(
        candidate_id="p0",
        round_idx=0,
        worker_kind="seed",
        rule_id=None,
        dsl_source=tracka_mod,
        track="unconstrained",
        ledger=ledger,
        residual=compute_static_dynamic_residual(ledger, {"dur_us": 10.0}),
        latency_us=10.0,
        accuracy_passed=True,
    )
    rule = BayesianRuleTracker.from_seed_yaml().active_rules()[0]
    for req in (
        build_exploit_request(parent, rule),
        build_creative_request(parent),
        build_syntax_repair_request(tracka_mod, "err", parent_source=tracka_mod),
    ):
        assert "<read_only_host_tail>" in req.user_prompt
        assert "kernel = _build_decode_func" in req.user_prompt
        assert "<<<<<<< SEARCH" in req.system_prompt
        if "### CURRENT KERNEL SOURCE\n```python\n" in req.user_prompt:
            code_blk = req.user_prompt.split("### CURRENT KERNEL SOURCE\n```python\n", 1)[1].split(
                "\n```\n", 1
            )[0]
            assert "kernel = _build_decode_func" not in code_blk


def test_scoped_trackb_search_replace_and_mode_a_splicing() -> None:
    """Enforce Mode-A/Mode-B scoping, window disambiguation, noop faults, and splicing."""
    assert "Mode A" in TRACKB_REPLY_CONTRACT_PROMPT
    assert "target_k" in format_trackb_reply_contract("target_k")
    assert splice_cpp_units_by_symbol is splice_cpp_units_into_tu

    patch = _hunk(
        "    x[0] = inner_simt_helper(x[0]);",
        "    x[0] = inner_simt_helper(x[0]) + 3.0f;",
    )
    patch += "\n" + _hunk("    return v + 1.0f;", "    return v + 4.0f;")
    updated, count = apply_scoped_trackb_patch(_SCOPED_CPP, patch, active_kernel="kernel_active")
    assert count == 2
    assert "inner_simt_helper(x[0]) + 3.0f;" in updated
    assert "return v + 4.0f;" in updated

    for bad_search, bad_replace in (
        ("    return v * 2.0f;", "    return v * 5.0f;"),
        ("    y[0] = shared_simt_helper(y[0]);", "    y[0] = 0.0f;"),
    ):
        with pytest.raises(ValueError, match="read-only code outside active kernel"):
            apply_scoped_trackb_patch(
                _SCOPED_CPP, _hunk(bad_search, bad_replace), active_kernel="kernel_active"
            )

    with pytest.raises(ValueError, match="read-only sibling kernel marker"):
        apply_scoped_trackb_patch(
            _SCOPED_CPP,
            "// [READONLY_SIBLING_DECLARATION] void kernel_sibling();\n",
            active_kernel="kernel_active",
        )

    dup_cpp = (
        "#include <tl_templates/ascend/common.h>\n"
        'extern "C" __global__ __vector__ void sym_k(__gm__ float* x, int flag) {\n'
        "    if (flag > 0) {\n"
        "        float acc = x[0];\n"
        "        acc = acc * 2.0f;\n"
        "        x[0] = acc;\n"
        "    } else {\n"
        "        float acc = x[0];\n"
        "        acc = acc * 2.0f;\n"
        "        x[0] = acc;\n"
        "    }\n"
        "}\n"
    )
    branch = "        float acc = x[0];\n        acc = acc * 2.0f;\n        x[0] = acc;"
    with pytest.raises(AmbiguousPatchAnchorError) as ambig_exc:
        apply_scoped_trackb_patch(
            dup_cpp, _hunk(branch, "        x[0] = 1.0f;"), active_kernel="sym_k"
        )
    assert isinstance(ambig_exc.value, PatchAnchorMissError)
    assert ambig_exc.value.fault_code == "PATCH_AMBIGUOUS_ANCHOR"

    win_updated, win_count = apply_scoped_trackb_patch(
        dup_cpp,
        _hunk(
            branch,
            "        float acc = x[0];\n        acc = acc * 3.0f;\n        x[0] = acc;",
            window="L8-L10",
        ),
        active_kernel="sym_k",
    )
    assert win_count == 1
    assert win_updated.count("acc = acc * 2.0f;") == 1
    assert win_updated.count("acc = acc * 3.0f;") == 1

    with pytest.raises(PatchAnchorMissError, match="did not match"):
        apply_scoped_trackb_patch(_SCOPED_CPP, _hunk("int missing = 1;", "int missing = 2;"))

    with pytest.raises(NoopIdenticalPatchError) as noop_exc:
        apply_scoped_trackb_patch(
            _SCOPED_CPP,
            _hunk(
                "    x[1] = shared_simt_helper(x[1]);",
                "        x[1] =   shared_simt_helper(x[1]);",
            ),
            active_kernel="kernel_active",
        )
    assert noop_exc.value.fault_plane == "NOOP_IDENTICAL"
    assert noop_exc.value.fault_code == "NOOP_IDENTICAL"

    fenced_blocks = parse_scoped_search_replace_blocks(
        "```cpp\n<<<<<<< SEARCH\n    int step = 1;\n=======\n"
        "```c++\n    int step = 4;\n```\n>>>>>>> REPLACE\n```\n"
    )
    assert len(fenced_blocks) == 1 and "```" not in fenced_blocks[0][1]

    mode_a = (
        "__simt_callee__ inline float brand_new_helper(float v) { return v + 9.0f; }\n"
        'extern "C" __global__ __vector__ void kernel_active(__gm__ float* x) {\n'
        "    asc_vf_call<vf_worker>(dim3(64, 1, 1), x);\n"
        "    x[1] = brand_new_helper(x[1]);\n"
        "}\n"
    )
    spliced, sp_count = apply_scoped_trackb_patch(
        _SCOPED_CPP, mode_a, active_kernel="kernel_active"
    )
    assert sp_count == 2
    assert spliced.index("brand_new_helper") < spliced.index("kernel_active(__gm__ float* x)")
    assert "async_invoke<vf_sibling_only>(y);" in spliced

    with pytest.raises(ValueError, match="outside the editable scope"):
        splice_cpp_units_into_tu(
            _SCOPED_CPP,
            'extern "C" __global__ __vector__ void kernel_sibling(__gm__ float* y) '
            "{ y[0] = 0.0f; }\n",
            active_kernel="kernel_active",
        )


def test_regime_slice_patcher_cow_tail_guard_and_config_resolution() -> None:
    """Enforce CoW helper namespacing, sibling AST invariance, pruning, and tail guards."""
    spec_a, spec_b = _regime_specs()
    specs = (spec_a, spec_b)
    primary, snippet = extract_regime_snippet(
        _SHARED_HELPER_MODULE,
        spec_a,
        operator_entrypoint="kernel",
        sibling_specs=specs,
    )
    assert primary == "_build_regime_a"
    assert "def _stage_leaf(" in snippet
    assert "def _emit_pass1_compute(" in snippet
    assert "def _exclusive_a_helper(" in snippet
    assert "def _build_regime_b(" not in snippet
    assert "_KERNEL_CACHE" not in snippet

    sib_syms = (
        "_stage_leaf", "_emit_pass1_compute", "_select_regime_b_config", "_build_regime_b", "kernel"
    )
    sib_before = {sym: _ast_sha256(_SHARED_HELPER_MODULE, sym) for sym in sib_syms}
    co_edited = snippet.replace(
        "return _stage_leaf(scale) * 2", "return _stage_leaf(scale) * 4"
    ).replace("return x + 3", "return x + 9")
    patched = patch_regime_into_module(
        _SHARED_HELPER_MODULE,
        spec_a,
        co_edited,
        operator_entrypoint="kernel",
        sibling_specs=specs,
    )
    assert "def _regime_a_emit_pass1_compute(" in patched
    assert "return _stage_leaf(scale) * 4" in patched
    assert "return _stage_leaf(scale) * 2" in patched
    assert "return x + 9" in patched and "return x + 3" not in patched
    for sym, digest in sib_before.items():
        assert _ast_sha256(patched, sym) == digest

    pruned = patch_regime_into_module(
        patched,
        spec_a,
        "def _build_regime_a(tile: int = 1) -> object:\n    return None, tile + 10\n",
        operator_entrypoint="kernel",
        sibling_specs=specs,
    )
    assert "def _regime_a_emit_pass1_compute(" not in pruned
    for sym, digest in sib_before.items():
        assert _ast_sha256(pruned, sym) == digest

    sr_patched = patch_regime_into_module(
        _SHARED_HELPER_MODULE,
        spec_a,
        _hunk(
            "    step = _emit_pass1_compute(tile) + _exclusive_a_helper(1)",
            "    step = _emit_pass1_compute(tile) + _exclusive_a_helper(5)",
        ),
        operator_entrypoint="kernel",
        sibling_specs=specs,
    )
    assert "_exclusive_a_helper(5)" in sr_patched
    for sym, digest in sib_before.items():
        assert _ast_sha256(sr_patched, sym) == digest

    with pytest.raises(ValueError, match="read-only host tail"):
        patch_regime_into_module(
            _SHARED_HELPER_MODULE,
            spec_a,
            _hunk(
                "def kernel(x: int) -> object:",
                "def kernel(x: int) -> object:  # tampered",
            ),
            operator_entrypoint="kernel",
            sibling_specs=specs,
        )
    with pytest.raises(ValueError, match="protected symbol"):
        patch_regime_into_module(
            _SHARED_HELPER_MODULE,
            spec_a,
            _hunk(
                'def _select_regime_b_config(d_dim: int = 64) -> dict[str, object]:',
                'def _select_regime_b_config(d_dim: int = 128) -> dict[str, object]:',
            ),
            operator_entrypoint="kernel",
            sibling_specs=specs,
        )

    ast_patched = apply_ast_patch(
        _SHARED_HELPER_MODULE,
        "_exclusive_a_helper",
        "def _exclusive_a_helper(x: int) -> int:\n    return x + 11\n",
    )
    assert "return x + 11" in ast_patched
    resolved = resolve_regime_config(_SHARED_HELPER_MODULE, spec_a, dims={"D": 256})
    assert resolved.builder_name == "_build_regime_a"
    assert resolved.selector_name == "_select_regime_a_config"
    assert resolved.builder_kwargs == {"tile": 256}
    assert resolved.compile_flags == ("-O2",)
    assert resolved.pass_configs == {"tl.disable_auto_sync": True}


def test_trackb_module_patch_back_and_dead_edit_guards() -> None:
    """Patch Track-B C++ constants inside Python modules and block dead-code edits."""
    tb_mod = (
        '_OP_TRACKB_CU = r"""\n'
        "#include <tl_templates/ascend/common.h>\n"
        'extern "C" __global__ __vector__ void op_k(__gm__ float* x) {\n'
        "    int tile = 32;\n"
        "    x[0] = (float)tile;\n"
        "}\n"
        'extern "C" __global__ __vector__ void sib_k(__gm__ float* y) {\n'
        "    int keep = 7;\n"
        "    y[0] = (float)keep;\n"
        "}\n"
        '"""\n\n'
        "def _select_op_config():\n    return {'trackb_symbol': '_OP_TRACKB_CU'}\n\n"
        "def _render_trackb_tu(**dims):\n    return str(dims.get('trackb_src', ''))\n\n"
        "def kernel(x):\n    return x\n"
    )
    spec = RegimeTargetSpec(
        regime_tag="main",
        builder_candidates=("_select_op_config",),
        sample_builder_kwargs={},
        config_selector_fn="_select_op_config",
        editable_symbols=("_OP_TRACKB_CU", "_select_op_config"),
        runtime_backend="ascendc_trackb",
        trackb_symbols=("_OP_TRACKB_CU",),
    )
    patched = patch_regime_into_module(
        tb_mod,
        spec,
        "```cpp\n" + _hunk("    int tile = 32;", "    int tile = 64;") + "```",
        active_kernel="op_k",
    )
    assert "int tile = 64;" in patched
    assert "int keep = 7;" in patched
    assert "def kernel(x):" in patched

    with pytest.raises(PatchAnchorMissError, match="PATCH_ANCHOR_MISS:"):
        patch_regime_into_module(
            tb_mod,
            spec,
            _hunk("    int keep = 7;", "    int keep = 9;"),
            active_kernel="op_k",
        )
