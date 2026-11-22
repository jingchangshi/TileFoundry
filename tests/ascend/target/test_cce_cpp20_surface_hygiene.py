"""Static and behavioral checks for the Track-B CCE (C++20) dialect surface."""

from __future__ import annotations

import pytest

from tilefoundry_ascend.agent.kb.schema import (
    AscendCausalRule,
    default_seed_rules_path,
    load_seed_rules,
    normalize_runtime_backend,
)
from tilefoundry_ascend.agent.orchestrator.cpp_slice_protocol import apply_scoped_trackb_patch
from tilefoundry_ascend.agent.orchestrator.gate_funnel import _check_custom_kernel_redlines
from tilefoundry_ascend.agent.orchestrator.llm_constraints import format_target_dialect_layer
from tilefoundry_ascend.agent.orchestrator.portfolio_runner import build_syntax_repair_request
from tilefoundry_ascend.ascend.target.dialect_surface import (
    CCE_CPP20_SURFACE,
    INLINE_ASCENDC_VF_SURFACE,
    INLINE_CCE_VF_SURFACE,
    TILELANG_DSL_SURFACE,
    TL_TEMPLATES_CPP20_SURFACE,
    get_dialect_surface,
    resolve_trackb_cce_surface,
)

TRACKB_ALIASES = (
    "cce_cpp20",
    "cce_trackb",
    "hybrid_trackb",
    "tl_templates_cpp20",
    "ascendc_trackb",
    "ascendc_template",
)

FORBIDDEN_CCE_TOKENS = (
    "kernel_operator.h",
    "using namespace AscendC;",
    "AscendC::",
    "TPipe",
    "TQue",
    "DataCopyPad",
    "FixpipeParamsArch3510",
    "copy_gm_to_ubuf",
    "copy_ubuf_to_gm",
    "set_flag(",
    "wait_flag(",
    "__shfl_down_sync",
    "__shfl_sync",
    "__shfl_xor_sync",
    "__ballot_sync",
)

CLEAN_TRACKB_SOURCE = (
    "#include <tl_templates/ascend/common.h>\n"
    'extern "C" __global__ void kernel_active() {\n'
    "    asc_copy_gm2ub_align(ub, gm, 1, 128);\n"
    "}\n"
)


def _positive_ascendc_lines(text: str) -> list[str]:
    """Collect lines that mention AscendC outside an explicit prohibition."""
    offenders: list[str] = []
    for line in text.splitlines():
        if "AscendC" not in line:
            continue
        if "Strictly Forbidden" in line or "Do NOT emit" in line:
            continue
        offenders.append(line)
    return offenders


def _patch(search: str, replace: str) -> str:
    """Render one SEARCH/REPLACE hunk in the scoped Track-B protocol format."""
    return f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


def test_cce_cpp20_surface_freezes_label_headers_and_two_sided_blacklist() -> None:
    """Freeze the canonical Track-B CCE (C++20) surface contract."""
    assert CCE_CPP20_SURFACE.dialect_id == "cce_cpp20"
    assert CCE_CPP20_SURFACE.display_label == "CCE C++20 (tl_templates/ascend)"
    assert CCE_CPP20_SURFACE.required_headers_or_imports == (
        "#include <tl_templates/ascend/common.h>",
        "#include <tl_templates/ascend/gemm.h>",
        "#include <tl_templates/ascend/simd_inst.h>",
    )
    assert CCE_CPP20_SURFACE.forbidden_tokens == FORBIDDEN_CCE_TOKENS
    assert set(FORBIDDEN_CCE_TOKENS) <= set(CCE_CPP20_SURFACE.forbidden_tokens)
    rendered = CCE_CPP20_SURFACE.render_prompt_reference()
    assert rendered.splitlines()[0] == (
        "Verified `CCE C++20 (tl_templates/ascend)` (`cce_cpp20`) callable surface:"
    )
    assert "Strictly Forbidden Symbols:" in rendered
    for token in FORBIDDEN_CCE_TOKENS:
        assert f"`{token}`" in rendered
    assert "Do NOT emit high-level AscendC classes" in rendered
    assert TL_TEMPLATES_CPP20_SURFACE is CCE_CPP20_SURFACE


@pytest.mark.parametrize("alias", TRACKB_ALIASES)
def test_trackb_new_and_legacy_aliases_share_the_cce_surface(alias: str) -> None:
    """Resolve every Track-B spelling to the frozen CCE C++20 contract."""
    assert resolve_trackb_cce_surface(alias) is CCE_CPP20_SURFACE
    assert get_dialect_surface(alias) is CCE_CPP20_SURFACE


def test_resolve_trackb_cce_surface_rejects_non_trackb_and_defaults_cleanly() -> None:
    """Keep a strict Track-B resolver while accepting an empty identifier as CCE."""
    assert resolve_trackb_cce_surface("") is CCE_CPP20_SURFACE
    with pytest.raises(ValueError, match="Unsupported Track-B CCE dialect identifier"):
        resolve_trackb_cce_surface("tilelang_dsl")


def test_inline_surface_renames_dialect_but_keeps_legacy_symbol_and_alias() -> None:
    """Expose the inline CCE VF dialect under its canonical id and legacy aliases."""
    assert INLINE_CCE_VF_SURFACE.dialect_id == "inline_cce_vf"
    assert INLINE_ASCENDC_VF_SURFACE is INLINE_CCE_VF_SURFACE
    assert get_dialect_surface("inline_cce_vf") is INLINE_CCE_VF_SURFACE
    assert get_dialect_surface("inline_ascendc_vf") is INLINE_CCE_VF_SURFACE
    assert get_dialect_surface("tilelang_dsl") is TILELANG_DSL_SURFACE
    assert get_dialect_surface("python_kernel") is TILELANG_DSL_SURFACE


@pytest.mark.parametrize("tier", TRACKB_ALIASES)
def test_trackb_layer2_prompt_has_zero_positive_ascendc_induction(tier: str) -> None:
    """Assert Layer-2 Track-B generation prompts only mention AscendC as prohibited."""
    prompt = format_target_dialect_layer(escalation_tier=tier)
    assert "CCE C++20 (tl_templates/ascend)" in prompt
    assert "Verified `CCE C++20 (tl_templates/ascend)` (`cce_cpp20`)" in prompt
    assert "asc_copy_gm2ub_align" in prompt
    assert "T.Kernel" not in prompt
    assert _positive_ascendc_lines(prompt) == []


def test_trackb_repair_prompt_is_cce_scoped_and_two_sided_blocked() -> None:
    """Route Track-B repairs to C++20 and forbid both AscendC generations."""
    request = build_syntax_repair_request(
        "__global__ void kernel_active() {}",
        "compile error",
        backend_kind="ascendc_trackb",
    )
    assert request.role == "syntax_repair"
    assert "Track-B CCE (C++20)" in request.system_prompt
    assert "Do NOT emit high-level AscendC classes" in request.system_prompt
    assert "```cpp" in request.user_prompt
    assert "CCE C++20 (tl_templates/ascend)" in request.user_prompt
    assert "asc_copy_gm2ub_align" in request.user_prompt
    assert "T.Kernel" not in request.user_prompt
    combined = f"{request.system_prompt}\n{request.user_prompt}"
    assert _positive_ascendc_lines(combined) == []
    legacy = build_syntax_repair_request(
        "__global__ void kernel_active() {}", "compile error", backend_kind="cce_trackb"
    )
    assert "```cpp" in legacy.user_prompt
    assert "CCE C++20 (tl_templates/ascend)" in legacy.user_prompt


def test_tilelang_repair_prompt_keeps_python_dialect() -> None:
    """Leave Track-A repair prompts on the TileLang surface and Python fence."""
    request = build_syntax_repair_request("def f():\n    pass\n", "compile error")
    assert "```python" in request.user_prompt
    assert "T.MixedKernel" in request.user_prompt
    assert "CCE C++20 (tl_templates/ascend)" not in request.user_prompt


def test_tilelang_surface_registers_simd_register_primitives_and_parallel_math_ban() -> None:
    """Expose every verified T.simd intrinsic and steer scalar math out of plain T.Parallel."""
    entries = {entry.symbol: entry for entry in TILELANG_DSL_SURFACE.primitives}
    expected_signatures = {
        "T.simd.pset": 'T.simd.pset(width, "PAT_ALL")',
        "T.simd.update_mask": "T.simd.update_mask(count, width=32)",
        "T.simd.vdup": "T.simd.vdup(value, dtype, mask=mask)",
        "T.simd.vld": 'T.simd.vld(buf[offset], dist="NORM_B32")',
        "T.simd.vcvt": "T.simd.vcvt(vec, dtype, mask=mask, part=0)",
        "T.simd.vexp": "T.simd.vexp(vec, mask=mask)",
        "T.simd.vadd": "T.simd.vadd(lhs, rhs, mask=mask)",
        "T.simd.vsub": "T.simd.vsub(lhs, rhs, mask=mask)",
        "T.simd.vmul": "T.simd.vmul(lhs, rhs, mask=mask)",
        "T.simd.vmuls": "T.simd.vmuls(vec, scalar, mask=mask)",
        "T.simd.vdiv": "T.simd.vdiv(lhs, rhs, mask=mask)",
        "T.simd.vsts": 'T.simd.vsts(buf[offset], vec, mask=mask, dist="PK_B32", extent=64)',
        "T.annotate_buffer_versions": "T.annotate_buffer_versions({buf: stages})",
        "T.annotate_unlimit_memory": 'T.annotate_unlimit_memory("shared")',
    }
    for symbol, signature in expected_signatures.items():
        assert symbol in entries, symbol
        assert entries[symbol].signature == signature
        assert entries[symbol].pipe == "PIPE_V" or symbol.startswith("T.annotate")
    symbols = [entry.symbol for entry in TILELANG_DSL_SURFACE.primitives]
    assert len(symbols) == len(set(symbols))

    rendered = TILELANG_DSL_SURFACE.render_prompt_reference()
    ban = (
        "Inside T.Parallel never call T.Cast, T.exp, T.log, T.tanh, T.sigmoid, or T.rsqrt "
        "unless the loop is nested under `with T.SimtVF(threads=128):`"
    )
    assert ban in rendered
    assert "T.simd.vexp(vec, mask=mask)" in rendered
    assert "T.simd.vcvt(vec, dtype, mask=mask, part=0)" in rendered
    assert 'vcvt("float32")' in rendered
    assert "US_B16" in rendered and "PK_B32" in rendered
    assert "`with T.SimtVF(threads=128):` with lowercase T.cast" in rendered


def test_cce_cpp20_zero_stack_array_note_is_frozen() -> None:
    """Pin the zero-byte __simd_vf__ stack frame guidance on the CCE surface."""
    rendered = CCE_CPP20_SURFACE.render_prompt_reference()
    assert (
        "The __simd_vf__ stack frame is exactly 0 bytes: never declare local C arrays "
        "(for example float tmp[64];) or runtime-indexed local structs inside a "
        "__simd_vf__ body; stage scratch in __ubuf__ pointer ranges or vector registers."
    ) in rendered


def test_custom_kernel_redlines_flag_local_arrays_only_inside_simdvf_bodies() -> None:
    """Flag scalar local arrays in __simd_vf__ bodies but not comments, strings, or registers."""
    expected = (
        "simdvf_nonzero_stack_array: local array inside __simd_vf__ violates vf stack "
        "size (0); allocate in __ubuf__ workspace instead"
    )
    flagged = (
        '__simd_vf__ inline void vf(__ubuf__ float* x) {\n  float tmp[64];\n}\n',
        '__simd_vf__ inline void vf(__ubuf__ float* x) {\n  uint16_t idx[8];\n}\n',
    )
    for source in flagged:
        assert _check_custom_kernel_redlines(source) == [expected]
    allowed = (
        '// float tmp[64];\n__simd_vf__ inline void vf(__ubuf__ float* x) {\n  x[0] = 1.0f;\n}\n',
        'const char* s = "float tmp[64];";\n'
        '__simd_vf__ inline void vf(__ubuf__ float* x) {\n  x[0] = 1.0f;\n}\n',
        '__simd_vf__ inline void vf(__ubuf__ float* x) {\n  vector_f32 rg0[4];\n}\n',
        'float tmp[64];\nextern "C" __global__ void k() {}\n',
        '// __simd_vf__ inline void vf() { float tmp[4]; }\nextern "C" __global__ void k() {}\n',
    )
    for source in allowed:
        assert _check_custom_kernel_redlines(source) == []


def test_kb_legacy_aliases_normalize_and_render_cce_guidance() -> None:
    """Keep checkpoint-compatible backend ids while rendering Track-B CCE guidance."""
    assert normalize_runtime_backend("cce_trackb") == "ascendc_trackb"
    assert normalize_runtime_backend("cce_cpp20") == "ascendc_trackb"
    assert normalize_runtime_backend("tl_templates_cpp20") == "ascendc_trackb"
    assert normalize_runtime_backend("ascendc_template") == "ascendc_trackb"
    rules = {rule.rule_id: rule for rule in load_seed_rules()}
    r02 = rules["R02_ONCHIP_L0C_TO_UB_DIRECT"]
    trackb = r02.render_for_backend("cce_trackb")
    assert "asc_copy_l0c2ub" in trackb.transform_template
    assert "T.dual_copy" not in trackb.transform_template
    hybrid = r02.render_for_backend("hybrid_trackb")
    assert "T.dual_copy" in hybrid.transform_template
    assert "asc_copy_l0c2ub" in hybrid.transform_template
    assert "Track-B CCE (C++20):" in hybrid.transform_template


def test_rule_aliases_round_trip_through_legacy_canonical_ids() -> None:
    """Deserialize new Track-B names into the legacy canonical id used by checkpoints."""
    rule = AscendCausalRule.from_dict(
        {
            "rule_id": "R_ALIAS_ROUNDTRIP",
            "regime_guard": "is_legal",
            "precondition": "PIPE_V > 0",
            "transform_template": "Optimize the vector stage.",
            "expected_postcondition": "PIPE_V <= prev.PIPE_V",
            "source_provenance": "flywheel:test",
            "compatible_backends": ["cce_cpp20", "hybrid_trackb"],
        }
    )
    assert rule.compatible_backends == ("ascendc_trackb", "hybrid_trackb")
    assert rule.supports_backend("cce_trackb")
    assert rule.supports_backend("ascendc_template")


def test_seed_rules_prompts_avoid_forbidden_cce_symbols() -> None:
    """Scan every shipped seed rule for positive high-level or legacy CCE symbols."""
    text = default_seed_rules_path().read_text(encoding="utf-8")
    for token in FORBIDDEN_CCE_TOKENS:
        assert token not in text
    assert "AscendC" not in text
    assert "Track-B C++20" not in text
    assert '"cce_trackb"' in text


def test_forbidden_token_counts_respect_cpp_token_boundaries() -> None:
    """Count only bare tokens, not namespace-qualified, member, or suffixed lookalikes."""
    legal = (
        "simd_inst::set_flag(PIPE_V, PIPE_MTE3, 0);\n"
        "asc_set_flag(PIPE_V, PIPE_MTE3, 0);\n"
        "obj.set_flag(PIPE_V, PIPE_MTE3, 0);\n"
        "ptr->set_flag(PIPE_V, PIPE_MTE3, 0);\n"
        "my_kernel_operator.h\n"
        "myTPipe\n"
        "my_TQue\n"
        "DataCopyPadV2\n"
        "my_copy_gm_to_ubuf\n"
    )
    counts = CCE_CPP20_SURFACE.count_forbidden_tokens(legal)
    for token in (
        "set_flag(",
        "kernel_operator.h",
        "TPipe",
        "TQue",
        "DataCopyPad",
        "copy_gm_to_ubuf",
    ):
        assert counts[token] == 0
    bare = (
        "set_flag(PIPE_V, PIPE_MTE3, 0);\n"
        '#include "kernel_operator.h"\n'
        "TPipe pipe;\n"
        "TQue<TPosition::VECIN, 1> que;\n"
        "DataCopyPad(ub, gm, 128);\n"
        "copy_gm_to_ubuf(ub, gm, 1, 128);\n"
    )
    bare_counts = CCE_CPP20_SURFACE.count_forbidden_tokens(bare)
    for token in (
        "set_flag(",
        "kernel_operator.h",
        "TPipe",
        "TQue",
        "DataCopyPad",
        "copy_gm_to_ubuf",
    ):
        assert bare_counts[token] == 1


def test_introduced_forbidden_tokens_compare_counts_not_presence() -> None:
    """Reject a preserved token growing from one to two uses, not legal namespace additions."""
    before = "set_flag(PIPE_V, PIPE_MTE3, 0);\n"
    grown = before + "set_flag(PIPE_MTE3, PIPE_V, 0);\n"
    assert CCE_CPP20_SURFACE.introduced_forbidden_tokens(before, before) == ()
    assert CCE_CPP20_SURFACE.introduced_forbidden_tokens(before, grown) == ("set_flag(",)
    legal = before + "simd_inst::set_flag(PIPE_V, PIPE_MTE3, 0);\n"
    assert CCE_CPP20_SURFACE.introduced_forbidden_tokens(before, legal) == ()


def test_trackb_scoped_patch_rejects_duplicate_growth_of_preserved_token() -> None:
    """Reject a hunk that keeps one bare legacy call and adds a second identical intrinsic."""
    source = (
        'extern "C" __global__ void kernel_active() {\n'
        "    set_flag(PIPE_V, PIPE_MTE3, 0);\n"
        "}\n"
    )
    grown = _patch(
        "    set_flag(PIPE_V, PIPE_MTE3, 0);",
        "    set_flag(PIPE_V, PIPE_MTE3, 0);\n    set_flag(PIPE_MTE3, PIPE_V, 0);",
    )
    with pytest.raises(ValueError, match=r"forbidden symbols: set_flag\("):
        apply_scoped_trackb_patch(source, grown, active_kernel="kernel_active")
    legal = _patch(
        "    set_flag(PIPE_V, PIPE_MTE3, 0);",
        "    set_flag(PIPE_V, PIPE_MTE3, 0);\n    simd_inst::set_flag(PIPE_V, PIPE_MTE3, 0);",
    )
    updated, applied = apply_scoped_trackb_patch(source, legal, active_kernel="kernel_active")
    assert applied == 1
    assert "simd_inst::set_flag" in updated


def test_trackb_scoped_patch_rejects_newly_introduced_forbidden_symbols() -> None:
    """Reject patches that newly write high-level AscendC or legacy 910B intrinsics."""
    high_level = _patch(
        "    asc_copy_gm2ub_align(ub, gm, 1, 128);",
        "    AscendC::DataCopyPad(ub, gm, 128);",
    )
    with pytest.raises(ValueError, match="introduces forbidden symbols: AscendC::"):
        apply_scoped_trackb_patch(CLEAN_TRACKB_SOURCE, high_level, active_kernel="kernel_active")

    legacy = _patch(
        "    asc_copy_gm2ub_align(ub, gm, 1, 128);",
        "    copy_gm_to_ubuf(ub, gm, 1, 128);",
    )
    with pytest.raises(ValueError, match="copy_gm_to_ubuf"):
        apply_scoped_trackb_patch(CLEAN_TRACKB_SOURCE, legacy, active_kernel="kernel_active")


def test_trackb_scoped_patch_allows_preserved_or_clean_replacements() -> None:
    """Allow clean edits and faithful rewrites that keep pre-existing legacy symbols."""
    clean = _patch(
        "    asc_copy_gm2ub_align(ub, gm, 1, 128);",
        "    asc_copy_gm2ub_align(ub, gm, 2, 256);",
    )
    updated, applied = apply_scoped_trackb_patch(
        CLEAN_TRACKB_SOURCE, clean, active_kernel="kernel_active"
    )
    assert applied == 1
    assert "asc_copy_gm2ub_align(ub, gm, 2, 256);" in updated

    legacy_source = (
        'extern "C" __global__ void kernel_active() {\n'
        "    using namespace AscendC;\n"
        "    int x = 1;\n"
        "}\n"
    )
    preserved = _patch(
        "    using namespace AscendC;\n    int x = 1;",
        "    using namespace AscendC;\n    int x = 2;",
    )
    patched, applied_legacy = apply_scoped_trackb_patch(
        legacy_source, preserved, active_kernel="kernel_active"
    )
    assert applied_legacy == 1
    assert "int x = 2;" in patched
