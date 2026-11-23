"""On-demand Layer-2 prompt pruning and lowering guardrails tests for CANN 9.3."""

from __future__ import annotations

from tilefoundry_ascend.agent.orchestrator.llm_constraints import (
    build_progressive_llm_constraints,
    format_target_dialect_layer,
)
from tilefoundry_ascend.ascend.sil.schema import (
    CasePhysicsRatios,
    ConstraintRelaxationMask,
    TopologyAxes,
)
from tilefoundry_ascend.ascend.target.dialect_surface import (
    LOWERING_GUARDRAILS,
    TILELANG_DSL_SURFACE,
    filter_primitives_on_demand,
    load_cann_surface,
    render_layer2_surface,
)
from tilefoundry_ascend.ascend.templates.registry import (
    CANONICAL_TOPOLOGY_PRESETS,
    SHELL_AIV_SIMD_STREAM,
    SHELL_CV_RING_PIPELINE,
)

_SUBATOM_RATIOS = CasePhysicsRatios(
    case_id=1,
    stride_ratio=1.0,
    contig_atom_ratio=0.5,
    is_atom_aligned=False,
    working_set_sram_ratio=0.1,
    working_set_vreg_ratio=0.1,
    wave_occupancy_ratio=1.0,
    oi_ridge_ratio=0.1,
    inner_contiguous_bytes=16,
)


def test_aiv_only_excludes_cube_and_fixpipe_on_cann93() -> None:
    """Exclude Cube, FixPipe, L1/L0, and coupled_1c2v primitives for aiv_only."""
    cce_93 = load_cann_surface("cann_9_3")
    tl_filtered = filter_primitives_on_demand(
        TILELANG_DSL_SURFACE,
        topology_axes=SHELL_AIV_SIMD_STREAM,
    )
    cce_filtered = filter_primitives_on_demand(
        cce_93,
        topology_axes=SHELL_AIV_SIMD_STREAM,
    )
    tl_syms = {entry.symbol for entry in tl_filtered}
    cce_syms = {entry.symbol for entry in cce_filtered}
    assert "T.gemm" not in tl_syms
    assert "T.dual_copy" not in tl_syms
    assert "T.alloc_l1" not in tl_syms
    assert "ascend_gemm_l1" not in cce_syms
    assert "asc_copy_l0c2ub" not in cce_syms
    assert "asc_copy_l0c2gm" not in cce_syms
    assert "asc_mmad" not in cce_syms

    layer2 = format_target_dialect_layer(
        escalation_tier="cce_cpp20",
        topology_axes=SHELL_AIV_SIMD_STREAM,
        cann_series="cann_9_3",
    )
    assert "ascend_gemm_l1" not in layer2
    assert "asc_copy_l0c2ub" not in layer2


def test_bottom_mask_retains_non_empty_core_matching_topology_axes() -> None:
    """Keep a non-empty engine-matched core at ConstraintRelaxationMask.bottom()."""
    cce_93 = load_cann_surface("cann_9_3")
    bottom = ConstraintRelaxationMask.bottom()
    top = ConstraintRelaxationMask.top()
    assert bottom.unlocked_facet_names() == bottom.unlocked_facets()

    for axes in CANONICAL_TOPOLOGY_PRESETS:
        tl_bot = filter_primitives_on_demand(
            TILELANG_DSL_SURFACE,
            topology_axes=axes,
            freedom_mask=bottom,
        )
        cce_bot = filter_primitives_on_demand(
            cce_93,
            topology_axes=axes,
            freedom_mask=bottom,
        )
        tl_top = filter_primitives_on_demand(
            TILELANG_DSL_SURFACE,
            topology_axes=axes,
            freedom_mask=top,
        )
        cce_top = filter_primitives_on_demand(
            cce_93,
            topology_axes=axes,
            freedom_mask=top,
        )
        assert len(tl_bot) > 0
        assert len(cce_bot) > 0
        assert {e.symbol for e in tl_bot} <= {e.symbol for e in tl_top}
        assert {e.symbol for e in cce_bot} <= {e.symbol for e in cce_top}
        if axes.compute_engine == "aic_only":
            assert "T.gemm" in {e.symbol for e in tl_bot}
            assert "ascend_gemm_l1" in {e.symbol for e in cce_bot}
        elif axes.compute_engine == "coupled_1c2v":
            assert "T.MixedKernel" in {e.symbol for e in tl_bot}
            assert "asc_copy_l0c2ub" in {e.symbol for e in cce_bot}
        else:
            assert "T.copy" in {e.symbol for e in tl_bot}
            assert "asc_copy_gm2ub_align" in {e.symbol for e in cce_bot}


def test_top_mask_and_subatom_retain_simtvf_and_simt_markers() -> None:
    """Retain T.SimtVF and __simt_vf__ for both top() and _subatom ratios."""
    cce_93 = load_cann_surface("cann_9_3")
    bottom = ConstraintRelaxationMask.bottom()
    top = ConstraintRelaxationMask.top()

    tl_bot_affine = filter_primitives_on_demand(
        TILELANG_DSL_SURFACE,
        topology_axes=SHELL_AIV_SIMD_STREAM,
        freedom_mask=bottom,
    )
    assert "T.SimtVF" not in {e.symbol for e in tl_bot_affine}

    tl_top = filter_primitives_on_demand(
        TILELANG_DSL_SURFACE,
        topology_axes=SHELL_AIV_SIMD_STREAM,
        freedom_mask=top,
    )
    assert "T.SimtVF" in {e.symbol for e in tl_top}

    tl_subatom = filter_primitives_on_demand(
        TILELANG_DSL_SURFACE,
        topology_axes=SHELL_AIV_SIMD_STREAM,
        freedom_mask=bottom,
        case_ratios=_SUBATOM_RATIOS,
    )
    assert "T.SimtVF" in {e.symbol for e in tl_subatom}

    for mask, ratios in ((top, None), (bottom, _SUBATOM_RATIOS)):
        cce_filtered = filter_primitives_on_demand(
            cce_93,
            topology_axes=SHELL_AIV_SIMD_STREAM,
            freedom_mask=mask,
            case_ratios=ratios,
        )
        assert "asc_vf_call" in {e.symbol for e in cce_filtered}
        rendered = render_layer2_surface(
            cce_93,
            primitives=cce_filtered,
            include_guardrails=True,
            max_chars=10000,
        )
        assert "__simt_vf__" in rendered


def test_coupled_1c2v_trackb_retains_exact_l0c2ub_and_gemm_signatures() -> None:
    """Preserve exact 15-arg asc_copy_l0c2ub and 6-template-arg ascend_gemm_l1 signatures."""
    prompt = format_target_dialect_layer(
        escalation_tier="cce_cpp20",
        topology_axes=SHELL_CV_RING_PIPELINE,
        freedom_mask=ConstraintRelaxationMask.bottom(),
        cann_series="cann_9_3",
    )
    expected_l0c2ub = (
        "asc_copy_l0c2ub(__ubuf__ float* dst, __cc__ float* src, uint16_t n_size, "
        "uint16_t m_size, uint32_t dst_stride, uint16_t src_stride, int8_t sub_blockid, "
        "asc_dual_dst_mode dual_dst_ctrl, asc_unit_flag_mode unit_flag_mode, "
        "asc_quant_mode quant_pre_mode, asc_relu_pre_mode relu_pre_mode, "
        "bool enable_channel_split, bool enable_nz2nd, bool enable_nz2dn, "
        "bool enable_clip_relu_pre)"
    )
    expected_gemm = (
        "template <int M, int K, int N, int TILE_K_SUB, bool TRANS_B, typename InT, "
        "typename AccumT> __aicore__ inline void ascend_gemm_l1(__cc__ AccumT* cc_ptr, "
        "__cbuf__ InT* cbuf_a_ptr, __cbuf__ InT* cbuf_b_ptr, int clear_accum, "
        "int buf_offset = 0, int unit_flag_ctrl = 0)"
    )
    assert expected_l0c2ub in prompt
    assert expected_gemm in prompt
    for line in prompt.splitlines():
        if line.startswith("- `") and ("asc_copy_l0c2ub" in line or "ascend_gemm_l1" in line):
            assert "..." not in line


def test_all_four_lowering_guardrails_rendered_in_layer2() -> None:
    """Render all 4 lowering guardrails in both TileLang and Track-B prompts."""
    assert len(LOWERING_GUARDRAILS) == 4
    for tier in ("tilelang_dsl", "cce_cpp20"):
        layer2 = format_target_dialect_layer(
            escalation_tier=tier,
            topology_axes=SHELL_AIV_SIMD_STREAM,
            cann_series="cann_9_3",
        )
        for guardrail in LOWERING_GUARDRAILS:
            assert guardrail in layer2
        assert "tirx.all" in layer2
        assert "int8x2" in layer2
        assert "bfloat16_t" in layer2
        assert "BufferStore" in layer2


def test_all_seven_presets_budget_and_45_percent_reduction_on_cann93() -> None:
    """Verify <=10000 Layer-2, <=18000 full prompt, and >=45% reduction on cann_9_3."""
    cce_93 = load_cann_surface("cann_9_3")
    monolithic_len = len(cce_93.render_prompt_reference())
    assert len(CANONICAL_TOPOLOGY_PRESETS) == 7

    for axes in CANONICAL_TOPOLOGY_PRESETS:
        assert isinstance(axes, TopologyAxes)
        layer2 = format_target_dialect_layer(
            escalation_tier="cce_cpp20",
            topology_axes=axes,
            cann_series="cann_9_3",
        )
        full_prompt = build_progressive_llm_constraints(
            escalation_tier="cce_cpp20",
            topology_axes=axes,
            cann_series="cann_9_3",
        )
        assert len(layer2) <= 10000
        assert len(full_prompt) <= 18000
        reduction = 1.0 - (len(layer2) / float(monolithic_len))
        assert reduction >= 0.45

    sym_filtered = filter_primitives_on_demand(
        cce_93,
        topology_axes=SHELL_AIV_SIMD_STREAM,
        active_symptom="DMA_UNALIGNED_SHORT_BURST",
    )
    assert len(sym_filtered) > 0
    assert "asc_copy_gm2ub_align" in {e.symbol for e in sym_filtered}
    fallback_filtered = filter_primitives_on_demand(
        cce_93,
        topology_axes=SHELL_AIV_SIMD_STREAM,
        active_symptom="UNKNOWN_SYMPTOM",
    )
    assert len(fallback_filtered) == len(
        filter_primitives_on_demand(cce_93, topology_axes=SHELL_AIV_SIMD_STREAM)
    )
