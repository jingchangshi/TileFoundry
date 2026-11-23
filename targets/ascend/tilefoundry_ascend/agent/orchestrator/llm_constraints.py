"""Unified 3-layer progressive LLM generation constraint synthesizer for Ascend 950."""

from __future__ import annotations

import json
from collections.abc import Sequence

from tilefoundry_ascend.agent.router.isa_gap_scanner import (
    PipeGapReport,
    scan_tilelang_pipe_gaps,
)
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import HirBlueprintPacket
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    ObjectivePhysicalLedger,
    ScopeWaterline,
)
from tilefoundry_ascend.ascend.sil.schema import (
    VALID_FREEDOM_FACETS,
    AdvisoryEnvelope,
    CasePhysicsRatios,
    ConstraintRelaxationMask,
    HardwareLegalityBounds,
    TopologyAxes,
)
from tilefoundry_ascend.ascend.target.dialect_surface import (
    filter_primitives_on_demand,
    get_dialect_surface,
    render_layer2_surface,
)

_SCOPE_TOPOLOGY_MAP: dict[str, str] = {
    "UB": "smem (AIV)",
    "L1": "tmem (AIC)",
    "L0A": "tmem (AIC)",
    "L0B": "tmem (AIC)",
    "L0C": "tmem (AIC)",
    "SSBUF": "ssbuf (block)",
}


def _ub_peak_bytes(ledger: ObjectivePhysicalLedger) -> int:
    """Return peak UB bytes recorded in *ledger*."""
    ub = ledger.waterlines.get("UB")
    return ub.peak_allocated_bytes if ub is not None else 0


def _format_scope_pct(waterline: ScopeWaterline | None) -> str:
    """Format scope utilization percentage or 'n/a' when absent."""
    if waterline is None or waterline.capacity_bytes <= 0:
        return "n/a"
    pct = 100.0 * float(waterline.peak_allocated_bytes) / float(waterline.capacity_bytes)
    return f"{pct:.2f}%"


def _format_regime_ratios(sil_advisory: object | None) -> list[str]:
    """Render each regime's representative real CasePhysicsRatios for prompt grounding."""
    if not isinstance(sil_advisory, AdvisoryEnvelope):
        return []
    lines: list[str] = []
    for regime in sil_advisory.regimes:
        ratios = regime.representative_ratios()
        if ratios is None:
            lines.append(
                f"- Regime {regime.regime_tag} cases={list(regime.case_nums)} "
                "CasePhysicsRatios: unavailable"
            )
            continue
        lines.append(
            f"- Regime {regime.regime_tag} cases={list(regime.case_nums)} "
            f"representative_case={ratios.case_id} "
            f"CasePhysicsRatios: {json.dumps(ratios.to_dict(), sort_keys=True)}"
        )
    return lines


def format_hardware_and_sil_layer(
    *,
    sil_advisory: object | None = None,
    sil_verification: object | None = None,
    freedom_mask: ConstraintRelaxationMask | None = None,
    vector_mode: str = "simdvf",
    ub_256kb_unlock: bool = False,
) -> str:
    """Render Layer 1: hardware legality redlines and active 7-facet MECE shell locks."""
    bounds = getattr(sil_advisory, "legality_bounds", None)
    if not isinstance(bounds, HardwareLegalityBounds):
        bounds = HardwareLegalityBounds()

    vmode = str(vector_mode or "simdvf").lower()
    if vmode in ("simtvf", "hybrid"):
        ub_cap = bounds.ub_simtvf_bytes
    elif ub_256kb_unlock:
        ub_cap = bounds.ub_unreserved_bytes
    else:
        ub_cap = bounds.ub_simdvf_bytes

    lines = [
        "### LAYER 1: HARDWARE REDLINES & 7-FACET SHELL CONTRACT",
        (
            f"- Scratchpad Ceilings: UB<={ub_cap} B (mode={vmode}), "
            f"L1<={bounds.l1_bytes} B (half-bank ping-pong<={bounds.l1_half_bank_bytes} B), "
            f"L0A<={bounds.l0a_bytes} B, L0B<={bounds.l0b_bytes} B, "
            f"L0C<={bounds.l0c_bytes} B, SSBUF<={bounds.ssbuf_bytes} B"
        ),
        (
            f"- Hard Invariants: forbidden_dma={list(bounds.forbidden_dma_paths)}, "
            f"max_mmad_mnk<={bounds.max_mmad_mnk}, "
            f"crosscore_flag_depth<={bounds.max_crosscore_depth}, "
            f"forbid_cube_int4={bounds.forbid_cube_int4}"
        ),
        "- Schedule Integrity Redlines (paired canonical escape paths):",
        (
            "  * G1 (No Scalar Compute Emulation): when vector/matrix semantics exist, never "
            "emulate main compute with scalar C++ `for` loops or bit-level emulation -> "
            "canonical path: `T.SimdVF` / `simd_inst::*` / `asc_mmad`."
        ),
        (
            "  * G2 (No Dense-Loop Narrow DCache-Bypass Stores): never emit elementwise `< 32B` "
            "GM stores triggering `write_gm_bypass_dcache` inside dense loops -> canonical path: "
            "pack multiple `< 32B` rows into contiguous `>= 32B` atomic bursts."
        ),
        (
            "  * G3 (No Cache-Key Pollution or Static Unrolled 2D Dynamic Copy): never put "
            "dynamic batch/seq dims or float hyperparams in `_KERNEL_CACHE` keys and never split "
            "2D dynamic `T.copy` with Python static loops -> canonical path: `T.dynamic` + 2D "
            "`T.copy` + `T.float32` scalar params."
        ),
        (
            "  * G4 (No Overlapping `coupled_1c2v` Dual-Subcore Writes): under `coupled_1c2v`, "
            "never issue overlapping `MTE3` writes from `AIV0`/`AIV1` to the same GM slice -> "
            "canonical path: disjointly bisect output slices by subcore index."
        ),
    ]

    active_mask = freedom_mask or ConstraintRelaxationMask.bottom()
    if active_mask.is_unrestricted_top:
        lines.append("- 7-Facet MECE Freedom Lattice: UNRESTRICTED TOP (all 7 facets unlocked)")
    else:
        unlocked = [f for f in VALID_FREEDOM_FACETS if getattr(active_mask, f, False)]
        locked = [f for f in VALID_FREEDOM_FACETS if not getattr(active_mask, f, False)]
        lines.append(f"- Unlocked Facets ({len(unlocked)}): {', '.join(unlocked)}")
        lines.append(f"- Locked Shell Facets ({len(locked)}): {', '.join(locked)}")

    if sil_advisory is not None:
        axes = getattr(sil_advisory, "topology_axes", None)
        axes_dict = axes.to_dict() if hasattr(axes, "to_dict") else axes
        roofline = getattr(sil_advisory, "roofline", None) or {}
        lines.append(f"- TopologyAxes: {json.dumps(axes_dict, sort_keys=True)}")
        if roofline:
            lines.append(f"- Advisory Roofline: {json.dumps(dict(roofline), sort_keys=True)}")
        lines.extend(_format_regime_ratios(sil_advisory))
        if getattr(sil_advisory, "has_nan_inf_cases", False):
            lines.append("- Guard: has_nan_inf_cases=True (avoid unshifted fast exp overflow)")

    if sil_verification is not None:
        passed = bool(getattr(sil_verification, "passed", True))
        errors = tuple(getattr(sil_verification, "errors", ()) or ())
        warnings = tuple(getattr(sil_verification, "warnings", ()) or ())
        fixes = dict(getattr(sil_verification, "suggested_fixes", {}) or {})
        lines.append(f"- SILVerifier Status: {'PASSED' if passed else 'FAILED'}")
        for err in errors:
            lines.append(f"  * error: {err}")
        for warn in warnings:
            lines.append(f"  * warning: {warn}")
        if fixes:
            lines.append(f"  * suggested_fixes: {json.dumps(fixes, sort_keys=True)}")

    return "\n".join(lines)


def format_target_dialect_layer(
    *,
    escalation_tier: str = "tilelang_dsl",
    track: str = "unconstrained",
    enable_track_router: bool = False,
    pipe_gap_report: PipeGapReport | None = None,
    excluded_mechanisms: Sequence[str] = (),
    topology_axes: TopologyAxes | None = None,
    freedom_mask: ConstraintRelaxationMask | None = None,
    active_symptom: str | None = None,
    case_ratios: CasePhysicsRatios | None = None,
    cann_series: str | None = None,
) -> str:
    """Render Layer 2: grounded target-dialect callable surface and negative exclusions."""
    tier = str(escalation_tier or "tilelang_dsl").strip().lower()
    dialect = get_dialect_surface(tier, cann_series=cann_series)
    header = f"### LAYER 2: TARGET DIALECT API SURFACE (tier={tier}, dialect={dialect.dialect_id})"

    footer_lines: list[str] = []
    if enable_track_router and track and track not in ("", "unconstrained"):
        footer_lines.append(f"- Routed Execution Track: {track}")

    gap_report = pipe_gap_report
    if gap_report is None and tier == "tilelang_dsl":
        gap_report = scan_tilelang_pipe_gaps()
    if gap_report is not None and gap_report.missing_primitives:
        missing_preview = list(gap_report.missing_primitives[:6])
        footer_lines.append(
            f"- TileLang Lowering Coverage: {gap_report.coverage_ratio:.2f} "
            f"(unsupported CANN dav-3510 primitives in pure DSL: {', '.join(missing_preview)})"
        )

    if excluded_mechanisms:
        footer_lines.append("- Negative Structural Exclusions (do NOT duplicate):")
        for item in excluded_mechanisms:
            footer_lines.append(f"  * {item}")

    overhead = len(header) + 1 + sum(len(line) + 1 for line in footer_lines)
    surface_budget = max(512, 10000 - overhead)
    filtered = filter_primitives_on_demand(
        dialect,
        track=track,
        topology_axes=topology_axes,
        freedom_mask=freedom_mask,
        active_symptom=active_symptom,
        case_ratios=case_ratios,
    )
    surface = render_layer2_surface(
        dialect,
        primitives=filtered,
        include_guardrails=True,
        max_chars=surface_budget,
    )
    return "\n".join([header, surface, *footer_lines])


def format_blueprint_delta_section(
    blueprint: HirBlueprintPacket,
    *,
    current_ledger: ObjectivePhysicalLedger | None = None,
) -> str:
    """Render Layer 3 dialect-neutral Roofline, Z3 floor, and physical ledger deltas.

    Microsecond floors render through repr so a positive value below 1e-4
    stays nonzero and round-trips.
    """
    rf = blueprint.roofline
    display_dims = rf.resolved_hir_dims or rf.anchor_dims
    dims_text = ", ".join(f"{k}={v}" for k, v in sorted(display_dims.items())) or "default"
    lines = [
        "### LAYER 3: PHYSICAL ROOFLINE & BLUEPRINT DELTA",
        f"- entry: {blueprint.entry_name} (resolved_dims: {dims_text})",
        (
            f"- compute_flops: {rf.compute_flops} "
            f"(cube={rf.cube_flops}, vec={rf.vector_flops}, special={rf.special_ops})"
        ),
        (
            f"- min_gm_bytes: {rf.min_gm_bytes}, "
            f"arithmetic_intensity: {rf.arithmetic_intensity:.4f}"
        ),
        (
            f"- compute_floor_us: {rf.compute_floor_us!r}, "
            f"memory_floor_us: {rf.memory_floor_us!r}, "
            f"ideal_roofline_us: {rf.ideal_roofline_us!r}, "
            f"dominant_bound: {rf.dominant_bound}"
        ),
    ]
    if not blueprint.shape_exact_match:
        lines.append(
            f"- [Structural Reference Only: blueprint dims={blueprint.resolved_hir_dims} "
            f"vs anchor dims={blueprint.requested_anchor_dims}]"
        )
    if blueprint.fallback_reason:
        lines.append(f"- fallback_reason: {blueprint.fallback_reason}")
    if blueprint.z3_pipeline_summary:
        lines.append(
            f"- Ideal Z3 Schedule Floor: {json.dumps(blueprint.z3_pipeline_summary, sort_keys=True)}"
        )
    if blueprint.lowering_diagnostics:
        lines.append(f"- Blueprint Diagnostics: {list(blueprint.lowering_diagnostics)}")

    ref_led = blueprint.reference_ledger
    if current_ledger is not None and ref_led is not None:
        if blueprint.shape_exact_match:
            ref_ub = _ub_peak_bytes(ref_led)
            cur_ub = _ub_peak_bytes(current_ledger)
            ub_delta = cur_ub - ref_ub
            gm_delta = current_ledger.gm_roundtrip_bytes - ref_led.gm_roundtrip_bytes
            if (
                ref_led.steady_state_ii is not None
                and current_ledger.steady_state_ii is not None
            ):
                ii_delta = str(current_ledger.steady_state_ii - ref_led.steady_state_ii)
            else:
                ii_delta = "n/a"
            lines.extend(
                [
                    (
                        f"- steady_state_ii: floor={ref_led.steady_state_ii}, "
                        f"candidate={current_ledger.steady_state_ii}, delta={ii_delta}"
                    ),
                    (
                        f"- dominant_pipe: floor={ref_led.dominant_pipe}, "
                        f"candidate={current_ledger.dominant_pipe}"
                    ),
                    f"- ub_peak_bytes: floor={ref_ub}, candidate={cur_ub}, delta={ub_delta:+d}",
                    (
                        f"- gm_roundtrip_bytes: floor={ref_led.gm_roundtrip_bytes}, "
                        f"candidate={current_ledger.gm_roundtrip_bytes}, delta={gm_delta:+d}"
                    ),
                ]
            )
        else:
            lines.append(
                f"- dominant_pipe: floor={ref_led.dominant_pipe}, "
                f"candidate={current_ledger.dominant_pipe}"
            )
            all_scopes = sorted(set(ref_led.waterlines) | set(current_ledger.waterlines))
            for scope in all_scopes:
                topo = _SCOPE_TOPOLOGY_MAP.get(scope, scope)
                ref_pct = _format_scope_pct(ref_led.waterlines.get(scope))
                cur_pct = _format_scope_pct(current_ledger.waterlines.get(scope))
                lines.append(
                    f"- scope_utilization[{scope} -> {topo}]: "
                    f"floor={ref_pct}, candidate={cur_pct}"
                )
        if (
            current_ledger.min_gm_dma_bytes > 0
            or current_ledger.uses_dynamic_offset_pipelining
            or current_ledger.has_unlimit_memory_attr
            or current_ledger.has_coupled_reduce_sfu_accum
        ):
            lines.append(
                f"- microarch_flags: min_gm_dma_bytes={current_ledger.min_gm_dma_bytes}, "
                f"dma_burst_eff={current_ledger.dma_burst_efficiency_ratio:.3f}, "
                f"dynamic_offset_pipe={current_ledger.uses_dynamic_offset_pipelining}, "
                f"coupled_reduce_sfu={current_ledger.has_coupled_reduce_sfu_accum}"
            )
    return "\n".join(lines)


def build_progressive_llm_constraints(
    *,
    escalation_tier: str = "tilelang_dsl",
    track: str = "unconstrained",
    enable_track_router: bool = False,
    sil_advisory: object | None = None,
    sil_verification: object | None = None,
    freedom_mask: ConstraintRelaxationMask | None = None,
    vector_mode: str = "simdvf",
    ub_256kb_unlock: bool = False,
    blueprint: HirBlueprintPacket | None = None,
    current_ledger: ObjectivePhysicalLedger | None = None,
    pipe_gap_report: PipeGapReport | None = None,
    excluded_mechanisms: Sequence[str] = (),
    extra_blueprint_context: str = "",
    topology_axes: TopologyAxes | None = None,
    active_symptom: str | None = None,
    case_ratios: CasePhysicsRatios | None = None,
    cann_series: str | None = None,
) -> str:
    """Assemble the complete 3-layer progressive LLM constraint block."""
    resolved_axes = topology_axes
    if resolved_axes is None and sil_advisory is not None:
        advisory_axes = getattr(sil_advisory, "topology_axes", None)
        if isinstance(advisory_axes, TopologyAxes):
            resolved_axes = advisory_axes
    sections = [
        format_hardware_and_sil_layer(
            sil_advisory=sil_advisory,
            sil_verification=sil_verification,
            freedom_mask=freedom_mask,
            vector_mode=vector_mode,
            ub_256kb_unlock=ub_256kb_unlock,
        ),
        format_target_dialect_layer(
            escalation_tier=escalation_tier,
            track=track,
            enable_track_router=enable_track_router,
            pipe_gap_report=pipe_gap_report,
            excluded_mechanisms=excluded_mechanisms,
            topology_axes=resolved_axes,
            freedom_mask=freedom_mask,
            active_symptom=active_symptom,
            case_ratios=case_ratios,
            cann_series=cann_series,
        ),
    ]
    if blueprint is not None:
        sections.append(
            format_blueprint_delta_section(blueprint, current_ledger=current_ledger)
        )
    elif extra_blueprint_context.strip():
        sections.append(extra_blueprint_context.strip())
    return "\n\n".join(s for s in sections if s)


def format_g5_parent_monotonicity_lock(*, include_blueprint_retention: bool = False) -> str:
    """Render G5 parent monotonicity lock rules for syntax repair and Stage-0 retries."""
    lines = ["### G5 PARENT MONOTONICITY LOCK"]
    if include_blueprint_retention:
        lines.append(
            "- Retain full Layer 1-3 blueprint architecture, Roofline floor targets, "
            "and PyTorch golden semantics; never degrade into a bare blueprint-less syntax patch."
        )
    lines.extend(
        [
            (
                "- Permitted edits: fix syntax, API arguments, or predicates, or halve tile "
                "knobs by power-of-2 on SRAM overflow."
            ),
            (
                "- Forbidden regressions: NEVER drop parent `T.Pipelined` stages, NEVER "
                "revert to scalar loops, and NEVER downgrade execution domain (e.g. "
                "`coupled_1c2v`/Cube or `T.SimdVF`/`__simd_vf__` down to scalar fallback)."
            ),
        ]
    )
    return "\n".join(lines)


__all__ = [
    "build_progressive_llm_constraints",
    "format_blueprint_delta_section",
    "format_g5_parent_monotonicity_lock",
    "format_hardware_and_sil_layer",
    "format_target_dialect_layer",
]
