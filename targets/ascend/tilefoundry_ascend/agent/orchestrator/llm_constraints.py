"""Unified 3-layer progressive LLM generation constraint synthesizer for Ascend 950."""

from __future__ import annotations

import json
from collections.abc import Sequence

from tilefoundry_ascend.agent.router.isa_gap_scanner import PipeGapReport, scan_tilelang_pipe_gaps
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import HirBlueprintPacket
from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger
from tilefoundry_ascend.ascend.sil.schema import (
    VALID_FREEDOM_FACETS, AdvisoryEnvelope, CasePhysicsRatios,
    ConstraintRelaxationMask, HardwareLegalityBounds, TopologyAxes,
)
from tilefoundry_ascend.ascend.target.dialect_surface import (
    filter_primitives_on_demand, get_dialect_surface, render_layer2_surface,
)

_SCOPE_TOPOLOGY_MAP = {
    "UB": "smem (AIV)", "L1": "tmem (AIC)", "L0A": "tmem (AIC)",
    "L0B": "tmem (AIC)", "L0C": "tmem (AIC)", "SSBUF": "ssbuf (block)",
}
_G1_G4_REDLINES = (
    "- Schedule Integrity Redlines (paired canonical escape paths):",
    "  * G1 (No Scalar Compute Emulation): when vector/matrix semantics exist, never emulate main compute with scalar C++ `for` loops or bit-level emulation -> canonical path: `T.SimdVF` / `simd_inst::*` / `asc_mmad`.",
    "  * G2 (No Dense-Loop Narrow DCache-Bypass Stores): never emit elementwise `< 32B` GM stores triggering `write_gm_bypass_dcache` inside dense loops -> canonical path: pack multiple `< 32B` rows into contiguous `>= 32B` atomic bursts.",
    "  * G3 (No Cache-Key Pollution or Static Unrolled 2D Dynamic Copy): never put dynamic batch/seq dims or float hyperparams in `_KERNEL_CACHE` keys and never split 2D dynamic `T.copy` with Python static loops -> canonical path: `T.dynamic` + 2D `T.copy` + `T.float32` scalar params.",
    "  * G4 (No Overlapping `coupled_1c2v` Dual-Subcore Writes): under `coupled_1c2v`, never issue overlapping `MTE3` writes from `AIV0`/`AIV1` to the same GM slice -> canonical path: disjointly bisect output slices by subcore index.",
)


def format_hardware_and_sil_layer(
    *, sil_advisory: object | None = None, sil_verification: object | None = None,
    freedom_mask: ConstraintRelaxationMask | None = None,
    vector_mode: str = "simdvf", ub_256kb_unlock: bool = False,
) -> str:
    """Render Layer 1: hardware legality redlines and active 7-facet MECE shell locks."""
    bounds = getattr(sil_advisory, "legality_bounds", None)
    if not isinstance(bounds, HardwareLegalityBounds):
        bounds = HardwareLegalityBounds()
    vmode = str(vector_mode or "simdvf").lower()
    ub_cap = bounds.ub_simtvf_bytes if vmode in ("simtvf", "hybrid") else (bounds.ub_unreserved_bytes if ub_256kb_unlock else bounds.ub_simdvf_bytes)
    lines = [
        "### LAYER 1: HARDWARE REDLINES & 7-FACET SHELL CONTRACT",
        f"- Scratchpad Ceilings: UB<={ub_cap} B (mode={vmode}), L1<={bounds.l1_bytes} B (half-bank ping-pong<={bounds.l1_half_bank_bytes} B), L0A<={bounds.l0a_bytes} B, L0B<={bounds.l0b_bytes} B, L0C<={bounds.l0c_bytes} B, SSBUF<={bounds.ssbuf_bytes} B",
        f"- Hard Invariants: forbidden_dma={list(bounds.forbidden_dma_paths)}, max_mmad_mnk<={bounds.max_mmad_mnk}, crosscore_flag_depth<={bounds.max_crosscore_depth}, forbid_cube_int4={bounds.forbid_cube_int4}",
        *_G1_G4_REDLINES,
    ]
    active_mask = freedom_mask or ConstraintRelaxationMask.bottom()
    if active_mask.is_unrestricted_top:
        lines.append("- 7-Facet MECE Freedom Lattice: UNRESTRICTED TOP (all 7 facets unlocked)")
    else:
        unlocked = [f for f in VALID_FREEDOM_FACETS if getattr(active_mask, f, False)]
        locked = [f for f in VALID_FREEDOM_FACETS if not getattr(active_mask, f, False)]
        lines.extend((f"- Unlocked Facets ({len(unlocked)}): {', '.join(unlocked)}", f"- Locked Shell Facets ({len(locked)}): {', '.join(locked)}"))
    if sil_advisory is not None:
        axes = getattr(sil_advisory, "topology_axes", None)
        lines.append(f"- TopologyAxes: {json.dumps(axes.to_dict() if hasattr(axes, 'to_dict') else axes, sort_keys=True)}")
        if roofline := getattr(sil_advisory, "roofline", None):
            lines.append(f"- Advisory Roofline: {json.dumps(dict(roofline), sort_keys=True)}")
        if isinstance(sil_advisory, AdvisoryEnvelope):
            for reg in sil_advisory.regimes:
                r = reg.representative_ratios()
                p = f"- Regime {reg.regime_tag} cases={list(reg.case_nums)}"
                lines.append(f"{p} CasePhysicsRatios: unavailable" if r is None else f"{p} representative_case={r.case_id} CasePhysicsRatios: {json.dumps(r.to_dict(), sort_keys=True)}")
        if getattr(sil_advisory, "has_nan_inf_cases", False):
            lines.append("- Guard: has_nan_inf_cases=True (avoid unshifted fast exp overflow)")
    if sil_verification is not None:
        lines.append(f"- SILVerifier Status: {'PASSED' if getattr(sil_verification, 'passed', True) else 'FAILED'}")
        lines.extend(f"  * error: {e}" for e in (getattr(sil_verification, "errors", ()) or ()))
        lines.extend(f"  * warning: {w}" for w in (getattr(sil_verification, "warnings", ()) or ()))
        if fixes := dict(getattr(sil_verification, "suggested_fixes", {}) or {}):
            lines.append(f"  * suggested_fixes: {json.dumps(fixes, sort_keys=True)}")
    return "\n".join(lines)


def format_target_dialect_layer(
    *, escalation_tier: str = "tilelang_dsl", track: str = "unconstrained",
    enable_track_router: bool = False, pipe_gap_report: PipeGapReport | None = None,
    excluded_mechanisms: Sequence[str] = (), topology_axes: TopologyAxes | None = None,
    freedom_mask: ConstraintRelaxationMask | None = None, active_symptom: str | None = None,
    case_ratios: CasePhysicsRatios | None = None, cann_series: str | None = None,
) -> str:
    """Render Layer 2: grounded target-dialect callable surface and negative exclusions."""
    tier = str(escalation_tier or "tilelang_dsl").strip().lower()
    dialect = get_dialect_surface(tier, cann_series=cann_series)
    header = f"### LAYER 2: TARGET DIALECT API SURFACE (tier={tier}, dialect={dialect.dialect_id})"
    footer_lines: list[str] = []
    if enable_track_router and track and track not in ("", "unconstrained"):
        footer_lines.append(f"- Routed Execution Track: {track}")
    gap = pipe_gap_report or (scan_tilelang_pipe_gaps() if tier == "tilelang_dsl" else None)
    if gap is not None and gap.missing_primitives:
        footer_lines.append(f"- TileLang Lowering Coverage: {gap.coverage_ratio:.2f} (unsupported CANN dav-3510 primitives in pure DSL: {', '.join(list(gap.missing_primitives[:6]))})")
    if excluded_mechanisms:
        footer_lines.append("- Negative Structural Exclusions (do NOT duplicate):")
        footer_lines.extend(f"  * {item}" for item in excluded_mechanisms)
    overhead = len(header) + 1 + sum(len(line) + 1 for line in footer_lines)
    filtered = filter_primitives_on_demand(
        dialect, track=track, topology_axes=topology_axes, freedom_mask=freedom_mask,
        active_symptom=active_symptom, case_ratios=case_ratios,
    )
    surface = render_layer2_surface(dialect, primitives=filtered, include_guardrails=True, max_chars=max(512, 10000 - overhead))
    return "\n".join([header, surface, *footer_lines])


def format_blueprint_delta_section(
    blueprint: HirBlueprintPacket, *, current_ledger: ObjectivePhysicalLedger | None = None
) -> str:
    """Render Layer 3 dialect-neutral Roofline and physical ledger deltas.

    Microsecond floors render through repr so a positive value below 1e-4
    stays nonzero and round-trips.
    """
    rf = blueprint.roofline
    display_dims = rf.resolved_hir_dims or rf.anchor_dims
    dims_text = ", ".join(f"{k}={v}" for k, v in sorted(display_dims.items())) or "default"
    lines = [
        "### LAYER 3: PHYSICAL ROOFLINE & BLUEPRINT DELTA",
        f"- entry: {blueprint.entry_name} (resolved_dims: {dims_text})",
        f"- compute_flops: {rf.compute_flops} (cube={rf.cube_flops}, vec={rf.vector_flops}, special={rf.special_ops})",
        f"- min_gm_bytes: {rf.min_gm_bytes}, arithmetic_intensity: {rf.arithmetic_intensity:.4f}",
        f"- compute_floor_us: {rf.compute_floor_us!r}, memory_floor_us: {rf.memory_floor_us!r}, ideal_roofline_us: {rf.ideal_roofline_us!r}, dominant_bound: {rf.dominant_bound}",
    ]
    if not blueprint.shape_exact_match:
        lines.append(f"- [Structural Reference Only: blueprint dims={blueprint.resolved_hir_dims} vs anchor dims={blueprint.requested_anchor_dims}]")
    if blueprint.fallback_reason:
        lines.append(f"- fallback_reason: {blueprint.fallback_reason}")
    if blueprint.lowering_diagnostics:
        lines.append(f"- Blueprint Diagnostics: {list(blueprint.lowering_diagnostics)}")
    if current_ledger is not None:
        if blueprint.shape_exact_match:
            ub = current_ledger.waterlines.get("UB")
            cur_ub = ub.peak_allocated_bytes if ub is not None else 0
            lines.extend([
                f"- steady_state_ii: candidate={current_ledger.steady_state_ii}",
                f"- dominant_pipe: candidate={current_ledger.dominant_pipe}",
                f"- ub_peak_bytes: candidate={cur_ub}",
                f"- gm_roundtrip_bytes: floor={rf.min_gm_bytes}, candidate={current_ledger.gm_roundtrip_bytes}, delta={current_ledger.gm_roundtrip_bytes - rf.min_gm_bytes:+d}",
            ])
        else:
            lines.append(f"- dominant_pipe: candidate={current_ledger.dominant_pipe}")
            for scope in sorted(current_ledger.waterlines):
                wl = current_ledger.waterlines.get(scope)
                cur_pct = f"{100.0 * wl.peak_allocated_bytes / wl.capacity_bytes:.2f}%" if wl is not None and wl.capacity_bytes > 0 else "n/a"
                lines.append(f"- scope_utilization[{scope} -> {_SCOPE_TOPOLOGY_MAP.get(scope, scope)}]: candidate={cur_pct}")
        if current_ledger.min_gm_dma_bytes > 0 or current_ledger.uses_dynamic_offset_pipelining or current_ledger.has_unlimit_memory_attr or current_ledger.has_coupled_reduce_sfu_accum:
            lines.append(
                f"- microarch_flags: min_gm_dma_bytes={current_ledger.min_gm_dma_bytes}, "
                f"dma_burst_eff={current_ledger.dma_burst_efficiency_ratio:.3f}, "
                f"dynamic_offset_pipe={current_ledger.uses_dynamic_offset_pipelining}, "
                f"coupled_reduce_sfu={current_ledger.has_coupled_reduce_sfu_accum}"
            )
    return "\n".join(lines)


def build_progressive_llm_constraints(
    *, escalation_tier: str = "tilelang_dsl", track: str = "unconstrained",
    enable_track_router: bool = False, sil_advisory: object | None = None,
    sil_verification: object | None = None, freedom_mask: ConstraintRelaxationMask | None = None,
    vector_mode: str = "simdvf", ub_256kb_unlock: bool = False,
    blueprint: HirBlueprintPacket | None = None, current_ledger: ObjectivePhysicalLedger | None = None,
    pipe_gap_report: PipeGapReport | None = None, excluded_mechanisms: Sequence[str] = (),
    extra_blueprint_context: str = "", topology_axes: TopologyAxes | None = None,
    active_symptom: str | None = None, case_ratios: CasePhysicsRatios | None = None,
    cann_series: str | None = None,
) -> str:
    """Assemble the complete 3-layer progressive LLM constraint block."""
    resolved_axes = topology_axes
    if resolved_axes is None and isinstance(getattr(sil_advisory, "topology_axes", None), TopologyAxes):
        resolved_axes = getattr(sil_advisory, "topology_axes")
    sections = [
        format_hardware_and_sil_layer(
            sil_advisory=sil_advisory, sil_verification=sil_verification,
            freedom_mask=freedom_mask, vector_mode=vector_mode, ub_256kb_unlock=ub_256kb_unlock,
        ),
        format_target_dialect_layer(
            escalation_tier=escalation_tier, track=track, enable_track_router=enable_track_router,
            pipe_gap_report=pipe_gap_report, excluded_mechanisms=excluded_mechanisms,
            topology_axes=resolved_axes, freedom_mask=freedom_mask,
            active_symptom=active_symptom, case_ratios=case_ratios, cann_series=cann_series,
        ),
    ]
    if blueprint is not None:
        sections.append(format_blueprint_delta_section(blueprint, current_ledger=current_ledger))
    elif extra_blueprint_context.strip():
        sections.append(extra_blueprint_context.strip())
    return "\n\n".join(s for s in sections if s)


def format_g5_parent_monotonicity_lock(*, include_blueprint_retention: bool = False) -> str:
    """Render G5 parent monotonicity lock rules for syntax repair and Stage-0 retries."""
    lines = ["### G5 PARENT MONOTONICITY LOCK"]
    if include_blueprint_retention:
        lines.append("- Retain full Layer 1-3 blueprint architecture, Roofline floor targets, and PyTorch golden semantics; never degrade into a bare blueprint-less syntax patch.")
    lines.extend([
        "- Permitted edits: fix syntax, API arguments, or predicates, or halve tile knobs by power-of-2 on SRAM overflow.",
        "- Forbidden regressions: NEVER drop parent `T.Pipelined` stages, NEVER revert to scalar loops, and NEVER downgrade execution domain (e.g. `coupled_1c2v`/Cube or `T.SimdVF`/`__simd_vf__` down to scalar fallback).",
    ])
    return "\n".join(lines)


__all__ = [
    "build_progressive_llm_constraints", "format_blueprint_delta_section",
    "format_g5_parent_monotonicity_lock", "format_hardware_and_sil_layer",
    "format_target_dialect_layer",
]
