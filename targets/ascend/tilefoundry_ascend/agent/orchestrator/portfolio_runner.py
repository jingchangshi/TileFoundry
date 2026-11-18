"""Portfolio evolution runner: K=2 causal rule workers + 1 rule-blind creative worker."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import json

from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.kb.schema import AscendCausalRule, build_predicate_context
from tilefoundry_ascend.agent.orchestrator.gate_funnel import FunnelEvaluation, GateFunnel
from tilefoundry_ascend.agent.orchestrator.llm_client import (
    CREATIVE_RESPONSE_SCHEMA,
    GenerationRequest,
    KernelLLMBackend,
    extract_creative_payload,
    extract_python_dsl,
)
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    BeamCandidate,
    TwoLaneBeamSearch,
    compute_static_dynamic_residual,
    extract_structural_signature,
)
from tilefoundry_ascend.agent.orchestrator.tilesize_prescreen import (
    TileConfigCandidate,
    count_dsl_gemm_calls,
    evaluate_tile_candidate,
    format_prescreen_summary,
    generate_aligned_neighborhood,
    prescreen_live_ir_neighborhood,
    prescreen_tile_configs,
)
from tilefoundry_ascend.agent.orchestrator.twin_bridge import format_hir_tir_navigation_section
from tilefoundry_ascend.agent.router.capability_gate import (
    EscalationDecision,
    evaluate_track_escalation,
)
from tilefoundry_ascend.ascend.cost_model.bisheng_res_parser import BishengResourceSummary
from tilefoundry_ascend.ascend.cost_model.hybrid_cost_model import (
    CostPrediction,
    HybridAscendCostModel,
)
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import HirTirNavigationPacket
from tilefoundry_ascend.ascend.sil.op_analyzer import resolve_canonical_topology_axes
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget
from tilefoundry_ascend.ascend.templates.registry import (
    ConstraintRelaxationMask,
    relax_facets_for_bottleneck,
)

CREATIVE_FOCUS_HINTS: tuple[str, ...] = (
    "memory_hierarchy_and_tiling",
    "compute_pipeline_and_sync",
    "cross_core_dataflow_and_fusion",
)

TILELANG_ASCEND_PRIMITIVE_MENU: str = "\n".join(
    (
        "- T.MixedKernel(28, sids=2): coupled 1 AIC + 2 AIV execution (or pure-AIV sids=2)",
        (
            "- T.Kernel(blocks) / T.MixedKernel(blocks, sids=2): 1D core grid only "
            "(NEVER pass is_npu=... or threads=...)"
        ),
        "- T.Persistent(...): persistent 2D core swizzle loop over logical tiles",
        "- T.Pipelined(count, num_stages=...): multi-buffered loop pipelining",
        (
            "- T.SimdVF(): native SIMD vector-function mode (256 KiB UB with "
            "PURE_SIMDVF_UBUF_FLAGS; T.Parallel must index plain UB tile buffers rather "
            "than pipelined version-counter modulo slices)"
        ),
        "- T.SimtVF(threads=128): SIMT vector-function mode for irregular/scalar-tail loops",
        (
            "- Module pre-imports tilelang, tilelang.ascend.language as T, torch, "
            "PURE_SIMDVF_UBUF_FLAGS (do NOT import tilelang.language as T or call T.Scope)"
        ),
        (
            "- Inside T.MixedKernel(28, sids=2) as (bx, sid), AIC Cube ops (T.gemm, "
            "L1/L0C T.copy) and AIV Vector blocks (with T.SimdVF(): ...) coexist directly "
            "without T.Scope"
        ),
        "- T.gemm(a, b, c, ...): Cube MMAD on L1/L0A/L0B -> L0C",
        "- T.copy(src, dst) / T.dual_copy(src, dst, ...): GM/L1/L0C/UB DMA and 1C2V split copy",
        "- T.alloc_l1 / T.alloc_l0a / T.alloc_l0b / T.alloc_l0c: AIC hierarchical SRAM buffers",
        "- T.alloc_shared(...) / T.alloc_ss(...): AIV Unified Buffer (UB) and 3 KiB shared SSBUF",
        "- T.set_atomic(...): atomic reduction/accumulation mode on GM outputs",
        (
            "- PURE_SIMDVF_UBUF_FLAGS: unlock 256 KiB UB via _select_*_config()"
            "[\"pass_configs\"]/\"compile_flags\" or PrimFunc.attrs"
        ),
        (
            "- Segmented GM DMA burst efficiency: <16 KiB bursts lose HBM bandwidth "
            "(~936 GB/s at 4 KiB vs ~1.64 TB/s at 32 KiB); size pure-vector row tiles "
            "to minimize waves * tile_rows / effective_burst_bw and clamp tail start "
            "min(t * TT, M - TT) when writes are idempotent"
        ),
        (
            "- Static ping-pong double buffering (buf_a/buf_b 2x unroll + cross-tile "
            "head prefetch + barrier pruning): avoids enable_offset dynamic indexing "
            "pointer-alias serialization in backend codegen"
        ),
        (
            "- Two-pass loop fission: split vcadd/vsqrt/vexp score pass into compact UB "
            "weights from pure wide vfma accumulation pass to hide SFU latency chains"
        ),
    )
)

_RULE_CONCEPT_DESCRIPTIONS: dict[str, str] = {
    "R01_DUAL_COPY_1C2V_SPLIT": (
        "1C2V row-splitting via dual_copy across paired AIV sub-cores"
    ),
    "R02_ONCHIP_L0C_TO_UB_DIRECT": (
        "Direct on-chip L0C-to-UB and UB-to-L1 datapath bypass"
    ),
    "R03_ACCUM_SPAN_FIXPIPE_RELIEF": (
        "Inner-K accumulation span enlargement for FixPipe drain amortization"
    ),
    "R04_MTE2_SPARSE_GATHER_COALESCE": (
        "Coalesced 32B-aligned MTE2 block transfers and UB vector gather"
    ),
    "R05_SSBUF_RING_METADATA_HANDOFF": (
        "Shared-scalar SSBUF cross-core metadata handoff"
    ),
    "R06_ONLINE_SOFTMAX_SKIP_SCALE": (
        "Guarded online reduction skip-scale and deferred epilogue normalization"
    ),
    "R07_IMPLICIT_BLOAT_HEADROOM_TRIM": (
        "Tile blocking trim for implicit compiler scratchpad headroom"
    ),
    "R08_PERSISTENT_SWIZZLE_L2_LOCALITY": (
        "Persistent 2D block swizzling for L2 cache locality"
    ),
    "R09_MIXED_QUANT_DEQUANT_STAGE_OVERLAP": (
        "Pipelined AIV dequantization staged ahead of Cube MMAD"
    ),
    "R10_BACKWARD_SYNC_CV_PINGPONG": (
        "Dual-slot ping-pong parity sync decoupling AIC and AIV"
    ),
    "R11_L2_NZ_PAD_BANK_CONFLICT_AVOID": (
        "NZ fractal stride padding for SRAM bank conflict avoidance"
    ),
    "R12_UNIT_FLAG_FIXPIPE_CHAIN_FUSION": (
        "Fine-grained hardware unit_flag handoff between MMAD and FixPipe"
    ),
    "R13_SPLIT_KV_WAVE_RECOVERY": (
        "Grid block splitting across paired sub-cores for wave occupancy recovery"
    ),
    "R14_GQA_HEAD_M_FOLDING": (
        "Query-group head folding into Cube M-dimension for fractal utilization"
    ),
    "R15_ASYMMETRIC_1C2V_STAGE_FISSION": (
        "Asymmetric 1C2V stage fission and pipelined UB footprint compression"
    ),
    "R16_PADDED_PITCH_TAIL_MASKING": (
        "32B-aligned vector pitch padding with SIMD tail masking"
    ),
    "R17_RMEM_ACCUM_AND_256KB_UB_UNLOCK": (
        "Register-resident loop accumulation and full 256 KiB UB unlock"
    ),
    "R18_WAVE_QUANTIZED_BURST_TILING": (
        "Wave-quantized pure-vector tiling co-selected with HBM DMA burst curve"
    ),
    "R19_STATIC_PINGPONG_UNROLL_NO_ALIAS": (
        "Static 2x ping-pong buffer unrolling and prefetch to avoid dynamic-offset aliasing"
    ),
    "R20_LOOP_FISSION_REDUCE_SFU_HIDING": (
        "Two-pass loop fission separating reduce/SFU chains from wide vfma accumulation"
    ),
    "R21_SUBROW_DEPTH_OR_WIDTH_CORE_SHARDING": (
        "Sub-row reduction depth or 64-element hidden-width core sharding across AIVs"
    ),
    "R22_HALF_L1_BANK_ISOLATION": (
        "Half-L1 ping/pong bank isolation ((TOTAL_L1_SIZE >> 1) * slot) across 2x256KB L1 banks"
    ),
    "R23_UB_16BANK_STRIDE_PAD_AND_SHIFT": (
        "UB 16-bank conflict avoidance via +1 DataBlock row stride pad and +8 DataBlock R/W base shift"
    ),
    "R24_VLOOPV2_LEGALIZATION_AND_DIRECT_VF": (
        "VLOOPv2 hardware loop legalization (uint16_t 0-based +1, branchless Select) and direct __simd_vf__"
    ),
    "R25_SCALAR_BOUND_SPILL_ELIMINATION": (
        "Eliminate scalar register/stack spills via local scalar binding and 3-stage prologue/loop/epilogue"
    ),
    "R26_SWAT_SNAKE_AND_TAIL_WAVE_SPLIT": (
        "SWAT serpentine tile swizzle (WINDOW_LEN=4) and tail-wave sub-tile / DPSK Stream-K rebalance"
    ),
    "R27_L2_CACHE_BYPASS_SINGLE_USE": (
        "Bypass L2 allocation (CACHE_MODE_DISABLE) on single-use operands when M_tiles==1 or streaming KV"
    ),
    "R28_INPLACE_L1_SLOT_ALIASING": (
        "In-place L1/UB slot aliasing reusing dead slices (e.g. K_rope -> P) across stages"
    ),
    "R29_W4_AIV_UNPACK_AND_FIXPIPE_SCALE_FOLD": (
        "AIV INT4/FP4 unpack via DIST_UNPACK4_B8 or bit-shift + FixPipe deq_scale=64 fold on dav-3510"
    ),
    "R30_REGBASE_MX_QUANT_DATABLOCK": (
        "RegBase microscaling MXFP8/MXFP4 quant via DIST_DINTLV_B16, ReduceDataBlock<MAX>, and SetCtrlSpr"
    ),
    "R31_SIMT_UB_PRIVATIZED_ATOMICS": (
        "SIMT warp-privatized +8-padded UB atomics with __launch_bounds__(512) 0-spill and Simt::UintDiv"
    ),
    "R32_GROUPED_ROLLING_CORE_AND_EMPTY_SKIP": (
        "Grouped MatMul rolling core assignment ((endBlockIdx+1)%coreNum) with m==0 empty-group fast skip"
    ),
    "R33_PERSISTENT_STATE_RESIDENCY": (
        "Persistent on-chip UB/L1 FP32 state residency across chunks/iterations with initC=false accumulation"
    ),
}

_SIGNATURE_DESCRIPTIONS: dict[str, str] = {
    "track:developer": "Standard developer track pipelining",
    "track:manual_ssbuf": "Manual SSBUF cross-core scheduling",
    "track:simt_vf": "SIMT vector-function execution",
    "T.alloc_ss": "SSBUF scalar descriptor allocation",
    "T.dual_copy": "Dual-copy coupled AIC/AIV transfer",
    "T.Persistent": "Persistent block dispatch scheduling",
    "T.SimtVF": "SIMT vector-function region",
    "T.Scope": "Explicit cross-core scope separation",
    "T.MixedKernel": "Coupled mixed-kernel 1C2V dispatch",
    "T.Pipelined": "Multi-stage loop pipelining",
    "T.SimdVF": "Native SIMD vector-function execution",
    "PURE_SIMDVF_UBUF_FLAGS": "Full 256 KiB UB unlock on pure SIMD vector functions",
    "T.set_atomic": "Hardware atomic GM accumulation",
    "unit_flag_ctrl": "Hardware unit-flag MMAD/FixPipe chaining",
    "vgather2": "UB vector-gather indexing",
    "static_pingpong_2buf": "Static double-buffered ping-pong buffers avoiding offset aliasing",
    "loop_fission_multi_pass": "Multi-pass loop fission separating reduction chains from FMA",
    "pipe_all_barrier": "Coarse PIPE_ALL full-pipeline barrier synchronization",
    "vfma_unshifted": "Unshifted vector FMA accumulation pass without per-step rescaling",
    "split_kv_multi": "Multi-split Split-KV stage-1 parallel partition",
}


def _describe_rule_concept(rule: AscendCausalRule) -> str:
    if rule.rule_id in _RULE_CONCEPT_DESCRIPTIONS:
        return _RULE_CONCEPT_DESCRIPTIONS[rule.rule_id]
    lower_id = rule.rule_id.lower()
    if "dual_copy" in lower_id or "1c2v" in lower_id:
        return "1C2V row-splitting via dual_copy across paired AIV sub-cores"
    if "ssbuf" in lower_id:
        return "Shared-scalar SSBUF cross-core handoff"
    if "pipeline" in lower_id or "stage" in lower_id:
        return "Multi-stage pipeline buffering and overlap"
    if "coalesce" in lower_id or "mte2" in lower_id:
        return "Coalesced contiguous MTE2 DMA transfers"
    if "core" in lower_id or "parallel" in lower_id:
        return "Multi-core block concurrency partitioning"
    if "pad" in lower_id or "align" in lower_id:
        return "Fractal alignment and SRAM stride padding"
    if "barrier" in lower_id or "sync" in lower_id or "flag" in lower_id:
        return "Fine-grained cross-pipe synchronization"
    return "Previously matched bottleneck transformation"


def summarize_excluded_mechanisms(
    matched_rules: Sequence[AscendCausalRule],
    exploited_signatures: Iterable[tuple[str, ...]],
) -> tuple[str, ...]:
    """Derive rule-blind generic mechanism descriptions for negative constraints."""
    mechanisms: list[str] = []
    seen: set[str] = set()
    for rule in matched_rules:
        desc = _describe_rule_concept(rule)
        if desc not in seen:
            seen.add(desc)
            mechanisms.append(desc)
    for sig in sorted(exploited_signatures):
        for token in sig:
            desc = _SIGNATURE_DESCRIPTIONS.get(token)
            if desc and desc not in seen:
                seen.add(desc)
                mechanisms.append(desc)
    return tuple(mechanisms)


def build_prescreen_summary_from_kwargs(
    kwargs: Mapping[str, object] | None,
    *,
    dsl_source: str | None = None,
    factory_name: str | None = None,
    var_bindings: Mapping[str, int] | None = None,
    config_selector_name: str | None = None,
    dims: Mapping[str, object] | None = None,
    pass_configs: Mapping[str, object] | None = None,
    compile_flags: Sequence[str] | str | None = None,
) -> str:
    """Build live-IR or analytical dual-accumulator tile prescreen summary."""
    if not kwargs:
        return ""
    lowered = {str(k).lower(): v for k, v in kwargs.items()}
    lowered_dims = {str(k).lower(): v for k, v in (dims or {}).items()}

    def _pick_pos_int(keys: tuple[str, ...]) -> int | None:
        for source_map in (lowered, lowered_dims):
            for key in keys:
                val = source_map.get(key)
                if isinstance(val, int) and not isinstance(val, bool) and val > 0:
                    return val
        return None

    bm_val = _pick_pos_int(("bm", "tile_m"))
    bn_val = _pick_pos_int(("bn", "tile_n"))
    bk_val = _pick_pos_int(("bk", "tile_k"))
    if bm_val is None and bn_val is None and bk_val is None:
        return ""
    stages_val = _pick_pos_int(("stages", "num_stages")) or 2
    dtype_str = str(lowered.get("tl_dtype", lowered.get("dtype", "bfloat16"))).lower()
    dtype_bytes = 4 if ("float32" in dtype_str or dtype_str == "f32") else 2
    d_out_val = _pick_pos_int(("d_out", "d_nope", "d_v")) or 0
    gemm_count = count_dsl_gemm_calls(dsl_source) if dsl_source else 0
    dual_acc = bool(gemm_count >= 2 or d_out_val > 0)

    center = TileConfigCandidate(
        bm=bm_val or 64,
        bn=bn_val or 64,
        bk=bk_val or 64,
        stages=stages_val,
        dtype_bytes=dtype_bytes,
        d_out=d_out_val,
        dual_accumulator=dual_acc,
    )

    if dsl_source and factory_name:
        try:
            live_summary = prescreen_live_ir_neighborhood(
                dsl_source,
                factory_name=factory_name,
                center=center,
                base_kwargs=kwargs,
                var_bindings=var_bindings,
                config_selector_name=config_selector_name,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
                dims=dims,
            )
            if live_summary:
                return live_summary
        except Exception:
            pass

    if gemm_count == 0 and d_out_val == 0:
        return ""

    facts = AscendTarget("ascend.950pr").get_facts(AscendSpaceFacts)
    center_score = evaluate_tile_candidate(center, facts)
    pool = list(generate_aligned_neighborhood(center))
    if not center_score.is_feasible and center.bm >= 64:
        step = 16 * max(1, center.aiv_split)
        half_bm = ((center.bm // 2) // step) * step
        if half_bm >= 16 and half_bm != center.bm:
            for bn_cand in dict.fromkeys((center.bn, max(16, center.bn // 2))):
                if bn_cand % 16 == 0:
                    pool.append(
                        TileConfigCandidate(
                            bm=half_bm,
                            bn=bn_cand,
                            bk=center.bk,
                            stages=center.stages,
                            dtype_bytes=center.dtype_bytes,
                            accum_bytes=center.accum_bytes,
                            aiv_split=center.aiv_split,
                            d_out=center.d_out,
                            dual_accumulator=center.dual_accumulator,
                        )
                    )

    feasible = prescreen_tile_configs(pool, top_k=4)
    display_scores = list(feasible)
    if not center_score.is_feasible:
        display_scores.append(center_score)
    if not display_scores:
        return ""
    return format_prescreen_summary(display_scores)


@dataclass(frozen=True)
class WorkerSpec:
    """Spawned worker specification for one optimization portfolio round."""

    worker_id: str
    worker_kind: str
    rule: AscendCausalRule | None
    request: GenerationRequest
    parent_candidate: BeamCandidate
    ideal_roofline_us: float = 0.0


@dataclass
class PortfolioRoundReport:
    """Summary of one parallel portfolio execution round and updates."""

    round_idx: int
    spawned_workers: tuple[WorkerSpec, ...]
    evaluated_candidates: list[BeamCandidate] = field(default_factory=list)
    failed_candidates: list[tuple[str, str, str]] = field(default_factory=list)
    repaired_worker_ids: list[str] = field(default_factory=list)
    pruned_rule_ids: tuple[str, ...] = ()
    exploit_beam: list[BeamCandidate] = field(default_factory=list)
    stepping_stone_beam: list[BeamCandidate] = field(default_factory=list)


@dataclass(frozen=True)
class RoundEscalationReport:
    """Closed-loop escalation, minimal facet relaxation, and cost prediction for a round."""

    escalation: EscalationDecision
    relaxed_mask: ConstraintRelaxationMask
    cost_prediction: CostPrediction
    roofline_eta: float
    l1_overflow: bool = False
    ub_overflow: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "should_escalate": self.escalation.should_escalate,
            "from_tier": self.escalation.from_tier,
            "to_tier": self.escalation.to_tier,
            "trigger_reasons": list(self.escalation.trigger_reasons),
            "freedom_mask": (
                self.escalation.freedom_mask.to_dict()
                if self.escalation.freedom_mask is not None
                else None
            ),
            "relaxed_mask": self.relaxed_mask.to_dict(),
            "cost_prediction": self.cost_prediction.to_dict(),
            "roofline_eta": self.roofline_eta,
            "l1_overflow": self.l1_overflow,
            "ub_overflow": self.ub_overflow,
        }


def _coerce_topology_axes_opt(raw: object | None) -> TopologyAxes | None:
    if raw is None:
        return None
    if isinstance(raw, TopologyAxes):
        return raw
    if isinstance(raw, Mapping) and all(
        k in raw for k in ("compute_engine", "memory_ring", "iteration_regularity")
    ):
        return TopologyAxes.from_dict(raw)
    return None


def _derive_active_topology_axes(
    ledger: object | None,
    *,
    features: object | None = None,
    sil_advisory: object | None = None,
    route: object | None = None,
    topology_axes: object | None = None,
) -> TopologyAxes:
    explicit = _coerce_topology_axes_opt(topology_axes)
    if explicit is not None:
        return explicit
    if sil_advisory is not None:
        adv_axes = (
            sil_advisory.get("topology_axes")
            if isinstance(sil_advisory, Mapping)
            else getattr(sil_advisory, "topology_axes", None)
        )
        coerced = _coerce_topology_axes_opt(adv_axes)
        if coerced is not None:
            return coerced
    if route is not None:
        r_axes = (
            route.get("topology_axes")
            if isinstance(route, Mapping)
            else getattr(route, "topology_axes", None)
        )
        coerced = _coerce_topology_axes_opt(r_axes)
        if coerced is not None:
            return coerced
    if sil_advisory is not None or features is not None:
        return resolve_canonical_topology_axes(sil_advisory, features=features)

    pipe_cycles = getattr(ledger, "pipe_cycles", None) or {}
    if isinstance(pipe_cycles, Mapping) and (
        pipe_cycles.get("AIC.PIPE_M", 0) > 0 or pipe_cycles.get("AIC.CUBE", 0) > 0
    ):
        return TopologyAxes(
            compute_engine="aic_only",
            memory_ring="gm_l1_l0_fixpipe",
            iteration_regularity="affine_simd",
        )
    return TopologyAxes(
        compute_engine="aiv_only",
        memory_ring="gm_ub_stream",
        iteration_regularity="affine_simd",
    )


def format_sil_diagnostics_section(
    *,
    sil_advisory: object | None = None,
    sil_verification: object | None = None,
) -> str:
    """Format SIL AdvisoryEnvelope and VerificationReport diagnostics for prompts."""
    sections: list[str] = []
    if sil_advisory is not None:
        adv = (
            sil_advisory
            if isinstance(sil_advisory, Mapping)
            else (
                sil_advisory.to_dict()
                if hasattr(sil_advisory, "to_dict")
                else vars(sil_advisory)
            )
        )
        lines = [
            "### SIL ADVISORY ENVELOPE",
            f"- op_name: {adv.get('op_name', '')} (task={adv.get('cannbot_task', '')}, level={adv.get('level', '')})",
            f"- topology_axes: {adv.get('topology_axes', '')}",
            f"- dtypes: in={adv.get('input_dtypes', ())}, out={adv.get('output_dtypes', ())}",
            f"- has_nan_inf_cases: {bool(adv.get('has_nan_inf_cases', False))}, has_inplace_aliasing: {bool(adv.get('has_inplace_aliasing', False))}",
        ]
        bounds = adv.get("legality_bounds")
        if isinstance(bounds, Mapping):
            lines.append(
                f"- dav3510_bounds: L1={bounds.get('l1_bytes')}, UB_simdvf={bounds.get('ub_simdvf_bytes')}, "
                f"UB_simtvf={bounds.get('ub_simtvf_bytes')}, forbidden_dma={bounds.get('forbidden_dma_paths')}, "
                f"forbid_cube_int4={bounds.get('forbid_cube_int4')}"
            )
        roofline = adv.get("roofline")
        if isinstance(roofline, Mapping) and roofline:
            lines.append(f"- roofline: {json.dumps(roofline, sort_keys=True)}")
        regimes = adv.get("regimes")
        if isinstance(regimes, (list, tuple)) and regimes:
            lines.append(f"- stratified_regimes: {len(regimes)}")
        sections.append("\n".join(lines))

    if sil_verification is not None:
        ver = (
            sil_verification
            if isinstance(sil_verification, Mapping)
            else (
                sil_verification.to_dict()
                if hasattr(sil_verification, "to_dict")
                else vars(sil_verification)
            )
        )
        lines = [
            "### SIL VERIFICATION DIAGNOSTICS",
            f"- passed: {bool(ver.get('passed', True))}",
        ]
        errors = ver.get("errors") or ()
        warnings = ver.get("warnings") or ()
        fixes = ver.get("suggested_fixes") or {}
        if errors:
            lines.append("Errors:")
            lines.extend(f"- {e}" for e in errors)
        if warnings:
            lines.append("Warnings:")
            lines.extend(f"- {w}" for w in warnings)
        if fixes:
            lines.append(f"Suggested Fixes: {json.dumps(fixes, sort_keys=True)}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def _resolve_navigation_section(
    parent: BeamCandidate,
    navigation_context: str,
    nav_packet: HirTirNavigationPacket | None,
    *,
    sil_advisory: object | None = None,
    sil_verification: object | None = None,
) -> str:
    parts: list[str] = []
    if nav_packet is not None:
        nav_text = format_hir_tir_navigation_section(
            nav_packet,
            current_tl_ledger=parent.ledger,
        )
        if nav_text:
            parts.append(nav_text)
    elif navigation_context.strip():
        parts.append(navigation_context.strip())
    sil_text = format_sil_diagnostics_section(
        sil_advisory=sil_advisory,
        sil_verification=sil_verification,
    )
    if sil_text:
        parts.append(sil_text)
    joined = "\n\n".join(parts)
    return f"{joined}\n\n" if joined else ""


def build_exploit_request(
    parent: BeamCandidate,
    rule: AscendCausalRule,
    *,
    navigation_context: str = "",
    nav_packet: HirTirNavigationPacket | None = None,
    sil_advisory: object | None = None,
    sil_verification: object | None = None,
) -> GenerationRequest:
    """Build causal-rule worker request routed to exploit model (Hydrogen Medium)."""
    sys_prompt = (
        "You are an Ascend 950PR TileLang DSL optimization engineer. Apply the given causal rule "
        "faithfully to eliminate the bottleneck identified in the physical ledger while "
        "preserving exact numerical equivalence. Output ONLY valid Python code."
    )
    if parent.track in ("", "unconstrained"):
        track_section = (
            f"### TILELANG ASCEND PRIMITIVE REFERENCE\n{TILELANG_ASCEND_PRIMITIVE_MENU}\n\n"
        )
    else:
        track_section = f"### TARGET TRACK\n{parent.track}\n\n"
    nav_section = _resolve_navigation_section(
        parent,
        navigation_context,
        nav_packet,
        sil_advisory=sil_advisory,
        sil_verification=sil_verification,
    )
    user_prompt = (
        f"{track_section}"
        f"{nav_section}"
        f"### SELECTED CAUSAL RULE\n"
        f"Rule ID: {rule.rule_id}\n"
        f"Precondition: {rule.precondition}\n"
        f"Transform Template:\n{rule.transform_template.strip()}\n"
        f"Expected Postcondition: {rule.expected_postcondition}\n\n"
        f"### OBJECTIVE PHYSICAL LEDGER\n{parent.ledger.to_json()}\n\n"
        f"### STATIC-DYNAMIC RESIDUAL\n{json.dumps(parent.residual.to_dict(), indent=2)}\n\n"
        f"### CURRENT KERNEL DSL\n```python\n{parent.dsl_source}\n```\n"
    )
    return GenerationRequest(
        role="exploit_rule",
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        json_schema=None,
    )


def build_creative_request(
    parent: BeamCandidate,
    *,
    excluded_mechanisms: Sequence[str] = (),
    focus_hint: str = "",
    prescreen_summary: str = "",
    navigation_context: str = "",
    nav_packet: HirTirNavigationPacket | None = None,
    effort: str | None = None,
    sil_advisory: object | None = None,
    sil_verification: object | None = None,
) -> GenerationRequest:
    """Build rule-blind creative search request routed to Hydrogen High without rules."""
    sys_prompt = (
        "You are a first-principles Ascend 950PR (dav-3510) systems architect. Propose an "
        "architectural kernel optimization directly from hardware limits (1 AIC + 2 AIV coupled "
        "block, 3 KiB SSBUF, L0C->UB direct path, UB->L1 direct path, 16-entry MTE2 queue, "
        "128 B/cycle FixPipe) and the physical telemetry below. Return JSON matching the schema."
    )
    if excluded_mechanisms:
        constraint_lines = "\n".join(f"- {item}" for item in excluded_mechanisms)
    else:
        constraint_lines = "- None recorded yet; explore any high-impact structural axis."
    neg_section = (
        "### NEGATIVE CONSTRAINTS (ALREADY EXPLORED BY PARALLEL WORKERS)\n"
        f"{constraint_lines}\n"
        "Propose a structurally orthogonal transformation that does NOT duplicate the "
        "mechanisms listed above.\n\n"
    )
    if parent.track in ("", "unconstrained"):
        track_section = (
            f"### TILELANG ASCEND PRIMITIVE REFERENCE\n{TILELANG_ASCEND_PRIMITIVE_MENU}\n\n"
        )
    else:
        track_section = f"### EXECUTION TRACK\n{parent.track}\n\n"
    nav_section = _resolve_navigation_section(
        parent,
        navigation_context,
        nav_packet,
        sil_advisory=sil_advisory,
        sil_verification=sil_verification,
    )
    focus_section = f"### ARCHITECTURAL FOCUS HINT\n{focus_hint}\n\n" if focus_hint else ""
    prescreen_section = (
        f"### TILE PRESCREEN GUIDANCE\n{prescreen_summary}\n\n"
        if prescreen_summary
        else ""
    )
    user_prompt = (
        f"{track_section}"
        f"{nav_section}"
        f"{focus_section}"
        f"{neg_section}"
        f"{prescreen_section}"
        f"### OBJECTIVE PHYSICAL LEDGER\n{parent.ledger.to_json()}\n\n"
        f"### STATIC-DYNAMIC RESIDUAL\n{json.dumps(parent.residual.to_dict(), indent=2)}\n\n"
        f"### CURRENT KERNEL DSL\n```python\n{parent.dsl_source}\n```\n"
    )
    return GenerationRequest(
        role="explore_creative",
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        json_schema=CREATIVE_RESPONSE_SCHEMA,
        effort=effort,
    )


def build_syntax_repair_request(
    dsl_source: str,
    error_message: str,
    *,
    slice_hint: str = "",
    sil_verification: object | None = None,
) -> GenerationRequest:
    """Build bounded syntax/compile repair request routed to Hydrogen Medium."""
    hint_sentence = f" Authorized target: {slice_hint}." if slice_hint else ""
    sys_prompt = (
        "You are a TileLang Ascend compiler repair engineer. Fix the syntax, API, or "
        "memory-capacity failure reported below without altering tensor semantics. "
        "Output ONLY corrected Python code. You may output the fixed subset of "
        "authorized slice functions or prefixed macros shown in BROKEN KERNEL DSL "
        "(omitted sibling functions automatically inherit their parent implementation); "
        "any new top-level helper MUST reuse the exact regime prefix shown in "
        f"BROKEN KERNEL DSL (do NOT add unprefixed or out-of-regime functions).{hint_sentence}"
    )
    sil_text = format_sil_diagnostics_section(sil_verification=sil_verification)
    sil_block = f"{sil_text}\n\n" if sil_text else ""
    user_prompt = (
        f"### VALID TILELANG ASCEND PRIMITIVES\n{TILELANG_ASCEND_PRIMITIVE_MENU}\n\n"
        f"{sil_block}"
        f"### COMPILER / GATE DIAGNOSTIC\n{error_message}\n\n"
        f"### BROKEN KERNEL DSL\n```python\n{dsl_source}\n```\n"
    )
    return GenerationRequest(
        role="syntax_repair",
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        json_schema=None,
    )


class PortfolioRunner:
    """Runs K=2 Rule Exploit + 1 Rule-Blind Creative worker with syntax repair and KB feedback."""

    def __init__(
        self,
        *,
        backend: KernelLLMBackend,
        funnel: GateFunnel,
        tracker: BayesianRuleTracker,
        beam: TwoLaneBeamSearch | None = None,
        k_rule_workers: int = 2,
        max_syntax_repairs: int = 1,
        enable_track_router: bool = False,
        enable_tilesize_prescreen: bool = False,
    ) -> None:
        self._backend = backend
        self._funnel = funnel
        self._tracker = tracker
        self._beam = beam or TwoLaneBeamSearch()
        self._k_rules = max(1, k_rule_workers)
        self._max_repairs = max(0, max_syntax_repairs)
        self._enable_track_router = enable_track_router
        self._enable_tilesize_prescreen = bool(enable_tilesize_prescreen)

    @property
    def beam(self) -> TwoLaneBeamSearch:
        """Return active TwoLaneBeamSearch instance."""
        return self._beam

    def plan_workers(
        self,
        parent: BeamCandidate,
        round_idx: int,
        *,
        factory_name: str | None = None,
        factory_kwargs: Mapping[str, object] | None = None,
        var_bindings: Mapping[str, int] | None = None,
        config_selector_name: str | None = None,
        dims: Mapping[str, object] | None = None,
        navigation_context: str = "",
        ideal_roofline_us: float = 0.0,
        nav_packet: HirTirNavigationPacket | None = None,
        creative_effort: str = "high",
        sil_advisory: object | None = None,
        sil_verification: object | None = None,
        topology_axes: object | None = None,
    ) -> tuple[WorkerSpec, ...]:
        """Select matching causal rules and fill remaining slots with creative explorers."""
        active_axes = _derive_active_topology_axes(
            parent.ledger,
            features=parent.features,
            sil_advisory=sil_advisory,
            topology_axes=topology_axes,
        )
        ctx = build_predicate_context(
            parent.ledger,
            features=parent.features,
            residual=parent.residual,
            extra={"topology_axes": active_axes},
        )
        matched_rules = list(
            self._tracker.rank_applicable_rules(
                ctx,
                top_k=self._k_rules,
                topology_axes=active_axes,
            )
        )
        eff_roofline = (
            ideal_roofline_us
            if ideal_roofline_us > 0.0
            else (nav_packet.roofline.ideal_roofline_us if nav_packet is not None else 0.0)
        )

        specs: list[WorkerSpec] = []
        for idx, rule in enumerate(matched_rules[: self._k_rules]):
            specs.append(
                WorkerSpec(
                    worker_id=f"r{round_idx}_rule_{idx}",
                    worker_kind="rule_exploit",
                    rule=rule,
                    request=build_exploit_request(
                        parent,
                        rule,
                        navigation_context=navigation_context,
                        nav_packet=nav_packet,
                        sil_advisory=sil_advisory,
                        sil_verification=sil_verification,
                    ),
                    parent_candidate=parent,
                    ideal_roofline_us=eff_roofline,
                )
            )

        excluded = summarize_excluded_mechanisms(
            matched_rules,
            self._beam.exploited_signatures,
        )
        if self._enable_tilesize_prescreen:
            prescreen_summary = build_prescreen_summary_from_kwargs(
                factory_kwargs,
                dsl_source=parent.dsl_source,
                factory_name=factory_name,
                var_bindings=var_bindings,
                config_selector_name=config_selector_name,
                dims=dims,
            )
        else:
            prescreen_summary = ""
        deficit = max(0, self._k_rules - len(matched_rules))
        total_creative = 1 + deficit
        for c_idx in range(total_creative):
            wid = f"r{round_idx}_creative" if c_idx == 0 else f"r{round_idx}_creative_{c_idx}"
            hint = (
                CREATIVE_FOCUS_HINTS[c_idx % len(CREATIVE_FOCUS_HINTS)]
                if total_creative > 1
                else ""
            )
            specs.append(
                WorkerSpec(
                    worker_id=wid,
                    worker_kind="creative_explore",
                    rule=None,
                    request=build_creative_request(
                        parent,
                        excluded_mechanisms=excluded,
                        focus_hint=hint,
                        prescreen_summary=prescreen_summary,
                        navigation_context=navigation_context,
                        nav_packet=nav_packet,
                        effort=creative_effort,
                        sil_advisory=sil_advisory,
                        sil_verification=sil_verification,
                    ),
                    parent_candidate=parent,
                    ideal_roofline_us=eff_roofline,
                )
            )
        return tuple(specs)

    async def _run_single_worker(
        self,
        spec: WorkerSpec,
        *,
        round_idx: int,
        factory_name: str | None,
        factory_kwargs: dict[str, object] | None,
        var_bindings: Mapping[str, int] | None = None,
        config_selector_name: str | None = None,
        dims: Mapping[str, object] | None = None,
        sil_advisory: object | None = None,
        sil_verification: object | None = None,
        topology_axes: object | None = None,
    ) -> tuple[BeamCandidate | None, bool, tuple[str, str, str] | None, bool]:
        active_axes = _derive_active_topology_axes(
            spec.parent_candidate.ledger,
            features=spec.parent_candidate.features,
            sil_advisory=sil_advisory,
            topology_axes=topology_axes,
        )
        try:
            raw_reply = await self._backend.generate(spec.request)
        except Exception as exc:
            return None, False, (spec.worker_id, "llm_transport", str(exc)[-400:]), False

        try:
            if spec.worker_kind == "creative_explore":
                creative = extract_creative_payload(raw_reply)
                dsl_source = creative["dsl_source"]
                hypothesis = creative["physical_hypothesis"]
            else:
                dsl_source = extract_python_dsl(raw_reply)
                hypothesis = spec.rule.rule_id if spec.rule is not None else ""
        except Exception as exc:
            if spec.rule is not None:
                self._tracker.record_outcome(
                    spec.rule.rule_id,
                    postcondition_realized=False,
                    topology_axes=active_axes,
                )
            return None, False, (spec.worker_id, "llm_parse", str(exc)[-400:]), False

        repaired = False
        evaluation: FunnelEvaluation = self._funnel.evaluate(
            dsl_source,
            factory_name=factory_name,
            factory_kwargs=factory_kwargs,
            config_selector_name=config_selector_name,
            dims=dims,
            var_bindings=var_bindings,
        )
        if not evaluation.passed and evaluation.failed_stage in ("stage_a", "stage_b"):
            for _ in range(self._max_repairs):
                repair_req = build_syntax_repair_request(
                    dsl_source,
                    evaluation.error_message,
                    sil_verification=sil_verification,
                )
                try:
                    repaired_raw = await self._backend.generate(repair_req)
                except Exception as exc:
                    return None, repaired, (spec.worker_id, "llm_transport", str(exc)[-400:]), False
                try:
                    dsl_source = extract_python_dsl(repaired_raw)
                except Exception as exc:
                    if spec.rule is not None:
                        self._tracker.record_outcome(
                            spec.rule.rule_id,
                            postcondition_realized=False,
                            topology_axes=active_axes,
                        )
                    return None, repaired, (spec.worker_id, "llm_parse", str(exc)[-400:]), False
                repaired = True
                evaluation = self._funnel.evaluate(
                    dsl_source,
                    factory_name=factory_name,
                    factory_kwargs=factory_kwargs,
                    config_selector_name=config_selector_name,
                    dims=dims,
                    var_bindings=var_bindings,
                )
                if evaluation.passed:
                    break

        if not evaluation.passed or evaluation.stage_a.ledger is None:
            if spec.rule is not None:
                self._tracker.record_outcome(
                    spec.rule.rule_id,
                    postcondition_realized=False,
                    topology_axes=active_axes,
                )
            fail_stage = evaluation.failed_stage or "unknown"
            return None, repaired, (spec.worker_id, fail_stage, evaluation.error_message), False

        ledger = evaluation.stage_a.ledger
        comp_summary = evaluation.stage_b.compiler_summary if evaluation.stage_b else None
        cand_axes = _derive_active_topology_axes(
            ledger,
            features=evaluation.stage_a.features or spec.parent_candidate.features,
            sil_advisory=sil_advisory,
            route=evaluation.stage_a.route,
            topology_axes=topology_axes,
        )
        track = (
            evaluation.stage_a.route.track
            if self._enable_track_router and evaluation.stage_a.route is not None
            else (
                evaluation.stage_a.route.track
                if evaluation.stage_a.route is not None
                else (spec.parent_candidate.track or "unconstrained")
            )
        )
        stage_c = evaluation.stage_c
        is_analytical_proxy = False
        if stage_c is not None:
            latency_us = stage_c.measured_latency_us
            acc_ok = stage_c.accuracy_passed
            tel = dict(stage_c.telemetry)
            is_analytical_proxy = bool(tel.get("is_analytical_proxy", False))
            dur_us = float(tel.get("dur_us", latency_us))
            if (
                spec.ideal_roofline_us > 0.0
                and dur_us > 0.0
                and "roofline_eta" not in tel
            ):
                eta = min(1.0, spec.ideal_roofline_us / max(1e-6, dur_us))
                tel["roofline_eta"] = eta
                if isinstance(stage_c.telemetry, dict):
                    stage_c.telemetry["roofline_eta"] = eta
                residual = compute_static_dynamic_residual(
                    ledger, tel, compiler_summary=comp_summary
                )
            else:
                residual = (
                    stage_c.residual
                    if stage_c.residual is not None
                    else spec.parent_candidate.residual
                )
        else:
            residual = spec.parent_candidate.residual
            latency_us = spec.parent_candidate.latency_us
            acc_ok = True

        if spec.rule is not None:
            post_ctx = build_predicate_context(
                ledger,
                features=evaluation.stage_a.features,
                residual=residual,
                compiler_summary=comp_summary,
                prev_ledger=spec.parent_candidate.ledger,
            )
            prev_ns = post_ctx.get("prev")
            if isinstance(prev_ns, dict):
                prev_ns.update(spec.parent_candidate.residual.to_dict())
            realized = spec.rule.postcondition_realized(post_ctx)
            parent_us = spec.parent_candidate.latency_us
            speed_gain = (
                ((parent_us - latency_us) / parent_us)
                if parent_us > 0.0 and latency_us > 0.0
                else 0.0
            )
            if not is_analytical_proxy or realized:
                self._tracker.record_outcome(
                    spec.rule.rule_id,
                    postcondition_realized=realized,
                    speed_gain=max(0.0, speed_gain),
                    topology_axes=cand_axes,
                )

        cand = BeamCandidate(
            candidate_id=spec.worker_id,
            round_idx=round_idx,
            worker_kind=spec.worker_kind,
            rule_id=spec.rule.rule_id if spec.rule else None,
            dsl_source=dsl_source,
            track=track,
            ledger=ledger,
            residual=residual,
            latency_us=latency_us,
            accuracy_passed=acc_ok,
            structural_signature=extract_structural_signature(
                dsl_source,
                track=track,
                residual=residual,
                stage_b=evaluation.stage_b,
            ),
            hypothesis=hypothesis,
            parent_id=spec.parent_candidate.candidate_id,
            features=evaluation.stage_a.features or spec.parent_candidate.features,
        )
        return cand, repaired, None, is_analytical_proxy

    async def run_round(
        self,
        parent: BeamCandidate,
        *,
        round_idx: int = 1,
        factory_name: str | None = None,
        factory_kwargs: dict[str, object] | None = None,
        var_bindings: Mapping[str, int] | None = None,
        config_selector_name: str | None = None,
        dims: Mapping[str, object] | None = None,
        navigation_context: str = "",
        ideal_roofline_us: float = 0.0,
        nav_packet: HirTirNavigationPacket | None = None,
        creative_effort: str = "high",
        sil_advisory: object | None = None,
        sil_verification: object | None = None,
        topology_axes: object | None = None,
    ) -> PortfolioRoundReport:
        """Execute one parallel portfolio round across Rule and Creative workers."""
        active_axes = _derive_active_topology_axes(
            parent.ledger,
            features=parent.features,
            sil_advisory=sil_advisory,
            topology_axes=topology_axes,
        )
        specs = self.plan_workers(
            parent,
            round_idx,
            factory_name=factory_name,
            factory_kwargs=factory_kwargs,
            var_bindings=var_bindings,
            config_selector_name=config_selector_name,
            dims=dims,
            navigation_context=navigation_context,
            ideal_roofline_us=ideal_roofline_us,
            nav_packet=nav_packet,
            creative_effort=creative_effort,
            sil_advisory=sil_advisory,
            sil_verification=sil_verification,
            topology_axes=active_axes,
        )
        tasks = [
            self._run_single_worker(
                spec,
                round_idx=round_idx,
                factory_name=factory_name,
                factory_kwargs=factory_kwargs,
                var_bindings=var_bindings,
                config_selector_name=config_selector_name,
                dims=dims,
                sil_advisory=sil_advisory,
                sil_verification=sil_verification,
                topology_axes=active_axes,
            )
            for spec in specs
        ]
        outcomes = await asyncio.gather(*tasks)

        report = PortfolioRoundReport(round_idx=round_idx, spawned_workers=specs)
        saw_proxy = False
        saw_real_hw = False
        for cand, repaired, fail, is_proxy in outcomes:
            if repaired:
                uid = cand.candidate_id if cand else (fail[0] if fail else "")
                report.repaired_worker_ids.append(uid)
            if cand is not None:
                report.evaluated_candidates.append(cand)
                if is_proxy:
                    saw_proxy = True
                else:
                    saw_real_hw = True
            elif fail is not None:
                report.failed_candidates.append(fail)

        if saw_proxy and not saw_real_hw:
            pruned: tuple[str, ...] = ()
        else:
            pruned = self._tracker.prune_underperforming(topology_axes=active_axes)
        report.pruned_rule_ids = pruned
        exploit, stone = self._beam.update([parent, *report.evaluated_candidates])
        report.exploit_beam = exploit
        report.stepping_stone_beam = stone
        return report

    def check_round_escalation(
        self,
        candidate: BeamCandidate,
        *,
        current_tier: str = "tilelang_dsl",
        target_eta: float = 0.85,
        stagnation_rounds: int = 0,
        ideal_roofline_us: float = 0.0,
        dominant_bound: str = "",
        compiler_summary: BishengResourceSummary | Mapping[str, object] | None = None,
        current_mask: ConstraintRelaxationMask | None = None,
        cost_model: HybridAscendCostModel | None = None,
        workload: Mapping[str, object] | None = None,
    ) -> RoundEscalationReport:
        """Closed-loop escalation check wiring HybridAscendCostModel + evaluate_track_escalation + relax_facets_for_bottleneck."""
        model = cost_model if cost_model is not None else HybridAscendCostModel(AscendTarget("ascend.950pr"))
        eff_workload = dict(workload or {})
        if not eff_workload:
            pipes = candidate.ledger.pipe_cycles if candidate.ledger else {}
            cube_c = int(pipes.get("AIC.PIPE_M", 0)) + int(pipes.get("AIC.CUBE", 0))
            vec_c = int(pipes.get("AIV0.PIPE_V", 0)) + int(pipes.get("AIV1.PIPE_V", 0))
            eff_workload = {
                "cube_flops": float(max(0, cube_c)) * 4096.0,
                "vec_flops": float(max(1, vec_c)) * 128.0,
                "gm_bytes": float(max(65536, getattr(candidate.ledger, "gm_roundtrip_bytes", 0) or 65536)),
                "contig_burst_bytes": int(getattr(candidate.ledger, "dominant_contig_bytes", 0) or 32768),
                "total_tiles": int(getattr(candidate.ledger, "grid_blocks", 0) or 56),
            }

        eff_summary: BishengResourceSummary | None
        if compiler_summary is None:
            eff_summary = BishengResourceSummary(
                spill_bytes=int(candidate.residual.spill_bytes),
                vloopv2_loops=1 if candidate.residual.vloopv2_hit else 0,
                software_loops=0 if candidate.residual.vloopv2_hit else 1,
                has_local_membar_in_vf_call=bool(candidate.residual.has_local_membar_in_vf_call),
                vf_fusion_breaks=int(candidate.residual.vf_fusion_breaks),
                alias_serialized_buffers=int(candidate.residual.alias_serialized_buffers),
                dual_issue_ipc_collapse=float(candidate.residual.dual_issue_ipc_collapse),
                dcache_dcci_flushes=int(candidate.residual.dcache_dcci_flushes),
            )
        elif isinstance(compiler_summary, BishengResourceSummary):
            eff_summary = compiler_summary
        else:
            eff_summary = BishengResourceSummary.from_dict(compiler_summary)

        cost_pred = model.predict(eff_workload, bisheng_res=eff_summary)

        if ideal_roofline_us > 0.0 and candidate.latency_us > 0.0:
            eta = min(1.0, float(ideal_roofline_us) / float(candidate.latency_us))
        elif cost_pred.pred_us > 0.0 and cost_pred.tier0_physics_us > 0.0:
            eta = min(1.0, float(cost_pred.tier0_physics_us) / float(cost_pred.pred_us))
        else:
            eta = max(0.0, 1.0 - min(1.0, float(candidate.residual.unexplained_gap_ratio)))

        eff_bound = dominant_bound.strip() or cost_pred.dominant_bound
        escalation = evaluate_track_escalation(
            current_tier=current_tier,
            roofline_eta=eta,
            target_eta=target_eta,
            stagnation_rounds=stagnation_rounds,
            dominant_bound=eff_bound,
            compiler_summary=eff_summary,
            structural_features=candidate.features,
        )

        waterlines = getattr(candidate.ledger, "waterlines", {}) or {}
        l1_wl = waterlines.get("L1")
        ub_wl = waterlines.get("UB")
        l1_overflow = bool(l1_wl is not None and int(getattr(l1_wl, "overflow_bytes", 0) or 0) > 0)
        ub_overflow = bool(ub_wl is not None and int(getattr(ub_wl, "overflow_bytes", 0) or 0) > 0)

        base_mask = current_mask if current_mask is not None else ConstraintRelaxationMask()
        relaxed_mask = relax_facets_for_bottleneck(
            base_mask,
            dominant_bound=eff_bound,
            residual=candidate.residual,
            compiler_summary=eff_summary,
            l1_overflow=l1_overflow,
            ub_overflow=ub_overflow,
        )
        return RoundEscalationReport(
            escalation=escalation,
            relaxed_mask=relaxed_mask,
            cost_prediction=cost_pred,
            roofline_eta=round(eta, 6),
            l1_overflow=l1_overflow,
            ub_overflow=ub_overflow,
        )


__all__ = [
    "CREATIVE_FOCUS_HINTS",
    "PortfolioRoundReport",
    "PortfolioRunner",
    "RoundEscalationReport",
    "TILELANG_ASCEND_PRIMITIVE_MENU",
    "WorkerSpec",
    "build_creative_request",
    "build_exploit_request",
    "build_prescreen_summary_from_kwargs",
    "build_syntax_repair_request",
    "format_sil_diagnostics_section",
    "summarize_excluded_mechanisms",
]
