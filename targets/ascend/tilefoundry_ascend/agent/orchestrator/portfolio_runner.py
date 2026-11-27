"""Portfolio evolution runner: K=2 causal rule workers + 1 rule-blind creative worker."""

from __future__ import annotations

import ast
import asyncio
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.kb.schema import AscendCausalRule, build_predicate_context, rank_symptom_families
from tilefoundry_ascend.agent.gates.cpp_slice_protocol import (
    _looks_like_python_primfunc_slice, classify_stage_failure_for_repair, compact_creative_source_slice,
    extract_repair_closure_slice, parse_scoped_search_replace_blocks, split_tracka_editable_and_readonly_tail,
)
from tilefoundry_ascend.agent.gates.gate_funnel import (
    FunnelEvaluation, GateFunnel, check_repair_monotonicity, extract_ordered_tensor_shapes,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import (
    GenerationRequest, KernelLLMBackend, extract_creative_payload, extract_python_dsl,
)
from tilefoundry_ascend.agent.orchestrator.llm_constraints import (
    build_progressive_llm_constraints, format_g5_parent_monotonicity_lock,
    format_hardware_and_sil_layer, format_target_dialect_layer,
)
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    BeamCandidate, TwoLaneBeamSearch, compiler_summary_from_residual,
    compute_static_dynamic_residual, extract_structural_signature,
)
from tilefoundry_ascend.agent.orchestrator.tilesize_prescreen import (
    TileConfigCandidate, count_dsl_gemm_calls, evaluate_tile_candidate, format_prescreen_summary,
    generate_aligned_neighborhood, prescreen_live_ir_neighborhood, prescreen_tile_configs,
)
from tilefoundry_ascend.agent.router.capability_gate import EscalationDecision, evaluate_track_escalation
from tilefoundry_ascend.ascend.cost_model.bisheng_res_parser import BishengResourceSummary
from tilefoundry_ascend.ascend.cost_model.cost_gate_policy import evaluate_cost_gate_policy
from tilefoundry_ascend.ascend.cost_model.hybrid_cost_model import (
    CostFeatures, CostPrediction, HybridAscendCostModel, build_cost_features_from_ledger,
)
from tilefoundry_ascend.ascend.json_record import JsonRecordMixin
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import HirBlueprintPacket
from tilefoundry_ascend.ascend.passes.observation_space import RooflineSummaryLike
from tilefoundry_ascend.ascend.sil.op_analyzer import resolve_canonical_topology_axes
from tilefoundry_ascend.ascend.sil.schema import CasePhysicsRatios, TopologyAxes
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget
from tilefoundry_ascend.ascend.target.dialect_surface import get_dialect_surface
from tilefoundry_ascend.ascend.templates.registry import ConstraintRelaxationMask, relax_facets_for_bottleneck

CREATIVE_FOCUS_HINTS: tuple[str, ...] = ("memory_hierarchy_and_tiling", "compute_pipeline_and_sync", "cross_core_dataflow_and_fusion")

TILELANG_ASCEND_PRIMITIVE_MENU: str = "\n".join((
    "- T.MixedKernel(28, sids=2): coupled 1 AIC + 2 AIV execution (or pure-AIV sids=2)",
    "- T.Kernel(blocks) / T.MixedKernel(blocks, sids=2): 1D core grid only (NEVER pass is_npu=... or threads=...)",
    "- T.Persistent(...): persistent 2D core swizzle loop over logical tiles",
    "- T.Pipelined(count, num_stages=...): multi-buffered loop pipelining",
    "- T.SimdVF(): native SIMD vector-function mode (256 KiB UB with PURE_SIMDVF_UBUF_FLAGS; T.Parallel must index plain UB tile buffers rather than pipelined version-counter modulo slices)",
    "- T.SimtVF(threads=128): SIMT vector-function mode for irregular/scalar-tail loops",
    "- Module pre-imports tilelang, tilelang.ascend.language as T, torch, PURE_SIMDVF_UBUF_FLAGS (do NOT import tilelang.language as T or call T.Scope)",
    "- Inside T.MixedKernel(28, sids=2) as (bx, sid), AIC Cube ops (T.gemm, L1/L0C T.copy) and AIV Vector blocks (with T.SimdVF(): ...) coexist directly without T.Scope",
    "- T.gemm(a, b, c, ...): Cube MMAD on L1/L0A/L0B -> L0C",
    "- T.copy(src, dst) / T.dual_copy(src, dst, ...): GM/L1/L0C/UB DMA and 1C2V split copy",
    "- T.alloc_l1 / T.alloc_l0a / T.alloc_l0b / T.alloc_l0c: AIC hierarchical SRAM buffers",
    "- T.alloc_shared(...) / T.alloc_ss(...): AIV Unified Buffer (UB) and 3 KiB shared SSBUF",
    "- T.set_atomic(...): atomic reduction/accumulation mode on GM outputs",
    "- PURE_SIMDVF_UBUF_FLAGS: unlock 256 KiB UB via _select_*_config()[\"pass_configs\"]/\"compile_flags\" or PrimFunc.attrs",
    "- Segmented GM DMA burst efficiency: <16 KiB bursts lose HBM bandwidth (~936 GB/s at 4 KiB vs ~1.64 TB/s at 32 KiB); size pure-vector row tiles to minimize waves * tile_rows / effective_burst_bw and clamp tail start min(t * TT, M - TT) when writes are idempotent",
    "- Static ping-pong double buffering (buf_a/buf_b 2x unroll + cross-tile head prefetch + barrier pruning): avoids enable_offset dynamic indexing pointer-alias serialization in backend codegen",
    "- Two-pass loop fission: split vcadd/vsqrt/vexp score pass into compact UB weights from pure wide vfma accumulation pass to hide SFU latency chains",
))

_RULE_CONCEPT_DESCRIPTIONS: dict[str, str] = {
    "R01_DUAL_COPY_1C2V_SPLIT": "1C2V row-splitting via dual_copy across paired AIV sub-cores",
    "R02_ONCHIP_L0C_TO_UB_DIRECT": "Direct on-chip L0C-to-UB and UB-to-L1 datapath bypass",
    "R03_ACCUM_SPAN_FIXPIPE_RELIEF": "Inner-K accumulation span enlargement for FixPipe drain amortization",
    "R04_MTE2_SPARSE_GATHER_COALESCE": "Coalesced 32B-aligned MTE2 block transfers and UB vector gather",
    "R05_SSBUF_RING_METADATA_HANDOFF": "Shared-scalar SSBUF cross-core metadata handoff",
    "R06_ONLINE_SOFTMAX_SKIP_SCALE": "Guarded online reduction skip-scale and deferred epilogue normalization",
    "R07_IMPLICIT_BLOAT_HEADROOM_TRIM": "Tile blocking trim for implicit compiler scratchpad headroom",
    "R08_PERSISTENT_SWIZZLE_L2_LOCALITY": "Persistent 2D block swizzling for L2 cache locality",
    "R09_MIXED_QUANT_DEQUANT_STAGE_OVERLAP": "Pipelined AIV dequantization staged ahead of Cube MMAD",
    "R10_BACKWARD_SYNC_CV_PINGPONG": "Dual-slot ping-pong parity sync decoupling AIC and AIV",
    "R11_L2_NZ_PAD_BANK_CONFLICT_AVOID": "NZ fractal stride padding for SRAM bank conflict avoidance",
    "R12_UNIT_FLAG_FIXPIPE_CHAIN_FUSION": "Fine-grained hardware unit_flag handoff between MMAD and FixPipe",
    "R13_SPLIT_KV_WAVE_RECOVERY": "Grid block splitting across paired sub-cores for wave occupancy recovery",
    "R14_GQA_HEAD_M_FOLDING": "Query-group head folding into Cube M-dimension for fractal utilization",
    "R15_ASYMMETRIC_1C2V_STAGE_FISSION": "Asymmetric 1C2V stage fission and pipelined UB footprint compression",
    "R16_PADDED_PITCH_TAIL_MASKING": "32B-aligned UB pitch padding and SIMD tail masking for atom-aligned non-32B-multiple rows",
    "R17_RMEM_ACCUM_AND_256KB_UB_UNLOCK": "Register-resident loop accumulation and full 256 KiB UB unlock",
    "R17_RMEM_ACCUMULATOR_HOIST": "Register-resident loop accumulation hoisting",
    "R18_WAVE_QUANTIZED_BURST_TILING": "Wave-quantized pure-vector tiling co-selected with HBM DMA burst curve",
    "R19_STATIC_PINGPONG_UNROLL_NO_ALIAS": "Static 2x ping-pong buffer unrolling and prefetch to avoid dynamic-offset aliasing",
    "R20_LOOP_FISSION_REDUCE_SFU_HIDING": "Two-pass loop fission separating reduce/SFU chains from wide vfma accumulation",
    "R21_SUBROW_DEPTH_OR_WIDTH_CORE_SHARDING": "Sub-row reduction depth or 64-element hidden-width core sharding across AIVs",
    "R22_HALF_L1_BANK_ISOLATION": "Half-L1 ping/pong bank isolation ((TOTAL_L1_SIZE >> 1) * slot) across 2x256KB L1 banks",
    "R23_UB_16BANK_STRIDE_PAD_AND_SHIFT": "UB 16-bank conflict avoidance via +1 DataBlock row stride pad and +8 DataBlock R/W base shift",
    "R24_VLOOPV2_LEGALIZATION_AND_DIRECT_VF": "VLOOPv2 hardware loop legalization (uint16_t 0-based +1, branchless Select) and direct __simd_vf__",
    "R25_SCALAR_BOUND_SPILL_ELIMINATION": "Eliminate scalar register/stack spills via local scalar binding and 3-stage prologue/loop/epilogue",
    "R26_SWAT_SNAKE_AND_TAIL_WAVE_SPLIT": "SWAT serpentine tile swizzle (WINDOW_LEN=4) and tail-wave sub-tile / DPSK Stream-K rebalance",
    "R26_UB_256KB_SPR_UNLOCK": "Full 256 KiB UB unlock via SPR / PURE_SIMDVF_UBUF_FLAGS",
    "R27_L2_CACHE_BYPASS_SINGLE_USE": "Bypass L2 allocation (CACHE_MODE_DISABLE) on single-use operands when M_tiles==1 or streaming KV",
    "R28_INPLACE_L1_SLOT_ALIASING": "In-place L1/UB slot aliasing reusing dead slices (e.g. K_rope -> P) across stages",
    "R29_W4_AIV_UNPACK_AND_FIXPIPE_SCALE_FOLD": "AIV INT4/FP4 unpack via DIST_UNPACK4_B8 or bit-shift + FixPipe deq_scale=64 fold on dav-3510",
    "R29_W4_AIV_SUBBYTE_UNPACK": "AIV INT4/FP4 sub-byte unpack via DIST_UNPACK4_B8 or bit-shift",
    "R30_REGBASE_MX_QUANT_DATABLOCK": "RegBase microscaling MXFP8/MXFP4 quant via DIST_DINTLV_B16, ReduceDataBlock<MAX>, and SetCtrlSpr",
    "R31_SIMT_UB_PRIVATIZED_ATOMICS": "SIMT warp-privatized +8-padded UB atomics with __launch_bounds__(512) 0-spill and Simt::UintDiv",
    "R32_GROUPED_ROLLING_CORE_AND_EMPTY_SKIP": "Grouped MatMul rolling core assignment ((endBlockIdx+1)%coreNum) with m==0 empty-group fast skip",
    "R32_GROUPED_ROLLING_CORE_DISPATCH": "Grouped MatMul rolling core dispatch ((endBlockIdx+1)%coreNum)",
    "R33_PERSISTENT_STATE_RESIDENCY": "Persistent on-chip UB/L1 FP32 state residency across chunks/iterations with initC=false accumulation",
    "R34_SCALAR_LOOP_TO_VECTOR_DMA_PIPELINE": "Refactor raw scalar global-memory loops into UB/L1 DMA staging plus SIMD vector or Cube pipelines",
    "R35_PURE_VECTOR_DUAL_SUBCORE_SPLIT": "Pure-vector dual-subcore split via T.MixedKernel(blocks, sids=2) across paired AIVs",
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

_TIR_SCOPE_CONTRACT = (
    "### TIR SCOPE CONTRACT\n"
    "- Never nest `def` (or `async def`) inside `@T.prim_func` — define any helper "
    "function at outer module or builder scope with explicit parameters.\n"
    "- Do NOT redefine read-only host tail bindings (`kernel`, `ref_kernel`, "
    "`build_kernel`, `_ENTRY_CACHE`); the host splicer preserves `<read_only_host_tail>` verbatim.\n\n"
)


def _describe_rule_concept(rule: AscendCausalRule) -> str:
    if rule.rule_id in _RULE_CONCEPT_DESCRIPTIONS:
        return _RULE_CONCEPT_DESCRIPTIONS[rule.rule_id]
    if desc := (str(getattr(rule, "concept_description", "") or "").strip() or str(rule.rationale or "").strip()):
        return desc
    low = rule.rule_id.lower()
    for needles, text in (
        (("dual_copy", "1c2v"), "1C2V row-splitting via dual_copy across paired AIV sub-cores"),
        (("ssbuf",), "Shared-scalar SSBUF cross-core handoff"),
        (("pipeline", "stage"), "Multi-stage pipeline buffering and overlap"),
        (("coalesce", "mte2"), "Coalesced contiguous MTE2 DMA transfers"),
        (("core", "parallel"), "Multi-core block concurrency partitioning"),
        (("pad", "align"), "Fractal alignment and SRAM stride padding"),
        (("barrier", "sync", "flag"), "Fine-grained cross-pipe synchronization"),
    ):
        if any(n in low for n in needles):
            return text
    return "Previously matched bottleneck transformation"


def summarize_excluded_mechanisms(
    matched_rules: Sequence[AscendCausalRule], exploited_signatures: Iterable[tuple[str, ...]]
) -> tuple[str, ...]:
    """Derive rule-blind generic mechanism descriptions for negative constraints."""
    mechanisms: list[str] = []
    seen: set[str] = set()
    for rule in matched_rules:
        if (desc := _describe_rule_concept(rule)) not in seen:
            seen.add(desc)
            mechanisms.append(desc)
    for sig in sorted(exploited_signatures):
        for token in sig:
            if (desc := _SIGNATURE_DESCRIPTIONS.get(token)) and desc not in seen:
                seen.add(desc)
                mechanisms.append(desc)
    return tuple(mechanisms)


def build_prescreen_summary_from_kwargs(
    kwargs: Mapping[str, object] | None, *, dsl_source: str | None = None, factory_name: str | None = None,
    var_bindings: Mapping[str, int] | None = None, config_selector_name: str | None = None,
    dims: Mapping[str, object] | None = None, pass_configs: Mapping[str, object] | None = None,
    compile_flags: Sequence[str] | str | None = None, cost_model: HybridAscendCostModel | None = None,
    axes: TopologyAxes | None = None, live_plan: object | None = None, blueprint: object | None = None,
    sil_advisory: object | None = None, ordered_tensor_shapes: Sequence[Sequence[int]] | None = None,
    case_entry: Mapping[str, object] | None = None, proto: Mapping[str, object] | None = None,
) -> str:
    """Build live-IR or analytical dual-accumulator tile prescreen summary."""
    if not kwargs:
        return ""
    lowered, lowered_dims = {str(k).lower(): v for k, v in kwargs.items()}, {str(k).lower(): v for k, v in (dims or {}).items()}
    pick_pos = lambda keys: next((v for src in (lowered, lowered_dims) for k in keys if isinstance(v := src.get(k), int) and not isinstance(v, bool) and v > 0), None)
    bm_val, bn_val, bk_val = pick_pos(("bm", "tile_m")), pick_pos(("bn", "tile_n")), pick_pos(("bk", "tile_k"))
    if bm_val is None and bn_val is None and bk_val is None:
        return ""
    dtype_str = str(lowered.get("tl_dtype", lowered.get("dtype", "bfloat16"))).lower()
    d_out_val, gemm_count = pick_pos(("d_out", "d_nope", "d_v")) or 0, (count_dsl_gemm_calls(dsl_source) if dsl_source else 0)
    center = TileConfigCandidate(
        bm=bm_val or 64, bn=bn_val or 64, bk=bk_val or 64, stages=pick_pos(("stages", "num_stages")) or 2,
        dtype_bytes=4 if ("float32" in dtype_str or dtype_str == "f32") else 2,
        d_out=d_out_val, dual_accumulator=bool(gemm_count >= 2 or d_out_val > 0),
    )
    if dsl_source and factory_name:
        resolved_shapes = extract_ordered_tensor_shapes(case_entry=case_entry, dims=dims, proto=proto, ordered_tensor_shapes=ordered_tensor_shapes)
        try:
            if live_summary := prescreen_live_ir_neighborhood(
                dsl_source, factory_name=factory_name, center=center, base_kwargs=kwargs, var_bindings=var_bindings,
                config_selector_name=config_selector_name, pass_configs=pass_configs, compile_flags=compile_flags,
                dims=dims, cost_model=cost_model, axes=axes, live_plan=live_plan, blueprint=blueprint,
                sil_advisory=sil_advisory, ordered_tensor_shapes=resolved_shapes,
            ):
                return live_summary
        except Exception:
            pass
    if gemm_count == 0 and d_out_val == 0:
        return ""
    center_score = evaluate_tile_candidate(center, AscendTarget("ascend.950pr").get_facts(AscendSpaceFacts))
    pool = list(generate_aligned_neighborhood(center))
    if not center_score.is_feasible and center.bm >= 64:
        step = 16 * max(1, center.aiv_split)
        if (half_bm := ((center.bm // 2) // step) * step) >= 16 and half_bm != center.bm:
            for bn_cand in dict.fromkeys((center.bn, max(16, center.bn // 2))):
                if bn_cand % 16 == 0:
                    pool.append(TileConfigCandidate(
                        bm=half_bm, bn=bn_cand, bk=center.bk, stages=center.stages, dtype_bytes=center.dtype_bytes,
                        accum_bytes=center.accum_bytes, aiv_split=center.aiv_split, d_out=center.d_out, dual_accumulator=center.dual_accumulator,
                    ))
    display_scores = list(prescreen_tile_configs(pool, top_k=4)) + ([] if center_score.is_feasible else [center_score])
    return format_prescreen_summary(display_scores) if display_scores else ""


@dataclass(frozen=True)
class WorkerSpec:
    """Spawned worker specification for one optimization portfolio round."""

    worker_id: str
    worker_kind: str
    rule: AscendCausalRule | None
    request: GenerationRequest
    parent_candidate: BeamCandidate
    ideal_roofline_us: float = 0.0
    roofline: RooflineSummaryLike | Mapping[str, object] | None = None


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
class RoundEscalationReport(JsonRecordMixin):
    """Closed-loop escalation, minimal facet relaxation, and cost prediction for a round."""

    escalation: EscalationDecision
    relaxed_mask: ConstraintRelaxationMask
    cost_prediction: CostPrediction
    roofline_eta: float
    l1_overflow: bool = False
    ub_overflow: bool = False


def _coerce_topology_axes_opt(raw: object | None) -> TopologyAxes | None:
    if isinstance(raw, TopologyAxes):
        return raw
    if isinstance(raw, Mapping) and all(k in raw for k in ("compute_engine", "memory_ring", "iteration_regularity")):
        return TopologyAxes.from_dict(raw)
    return None


def _project_features_with_ratios(features: Mapping[str, object] | None, case_ratios: CasePhysicsRatios | None) -> dict[str, object] | None:
    if features is None:
        return None
    if case_ratios is None:
        return features if isinstance(features, dict) else {str(k): v for k, v in features.items()}
    return {
        **{str(k): v for k, v in features.items()},
        "inner_contiguous_bytes": case_ratios.inner_contiguous_bytes,
        "persistent_wave_occupancy": case_ratios.wave_occupancy_ratio, "wave_occupancy_ratio": case_ratios.wave_occupancy_ratio,
        "contig_atom_ratio": case_ratios.contig_atom_ratio, "oi_ridge_ratio": case_ratios.oi_ridge_ratio,
        "has_empty_group": case_ratios.has_empty_group,
    }


def _derive_active_topology_axes(
    ledger: object | None, *, features: object | None = None, sil_advisory: object | None = None,
    route: object | None = None, topology_axes: object | None = None, case_ratios: CasePhysicsRatios | None = None,
) -> TopologyAxes:
    axes = _coerce_topology_axes_opt(topology_axes)
    for src in (sil_advisory, route):
        if axes is None and src is not None:
            axes = _coerce_topology_axes_opt(src.get("topology_axes") if isinstance(src, Mapping) else getattr(src, "topology_axes", None))
    if axes is None:
        if sil_advisory is not None or features is not None:
            axes = resolve_canonical_topology_axes(sil_advisory, features=features)
        else:
            pc = getattr(ledger, "pipe_cycles", None) or {}
            is_aic = isinstance(pc, Mapping) and (pc.get("AIC.PIPE_M", 0) > 0 or pc.get("AIC.CUBE", 0) > 0)
            axes = TopologyAxes(compute_engine="aic_only" if is_aic else "aiv_only", memory_ring="gm_l1_l0_fixpipe" if is_aic else "gm_ub_stream", iteration_regularity="affine_simd")
    return case_ratios.effective_axes(axes) if case_ratios is not None else axes


def _to_mapping(obj: object) -> Mapping[str, object]:
    return obj if isinstance(obj, Mapping) else (obj.to_dict() if hasattr(obj, "to_dict") else vars(obj))


def format_sil_diagnostics_section(*, sil_advisory: object | None = None, sil_verification: object | None = None) -> str:
    """Format SIL AdvisoryEnvelope and VerificationReport diagnostics for prompts."""
    sections: list[str] = []
    if sil_advisory is not None:
        adv = _to_mapping(sil_advisory)
        lines = [
            "### SIL ADVISORY ENVELOPE",
            f"- op_name: {adv.get('op_name', '')} (task={adv.get('cannbot_task', '')}, level={adv.get('level', '')})",
            f"- topology_axes: {adv.get('topology_axes', '')}",
            f"- dtypes: in={adv.get('input_dtypes', ())}, out={adv.get('output_dtypes', ())}",
            f"- has_nan_inf_cases: {bool(adv.get('has_nan_inf_cases', False))}, has_inplace_aliasing: {bool(adv.get('has_inplace_aliasing', False))}",
        ]
        if isinstance(b := adv.get("legality_bounds"), Mapping):
            lines.append(f"- dav3510_bounds: L1={b.get('l1_bytes')}, UB_simdvf={b.get('ub_simdvf_bytes')}, UB_simtvf={b.get('ub_simtvf_bytes')}, forbidden_dma={b.get('forbidden_dma_paths')}, forbid_cube_int4={b.get('forbid_cube_int4')}")
        if isinstance(rf := adv.get("roofline"), Mapping) and rf:
            lines.append(f"- roofline: {json.dumps(rf, sort_keys=True)}")
        if isinstance(regimes := adv.get("regimes"), (list, tuple)) and regimes:
            lines.append(f"- stratified_regimes: {len(regimes)}")
            for reg in regimes:
                if isinstance(reg, Mapping):
                    tag = str(reg.get("regime_tag", ""))
                    cnums = [int(c) for c in (reg.get("case_nums") or ())] if isinstance(reg.get("case_nums"), (list, tuple)) else []
                    ratios = tuple(r for r in (reg.get("case_ratios") or ()) if isinstance(r, Mapping)) if isinstance(reg.get("case_ratios"), (list, tuple)) else ()
                    rep_id = reg.get("representative_case_id")
                    sel = next((r for r in ratios if r.get("case_id") == rep_id), None) if (ratios and isinstance(rep_id, int) and not isinstance(rep_id, bool)) else (ratios[len(ratios) // 2] if ratios else None)
                    lines.append(f"- regime {tag} cases={cnums} CasePhysicsRatios: unavailable" if sel is None else f"- regime {tag} cases={cnums} CasePhysicsRatios: {json.dumps(dict(sel), sort_keys=True)}")
        sections.append("\n".join(lines))
    if sil_verification is not None:
        ver = _to_mapping(sil_verification)
        lines = ["### SIL VERIFICATION DIAGNOSTICS", f"- passed: {bool(ver.get('passed', True))}"]
        if errors := ver.get("errors") or ():
            lines.extend(("Errors:", *(f"- {e}" for e in errors)))
        if warnings := ver.get("warnings") or ():
            lines.extend(("Warnings:", *(f"- {w}" for w in warnings)))
        if fixes := ver.get("suggested_fixes") or {}:
            lines.append(f"Suggested Fixes: {json.dumps(fixes, sort_keys=True)}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _resolve_constraint_section(
    parent: BeamCandidate, blueprint_context: str, blueprint: HirBlueprintPacket | None,
    *, escalation_tier: str = "tilelang_dsl", enable_track_router: bool = False,
    sil_advisory: object | None = None, sil_verification: object | None = None,
    freedom_mask: ConstraintRelaxationMask | None = None, excluded_mechanisms: Sequence[str] = (),
    active_symptom: str | None = None,
) -> str:
    enable_router = bool(enable_track_router or (parent.track and parent.track not in ("", "unconstrained")))
    rendered = build_progressive_llm_constraints(
        escalation_tier=escalation_tier, track=parent.track, enable_track_router=enable_router,
        sil_advisory=sil_advisory, sil_verification=sil_verification, freedom_mask=freedom_mask,
        ub_256kb_unlock=bool(parent.ledger.uses_unreserved_ubuf), blueprint=blueprint,
        current_ledger=parent.ledger, excluded_mechanisms=excluded_mechanisms,
        extra_blueprint_context=blueprint_context, topology_axes=parent.topology_axes,
        active_symptom=active_symptom, case_ratios=parent.case_ratios,
    )
    return f"{rendered}\n\n" if rendered else ""


def _unified_reply_contract(fence_lang: str) -> str:
    delta = (
        " For focused edits inside an existing function, prefer minimal `<<<<<<< SEARCH` / `=======` / "
        "`>>>>>>> REPLACE` hunks containing exact verbatim lines from `CURRENT KERNEL SOURCE` "
        "(never touching `<read_only_host_tail>` or sibling regimes), OR output ONLY the modified "
        "top-level `def ...` functions (omitting unchanged functions/helpers)."
        if fence_lang == "python" else ""
    )
    return (
        f"Output ONLY up to 2 leading metadata comment lines followed immediately by one fenced ```{fence_lang} "
        f"code block (no JSON wrapper, no extra prose):{delta}\n"
        "# target_bottleneck: <BOTTLENECK_NAME>\n# structural_hypothesis: <concise hypothesis>"
    )


def _format_source_and_readonly_tail(source: str, *, fence_lang: str) -> str:
    if fence_lang == "python":
        editable, readonly_tail = split_tracka_editable_and_readonly_tail(source)
        if editable.strip() and readonly_tail:
            return f"### CURRENT KERNEL SOURCE\n```python\n{editable.strip()}\n```\n\n<read_only_host_tail>\n{readonly_tail}\n</read_only_host_tail>\n"
    return f"### CURRENT KERNEL SOURCE\n```{fence_lang}\n{source.strip()}\n```\n"


def _ledger_residual_sections(parent: BeamCandidate) -> str:
    return (
        f"### OBJECTIVE PHYSICAL LEDGER\n{parent.ledger.to_json()}\n\n"
        f"### STATIC-DYNAMIC RESIDUAL\n{json.dumps(parent.residual.to_dict(), indent=2)}\n\n"
    )


def build_exploit_request(
    parent: BeamCandidate, rule: AscendCausalRule, *, blueprint_context: str = "",
    blueprint: HirBlueprintPacket | None = None, escalation_tier: str = "tilelang_dsl",
    enable_track_router: bool = False, sil_advisory: object | None = None,
    sil_verification: object | None = None, freedom_mask: ConstraintRelaxationMask | None = None,
) -> GenerationRequest:
    """Build causal-rule worker request under the 3-layer progressive constraint stack."""
    fence_lang = "cpp" if get_dialect_surface(escalation_tier).dialect_id == "cce_cpp20" else "python"
    sys_prompt = (
        "You are an Ascend 950PR kernel optimization engineer. Apply the given causal rule faithfully "
        "to eliminate the bottleneck identified in the physical ledger while preserving exact numerical "
        f"equivalence. {_unified_reply_contract(fence_lang)}"
    )
    constraint_section = _resolve_constraint_section(
        parent, blueprint_context, blueprint, escalation_tier=escalation_tier,
        enable_track_router=enable_track_router, sil_advisory=sil_advisory,
        sil_verification=sil_verification, freedom_mask=freedom_mask,
        active_symptom=rule.symptom_family.value if hasattr(rule.symptom_family, "value") else str(rule.symptom_family),
    )
    user_prompt = (
        f"{constraint_section}{_TIR_SCOPE_CONTRACT if fence_lang == 'python' else ''}"
        f"### SELECTED CAUSAL RULE\nRule ID: {rule.rule_id}\nPrecondition: {rule.precondition}\n"
        f"Transform Template:\n{rule.transform_template.strip()}\nExpected Postcondition: {rule.expected_postcondition}\n\n"
        f"{_ledger_residual_sections(parent)}{_format_source_and_readonly_tail(parent.dsl_source, fence_lang=fence_lang)}"
    )
    return GenerationRequest(role="exploit_rule", system_prompt=sys_prompt, user_prompt=user_prompt)


def build_creative_request(
    parent: BeamCandidate, *, excluded_mechanisms: Sequence[str] = (), focus_hint: str = "",
    prescreen_summary: str = "", blueprint_context: str = "", blueprint: HirBlueprintPacket | None = None,
    escalation_tier: str = "tilelang_dsl", enable_track_router: bool = False, effort: str | None = None,
    sil_advisory: object | None = None, sil_verification: object | None = None,
    freedom_mask: ConstraintRelaxationMask | None = None, focus_lines: Sequence[int] = (),
) -> GenerationRequest:
    """Build rule-blind creative exploration request under 3-layer constraints."""
    is_trackb = get_dialect_surface(escalation_tier).dialect_id == "cce_cpp20"
    fence_lang = "cpp" if is_trackb else "python"
    sys_prompt = (
        "You are a first-principles Ascend 950PR (dav-3510) systems architect. Propose an architectural "
        "kernel optimization directly from hardware limits (1 AIC + 2 AIV coupled block, 3 KiB SSBUF, "
        "L0C->UB direct path, UB->L1 direct path, 16-entry MTE2 queue, 128 B/cycle FixPipe) and the "
        f"physical telemetry below. {_unified_reply_contract(fence_lang)}"
    )
    constraint_section = _resolve_constraint_section(
        parent, blueprint_context, blueprint, escalation_tier=escalation_tier,
        enable_track_router=enable_track_router, sil_advisory=sil_advisory,
        sil_verification=sil_verification, freedom_mask=freedom_mask, excluded_mechanisms=excluded_mechanisms,
    )
    scoped_src = compact_creative_source_slice(parent.dsl_source, focus_lines=focus_lines)
    user_prompt = (
        f"{constraint_section}{_TIR_SCOPE_CONTRACT if fence_lang == 'python' else ''}"
        f"{f'### ARCHITECTURAL FOCUS HINT{chr(10)}{focus_hint}{chr(10)}{chr(10)}' if focus_hint else ''}"
        f"{f'### TILE PRESCREEN GUIDANCE{chr(10)}{prescreen_summary}{chr(10)}{chr(10)}' if prescreen_summary else ''}"
        f"{_ledger_residual_sections(parent)}{_format_source_and_readonly_tail(scoped_src, fence_lang=fence_lang)}"
    )
    return GenerationRequest(
        role="escalate_trackb" if is_trackb else "explore_creative",
        system_prompt=sys_prompt, user_prompt=user_prompt, effort=effort or "high",
    )


def extract_lowered_compiler_excerpt(stage_b: object, *, window: int = 12, max_chars: int = 2400) -> str:
    """Slice raw lowered CCE/TIR lines around reported :line: numbers or leading 24 lines."""
    cce_src = str(getattr(stage_b, "generated_cce", "") or "")
    has_cce = bool(cce_src.strip())
    raw_src = cce_src if has_cce else str(getattr(stage_b, "lowered_tir", "") or "")
    if not raw_src.strip() or not (lines := raw_src.splitlines()):
        return ""
    diag = str(getattr(stage_b, "error_message", "") or "")
    pats = (r"\.cce:(\d+)(?::\d+)?\b", r":(\d+)(?::\d+)?\b") if has_cce else (r":(\d+)(?::\d+)?\b",)
    line_nums: list[int] = []
    for pat in pats:
        line_nums = [int(m.group(1)) for m in re.finditer(pat, diag) if 1 <= int(m.group(1)) <= len(lines)]
        if line_nums:
            break
    if line_nums:
        merged: list[list[int]] = []
        for ln in sorted(set(line_nums)):
            s, e = max(1, ln - window), min(len(lines), ln + window)
            if merged and s <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        excerpt_lines: list[str] = []
        for idx, (s, e) in enumerate(merged):
            if idx > 0:
                excerpt_lines.append("...")
            excerpt_lines.extend(f"{ln}: {lines[ln - 1]}" for ln in range(s, e + 1))
        excerpt = "\n".join(excerpt_lines)
    else:
        excerpt = "\n".join(f"{ln}: {lines[ln - 1]}" for ln in range(1, min(len(lines), 24) + 1))
    if len(excerpt) > max_chars:
        trunc = "\n/* ... lowered excerpt truncated ... */"
        return excerpt[: max(0, max_chars - len(trunc))].rstrip() + trunc
    return excerpt


def extract_lowered_compiler_excerpt_with_lang(
    stage_b: object | None, *, window: int = 12, max_chars: int = 2400
) -> tuple[str, str | None]:
    """Return bounded lowered excerpt and its fence language from StageBResult."""
    if stage_b is None:
        return "", None
    has_cce = bool(str(getattr(stage_b, "generated_cce", "") or "").strip())
    has_tir = bool(str(getattr(stage_b, "lowered_tir", "") or "").strip())
    return extract_lowered_compiler_excerpt(stage_b, window=window, max_chars=max_chars), (
        "cpp" if has_cce else ("python" if has_tir else None)
    )


def build_syntax_repair_request(
    dsl_source: str, error_message: str, *, slice_hint: str = "", sil_advisory: object | None = None,
    sil_verification: object | None = None, backend_kind: str = "tilelang_dsl", parent_source: str = "",
    freedom_mask: ConstraintRelaxationMask | None = None, vector_mode: str = "simdvf",
    ub_256kb_unlock: bool = False, topology_axes: TopologyAxes | None = None,
    case_ratios: CasePhysicsRatios | None = None, lowered_excerpt: str = "",
    lowered_excerpt_lang: str | None = None,
) -> GenerationRequest:
    """Build bounded syntax/compile repair request routed to the active backend dialect."""
    hint_sentence = f" Authorized target: {slice_hint}." if slice_hint else ""
    surface = get_dialect_surface(backend_kind)
    is_cpp = (surface.dialect_id == "cce_cpp20") and not _looks_like_python_primfunc_slice(dsl_source)
    eff_tier = "tilelang_dsl" if (surface.dialect_id == "cce_cpp20" and not is_cpp) else backend_kind
    eff_surface = get_dialect_surface(eff_tier)
    fence_lang = "cpp" if is_cpp else "python"
    if is_cpp:
        sys_prompt = (
            "You are a Track-B CCE (C++20) kernel repair engineer for the tl_templates/ascend surface. "
            "Fix the syntax, API, sync-pairing, or SRAM-capacity failure reported below without altering tensor semantics. "
            "Output ONLY valid C++20 SEARCH/REPLACE hunks or authorized kernel code matching the target regime contract. "
            f"Do NOT emit high-level AscendC classes or legacy non-dav-3510 CCE intrinsics.{hint_sentence}"
        )
    else:
        sys_prompt = (
            "You are a TileLang Ascend compiler repair engineer. Fix the syntax, API, or memory-capacity failure reported below "
            "without altering tensor semantics. Output ONLY corrected Python code. For focused fixes inside an existing function, "
            "prefer minimal `<<<<<<< SEARCH` / `=======` / `>>>>>>> REPLACE` hunks containing exact verbatim lines from `BROKEN KERNEL SOURCE` "
            "(never touching `<read_only_host_tail>` or sibling regimes), OR output ONLY the fixed subset of authorized slice functions "
            "or prefixed macros shown in BROKEN KERNEL SOURCE (omitted sibling functions automatically inherit their parent implementation); "
            f"any new top-level helper MUST reuse the exact regime prefix shown in BROKEN KERNEL SOURCE (do NOT add unprefixed or out-of-regime functions).{hint_sentence}"
        )

    layer1 = format_hardware_and_sil_layer(sil_advisory=sil_advisory, sil_verification=sil_verification, freedom_mask=freedom_mask, vector_mode=vector_mode, ub_256kb_unlock=ub_256kb_unlock)
    layer2 = format_target_dialect_layer(escalation_tier=eff_tier, topology_axes=topology_axes, freedom_mask=freedom_mask, case_ratios=case_ratios)
    sil_text = format_sil_diagnostics_section(sil_advisory=sil_advisory, sil_verification=sil_verification)
    closure = extract_repair_closure_slice(dsl_source, backend_kind=eff_tier, diagnostic_text=error_message)
    broken_src = closure.editable_device_source or dsl_source.strip()
    parent_block = ""
    if stripped_parent := parent_source.strip():
        p_closure = extract_repair_closure_slice(stripped_parent, backend_kind=eff_tier, diagnostic_text=error_message)
        parent_block = f"### PARENT AUTHORIZED SOURCE\n```{fence_lang}\n{p_closure.editable_device_source or stripped_parent}\n```\n\n"
    sibling_block = f"### READ-ONLY SIBLING PROTOTYPES\n{chr(10).join(closure.sibling_prototypes)}\n\n" if closure.sibling_prototypes else ""
    abi_tail_block = ""
    if closure.readonly_host_abi_tail:
        abi_tail_block = (
            f"### HOST-DEVICE ABI SLICE (READ-ONLY)\n```{'python' if 'def ' in closure.readonly_host_abi_tail else 'cpp'}\n{closure.readonly_host_abi_tail}\n```\n\n"
            if is_cpp else f"<read_only_host_tail>\n{closure.readonly_host_abi_tail}\n</read_only_host_tail>\n\n"
        )
    lowered_block = ""
    if (s_low := lowered_excerpt.strip()) and s_low not in dsl_source.strip() and s_low not in broken_src:
        efence = str(lowered_excerpt_lang) if lowered_excerpt_lang in ("cpp", "python") else ("python" if ("@T.prim_func" in s_low or "I.ir_module" in s_low or ("def " in s_low and "#include" not in s_low)) else "cpp")
        lowered_block = f"### COMPILER LOWERED EXCERPT (READ-ONLY CONTEXT)\n```{efence}\n{s_low}\n```\n\n"
    user_prompt = (
        f"{layer1}\n\n### VALID TARGET DIALECT PRIMITIVES ({eff_surface.dialect_id})\n{layer2}\n\n"
        f"{'' if is_cpp else _TIR_SCOPE_CONTRACT}{format_g5_parent_monotonicity_lock()}\n\n"
        f"{f'{sil_text}{chr(10)}{chr(10)}' if sil_text else ''}{parent_block}### COMPILER / GATE DIAGNOSTIC\n{error_message}\n\n"
        f"{lowered_block}{sibling_block}{abi_tail_block}### BROKEN KERNEL SOURCE\n```{fence_lang}\n{broken_src}\n```\n"
    )
    return GenerationRequest(role="syntax_repair", system_prompt=sys_prompt, user_prompt=user_prompt)


def _resolve_evaluation_failure_code(evaluation: FunnelEvaluation) -> str:
    if direct := str(getattr(evaluation, "fault_code", "") or "").strip():
        return direct
    if evaluation.failed_stage == "stage_b" and evaluation.stage_b is not None:
        if sb_code := str(getattr(evaluation.stage_b, "fault_code", "") or "").strip():
            return sb_code
    return str(getattr(evaluation.stage_a, "fault_code", "") or "").strip()


class PortfolioRunner:
    """Runs K=2 Rule Exploit + 1 Rule-Blind Creative worker with syntax repair and KB feedback."""

    def __init__(
        self, *, backend: KernelLLMBackend, funnel: GateFunnel, tracker: BayesianRuleTracker,
        beam: TwoLaneBeamSearch | None = None, k_rule_workers: int = 2, max_syntax_repairs: int = 1,
        enable_track_router: bool = False, enable_tilesize_prescreen: bool = True,
        cost_model: HybridAscendCostModel | None = None, cost_model_mode: str = "off",
    ) -> None:
        self._backend, self._funnel, self._tracker = backend, funnel, tracker
        self._beam = beam or TwoLaneBeamSearch()
        self._k_rules, self._max_repairs = max(1, k_rule_workers), max(0, max_syntax_repairs)
        self._enable_track_router = enable_track_router
        self._enable_tilesize_prescreen = bool(enable_tilesize_prescreen)
        self._cost_model = cost_model
        self._cost_model_mode = str(cost_model_mode).strip().lower() or "off"
        self._pending_speculative_creative: (
            asyncio.Task[tuple[BeamCandidate | None, bool, tuple[str, str, str] | None, bool]] | None
        ) = None

    def cancel_pending_speculative(self) -> None:
        """Cancel any in-flight speculative creative task and clear the pending slot."""
        pending = self._pending_speculative_creative
        self._pending_speculative_creative = None
        if pending is not None and not pending.done():
            pending.cancel()

    @property
    def beam(self) -> TwoLaneBeamSearch:
        return self._beam

    def _record_rule_failure(self, rule: AscendCausalRule | None, axes: TopologyAxes) -> None:
        if rule is not None:
            self._tracker.record_outcome(rule.rule_id, postcondition_realized=False, topology_axes=axes)

    def plan_workers(
        self, parent: BeamCandidate, round_idx: int, *, factory_name: str | None = None,
        factory_kwargs: Mapping[str, object] | None = None, var_bindings: Mapping[str, int] | None = None,
        config_selector_name: str | None = None, dims: Mapping[str, object] | None = None,
        blueprint_context: str = "", ideal_roofline_us: float = 0.0, blueprint: HirBlueprintPacket | None = None,
        creative_effort: str = "high", sil_advisory: object | None = None, sil_verification: object | None = None,
        freedom_mask: ConstraintRelaxationMask | None = None, topology_axes: object | None = None,
        case_ratios: CasePhysicsRatios | None = None, cost_model: HybridAscendCostModel | None = None,
        escalation_tier: str = "tilelang_dsl", roofline: RooflineSummaryLike | Mapping[str, object] | None = None,
        ordered_tensor_shapes: Sequence[Sequence[int]] | None = None, case_entry: Mapping[str, object] | None = None,
        proto: Mapping[str, object] | None = None, cost_gate_decision: object | None = None,
        cost_model_mode: object | None = None, stagnation_rounds: int = 0,
        incumbent_roofline_eta: float | None = None, target_roofline_eta: float = 0.85,
        prev_local_rel_gain: float | None = None, mean_recoverable_excess_us: float | None = None,
        bucket_clean_samples: int = 0, out_of_sample_spearman_rho: float | None = None,
        focus_lines: Sequence[int] = (),
    ) -> tuple[WorkerSpec, ...]:
        """Select matching causal rules and fill remaining slots with creative explorers."""
        _ = (stagnation_rounds, incumbent_roofline_eta, target_roofline_eta, prev_local_rel_gain, mean_recoverable_excess_us)
        resolved_shapes = extract_ordered_tensor_shapes(
            case_entry=case_entry, dims=dims, proto=proto, ordered_tensor_shapes=ordered_tensor_shapes
        )
        eff_roofline_obj = roofline if roofline is not None else (blueprint.roofline if blueprint is not None else None)
        eff_ratios = case_ratios if case_ratios is not None else getattr(parent, "case_ratios", None)
        parent_features = _project_features_with_ratios(parent.features, eff_ratios)
        active_axes = _derive_active_topology_axes(
            parent.ledger, features=parent_features, sil_advisory=sil_advisory, topology_axes=topology_axes, case_ratios=eff_ratios
        )
        ctx = build_predicate_context(
            parent.ledger, features=parent_features, residual=parent.residual,
            compiler_summary=compiler_summary_from_residual(parent.residual),
            extra={"topology_axes": active_axes}, roofline=eff_roofline_obj, case_ratios=eff_ratios,
        )
        matched_rules = list(self._tracker.rank_applicable_rules(
            ctx, top_k=self._k_rules, topology_axes=active_axes,
            runtime_backend=escalation_tier, symptom_ranking=rank_symptom_families(ctx),
        ))
        eff_roofline = ideal_roofline_us if ideal_roofline_us > 0.0 else (blueprint.roofline.ideal_roofline_us if blueprint is not None else 0.0)
        specs: list[WorkerSpec] = [
            WorkerSpec(
                worker_id=f"r{round_idx}_rule_{idx}", worker_kind="rule_exploit", rule=rule,
                request=build_exploit_request(
                    parent, rule, blueprint_context=blueprint_context, blueprint=blueprint,
                    escalation_tier=escalation_tier, enable_track_router=self._enable_track_router,
                    sil_advisory=sil_advisory, sil_verification=sil_verification, freedom_mask=freedom_mask,
                ),
                parent_candidate=parent, ideal_roofline_us=eff_roofline, roofline=eff_roofline_obj,
            )
            for idx, rule in enumerate(matched_rules[: self._k_rules])
        ]

        excluded = summarize_excluded_mechanisms(matched_rules, self._beam.exploited_signatures)
        eff_mode = str(cost_model_mode if cost_model_mode is not None else self._cost_model_mode).strip().lower() or "off"
        eff_prescreen = self._enable_tilesize_prescreen and eff_mode != "off"
        if cost_gate_decision is None:
            cost_gate_decision = evaluate_cost_gate_policy(
                mode=eff_mode, enable_prescreen_flag=eff_prescreen,
                bucket_clean_samples=bucket_clean_samples, out_of_sample_spearman_rho=out_of_sample_spearman_rho,
            )
        policy_ok = (bool(getattr(cost_gate_decision, "engaged", True)) and bool(getattr(cost_gate_decision, "allow_prompt_guidance", True))) if cost_gate_decision is not None else True
        prescreen_summary = (
            build_prescreen_summary_from_kwargs(
                factory_kwargs, dsl_source=parent.dsl_source, factory_name=factory_name, var_bindings=var_bindings,
                config_selector_name=config_selector_name, dims=dims, cost_model=cost_model or self._cost_model,
                axes=active_axes, blueprint=blueprint, sil_advisory=sil_advisory, ordered_tensor_shapes=resolved_shapes,
            ) if (eff_prescreen and policy_ok) else ""
        )
        total_creative = 1 + max(0, self._k_rules - len(matched_rules))
        for c_idx in range(total_creative):
            specs.append(WorkerSpec(
                worker_id=f"r{round_idx}_creative" if c_idx == 0 else f"r{round_idx}_creative_{c_idx}",
                worker_kind="creative_explore", rule=None,
                request=build_creative_request(
                    parent, excluded_mechanisms=excluded,
                    focus_hint=CREATIVE_FOCUS_HINTS[c_idx % len(CREATIVE_FOCUS_HINTS)] if total_creative > 1 else "",
                    prescreen_summary=prescreen_summary, blueprint_context=blueprint_context, blueprint=blueprint,
                    effort=creative_effort, escalation_tier=escalation_tier, enable_track_router=self._enable_track_router,
                    sil_advisory=sil_advisory, sil_verification=sil_verification, freedom_mask=freedom_mask, focus_lines=focus_lines,
                ),
                parent_candidate=parent, ideal_roofline_us=eff_roofline, roofline=eff_roofline_obj,
            ))
        return tuple(specs)

    async def _run_single_worker(
        self, spec: WorkerSpec, *, round_idx: int, factory_name: str | None,
        factory_kwargs: dict[str, object] | None, var_bindings: Mapping[str, int] | None = None,
        config_selector_name: str | None = None, dims: Mapping[str, object] | None = None,
        sil_advisory: object | None = None, sil_verification: object | None = None,
        freedom_mask: ConstraintRelaxationMask | None = None, topology_axes: object | None = None,
        case_ratios: CasePhysicsRatios | None = None, escalation_tier: str = "tilelang_dsl",
        ordered_tensor_shapes: Sequence[Sequence[int]] | None = None,
        case_entry: Mapping[str, object] | None = None, proto: Mapping[str, object] | None = None,
    ) -> tuple[BeamCandidate | None, bool, tuple[str, str, str] | None, bool]:
        resolved_shapes = extract_ordered_tensor_shapes(
            case_entry=case_entry, dims=dims, proto=proto, ordered_tensor_shapes=ordered_tensor_shapes
        )
        active_axes = _derive_active_topology_axes(
            spec.parent_candidate.ledger,
            features=_project_features_with_ratios(spec.parent_candidate.features, case_ratios),
            sil_advisory=sil_advisory, topology_axes=topology_axes, case_ratios=case_ratios,
        )
        try:
            raw_reply = await self._backend.generate(spec.request)
        except Exception as exc:
            return None, False, (spec.worker_id, "llm_transport", str(exc)[-400:]), False
        try:
            if spec.worker_kind == "creative_explore":
                creative = extract_creative_payload(raw_reply)
                dsl_source, hypothesis = creative["dsl_source"], creative["physical_hypothesis"]
            else:
                dsl_source = extract_python_dsl(raw_reply)
                hypothesis = spec.rule.rule_id if spec.rule is not None else ""
        except Exception as exc:
            self._record_rule_failure(spec.rule, active_axes)
            return None, False, (spec.worker_id, "llm_parse", str(exc)[-400:]), False

        def _eval(src: str) -> FunnelEvaluation:
            return self._funnel.evaluate(
                src, factory_name=factory_name, factory_kwargs=factory_kwargs,
                config_selector_name=config_selector_name, dims=dims, var_bindings=var_bindings,
                roofline=spec.roofline, ordered_tensor_shapes=resolved_shapes,
            )

        repaired = False
        evaluation = _eval(dsl_source)
        if not evaluation.passed and evaluation.failed_stage in ("stage_a", "stage_b"):
            parent_dsl = spec.parent_candidate.dsl_source
            if classify_stage_failure_for_repair(_resolve_evaluation_failure_code(evaluation)):
                for _ in range(self._max_repairs):
                    low_ex, low_lang = extract_lowered_compiler_excerpt_with_lang(evaluation.stage_b)
                    repair_req = build_syntax_repair_request(
                        dsl_source, evaluation.error_message, sil_advisory=sil_advisory,
                        sil_verification=sil_verification, backend_kind=escalation_tier,
                        parent_source=parent_dsl, freedom_mask=freedom_mask,
                        ub_256kb_unlock=bool(spec.parent_candidate.ledger.uses_unreserved_ubuf),
                        topology_axes=active_axes, case_ratios=case_ratios,
                        lowered_excerpt=low_ex, lowered_excerpt_lang=low_lang,
                    )
                    try:
                        repaired_raw = await self._backend.generate(repair_req)
                    except Exception as exc:
                        return None, repaired, (spec.worker_id, "llm_transport", str(exc)[-400:]), False
                    try:
                        dsl_source = extract_python_dsl(repaired_raw)
                    except Exception as exc:
                        self._record_rule_failure(spec.rule, active_axes)
                        return None, repaired, (spec.worker_id, "llm_parse", str(exc)[-400:]), False
                    if parent_dsl.strip() and (violations := check_repair_monotonicity(parent_dsl, dsl_source, backend_kind=escalation_tier)):
                        self._record_rule_failure(spec.rule, active_axes)
                        return None, repaired, (spec.worker_id, "repair_monotonicity_violation", "; ".join(violations)), False
                    repaired = True
                    evaluation = _eval(dsl_source)
                    if evaluation.passed or not classify_stage_failure_for_repair(_resolve_evaluation_failure_code(evaluation)):
                        break

        if not evaluation.passed or evaluation.stage_a.ledger is None:
            self._record_rule_failure(spec.rule, active_axes)
            return None, repaired, (spec.worker_id, evaluation.failed_stage or "unknown", evaluation.error_message), False

        ledger = evaluation.stage_a.ledger
        comp_summary = evaluation.stage_b.compiler_summary if evaluation.stage_b else None
        stage_feats = evaluation.stage_a.features
        worker_features = _project_features_with_ratios(
            stage_feats if isinstance(stage_feats, Mapping) else spec.parent_candidate.features, case_ratios
        )
        cand_axes = _derive_active_topology_axes(
            ledger, features=worker_features, sil_advisory=sil_advisory,
            route=evaluation.stage_a.route, topology_axes=topology_axes, case_ratios=case_ratios,
        )
        track = evaluation.stage_a.route.track if evaluation.stage_a.route is not None else (spec.parent_candidate.track or "unconstrained")
        stage_c = evaluation.stage_c
        is_proxy = False
        if stage_c is not None:
            latency_us, acc_ok, tel = stage_c.measured_latency_us, stage_c.accuracy_passed, dict(stage_c.telemetry)
            is_proxy = bool(tel.get("is_analytical_proxy", False))
            dur_us = float(tel.get("dur_us", latency_us))
            needs_refresh = spec.roofline is not None
            if spec.ideal_roofline_us > 0.0 and dur_us > 0.0 and "roofline_eta" not in tel:
                eta = min(1.0, spec.ideal_roofline_us / max(1e-6, dur_us))
                tel["roofline_eta"] = eta
                if isinstance(stage_c.telemetry, dict):
                    stage_c.telemetry["roofline_eta"] = eta
                needs_refresh = True
            residual = (
                compute_static_dynamic_residual(ledger, tel, compiler_summary=comp_summary, roofline=spec.roofline)
                if needs_refresh else (stage_c.residual if stage_c.residual is not None else spec.parent_candidate.residual)
            )
        else:
            residual, latency_us, acc_ok = spec.parent_candidate.residual, spec.parent_candidate.latency_us, True

        if spec.rule is not None:
            post_ctx = build_predicate_context(
                ledger, features=worker_features, residual=residual, compiler_summary=comp_summary,
                prev_ledger=spec.parent_candidate.ledger, prev_residual=spec.parent_candidate.residual,
                prev_compiler_summary=compiler_summary_from_residual(spec.parent_candidate.residual),
                roofline=spec.roofline,
            )
            if isinstance(prev_ns := post_ctx.get("prev"), dict):
                prev_ns.update(spec.parent_candidate.residual.to_dict())
            realized = spec.rule.postcondition_realized(post_ctx)
            p_us = spec.parent_candidate.latency_us
            gain = ((p_us - latency_us) / p_us) if (p_us > 0.0 and latency_us > 0.0) else 0.0
            if not is_proxy or realized:
                self._tracker.record_outcome(
                    spec.rule.rule_id, postcondition_realized=realized, speed_gain=max(0.0, gain), topology_axes=cand_axes
                )

        cand = BeamCandidate(
            candidate_id=spec.worker_id, round_idx=round_idx, worker_kind=spec.worker_kind,
            rule_id=spec.rule.rule_id if spec.rule else None, dsl_source=dsl_source, track=track,
            ledger=ledger, residual=residual, latency_us=latency_us, accuracy_passed=acc_ok,
            structural_signature=extract_structural_signature(dsl_source, track=track, residual=residual, stage_b=evaluation.stage_b),
            hypothesis=hypothesis, parent_id=spec.parent_candidate.candidate_id, features=worker_features,
        )
        return cand, repaired, None, is_proxy

    async def run_round(
        self, parent: BeamCandidate, *, round_idx: int = 1, factory_name: str | None = None,
        factory_kwargs: dict[str, object] | None = None, var_bindings: Mapping[str, int] | None = None,
        config_selector_name: str | None = None, dims: Mapping[str, object] | None = None,
        blueprint_context: str = "", ideal_roofline_us: float = 0.0, blueprint: HirBlueprintPacket | None = None,
        creative_effort: str = "high", sil_advisory: object | None = None, sil_verification: object | None = None,
        freedom_mask: ConstraintRelaxationMask | None = None, topology_axes: object | None = None,
        case_ratios: CasePhysicsRatios | None = None, escalation_tier: str = "tilelang_dsl",
        roofline: RooflineSummaryLike | Mapping[str, object] | None = None,
        ordered_tensor_shapes: Sequence[Sequence[int]] | None = None,
        case_entry: Mapping[str, object] | None = None, proto: Mapping[str, object] | None = None,
    ) -> PortfolioRoundReport:
        """Execute one parallel portfolio round across Rule and Creative workers."""
        resolved_shapes = extract_ordered_tensor_shapes(
            case_entry=case_entry, dims=dims, proto=proto, ordered_tensor_shapes=ordered_tensor_shapes
        )
        active_axes = _derive_active_topology_axes(
            parent.ledger, features=_project_features_with_ratios(parent.features, case_ratios),
            sil_advisory=sil_advisory, topology_axes=topology_axes, case_ratios=case_ratios,
        )
        specs = self.plan_workers(
            parent, round_idx, factory_name=factory_name, factory_kwargs=factory_kwargs,
            var_bindings=var_bindings, config_selector_name=config_selector_name, dims=dims,
            blueprint_context=blueprint_context, ideal_roofline_us=ideal_roofline_us, blueprint=blueprint,
            creative_effort=creative_effort, sil_advisory=sil_advisory, sil_verification=sil_verification,
            freedom_mask=freedom_mask, topology_axes=active_axes, case_ratios=case_ratios,
            escalation_tier=escalation_tier, roofline=roofline, ordered_tensor_shapes=resolved_shapes,
        )

        async def _run_spec(spec: WorkerSpec) -> tuple[BeamCandidate | None, bool, tuple[str, str, str] | None, bool]:
            return await self._run_single_worker(
                spec, round_idx=round_idx, factory_name=factory_name, factory_kwargs=factory_kwargs,
                var_bindings=var_bindings, config_selector_name=config_selector_name, dims=dims,
                sil_advisory=sil_advisory, sil_verification=sil_verification, freedom_mask=freedom_mask,
                topology_axes=active_axes, case_ratios=case_ratios, escalation_tier=escalation_tier,
                ordered_tensor_shapes=resolved_shapes,
            )

        outcomes: list[tuple[BeamCandidate | None, bool, tuple[str, str, str] | None, bool]] = []
        if self._pending_speculative_creative is not None and self._pending_speculative_creative.done():
            done_task = self._pending_speculative_creative
            self._pending_speculative_creative = None
            if not done_task.cancelled():
                outcomes.append(await done_task)

        exploit_specs = [s for s in specs if s.worker_kind != "creative_explore"]
        creative_specs = [s for s in specs if s.worker_kind == "creative_explore"]
        if not exploit_specs:
            pending_tasks = []
            if self._pending_speculative_creative is not None:
                pending_tasks.append(self._pending_speculative_creative)
                self._pending_speculative_creative = None
            outcomes.extend(await asyncio.gather(*pending_tasks, *(_run_spec(s) for s in creative_specs)))
        else:
            sync_specs = list(exploit_specs)
            if creative_specs:
                if self._pending_speculative_creative is None:
                    self._pending_speculative_creative = asyncio.create_task(_run_spec(creative_specs[0]))
                sync_specs.extend(creative_specs[1:])
            outcomes.extend(await asyncio.gather(*(_run_spec(s) for s in sync_specs)))
            if self._pending_speculative_creative is not None and not self._pending_speculative_creative.done():
                await asyncio.sleep(0)
            if self._pending_speculative_creative is not None and self._pending_speculative_creative.done():
                done_task = self._pending_speculative_creative
                self._pending_speculative_creative = None
                if not done_task.cancelled():
                    outcomes.append(await done_task)

        report = PortfolioRoundReport(round_idx=round_idx, spawned_workers=specs)
        saw_proxy = saw_real_hw = False
        for cand, repaired, fail, is_proxy in outcomes:
            if repaired:
                report.repaired_worker_ids.append(cand.candidate_id if cand else (fail[0] if fail else ""))
            if cand is not None:
                report.evaluated_candidates.append(cand)
                saw_proxy = saw_proxy or is_proxy
                saw_real_hw = saw_real_hw or not is_proxy
            elif fail is not None:
                report.failed_candidates.append(fail)
        report.pruned_rule_ids = () if (saw_proxy and not saw_real_hw) else self._tracker.prune_underperforming(topology_axes=active_axes)
        report.exploit_beam, report.stepping_stone_beam = self._beam.update([parent, *report.evaluated_candidates])
        return report

    def check_round_escalation(
        self, candidate: BeamCandidate, *, current_tier: str = "tilelang_dsl", target_eta: float = 0.85,
        stagnation_rounds: int = 0, ideal_roofline_us: float = 0.0, dominant_bound: str = "",
        compiler_summary: BishengResourceSummary | Mapping[str, object] | None = None,
        current_mask: ConstraintRelaxationMask | None = None, cost_model: HybridAscendCostModel | None = None,
        workload: Mapping[str, object] | None = None, case_ratios: CasePhysicsRatios | None = None,
        blueprint: HirBlueprintPacket | None = None, axes: TopologyAxes | None = None,
        sil_advisory: object | None = None,
    ) -> RoundEscalationReport:
        """Run closed-loop track escalation and facet relaxation using HybridAscendCostModel."""
        model = cost_model if cost_model is not None else (self._cost_model or HybridAscendCostModel(AscendTarget("ascend.950pr")))
        eff_workload: CostFeatures | Mapping[str, object] = (
            dict(workload) if workload else build_cost_features_from_ledger(
                candidate.ledger, axes=axes, blueprint=blueprint, sil_advisory=sil_advisory, target=model.target
            )
        )
        if compiler_summary is None:
            eff_summary = BishengResourceSummary(
                spill_bytes=int(candidate.residual.spill_bytes), vloopv2_loops=1 if candidate.residual.vloopv2_hit else 0,
                software_loops=0 if candidate.residual.vloopv2_hit else 1,
                has_local_membar_in_vf_call=bool(candidate.residual.has_local_membar_in_vf_call),
                vf_fusion_breaks=int(candidate.residual.vf_fusion_breaks),
                alias_serialized_buffers=int(candidate.residual.alias_serialized_buffers),
                dual_issue_ipc_collapse=float(candidate.residual.dual_issue_ipc_collapse),
                dcache_dcci_flushes=int(candidate.residual.dcache_dcci_flushes),
            )
        else:
            eff_summary = compiler_summary if isinstance(compiler_summary, BishengResourceSummary) else BishengResourceSummary.from_dict(compiler_summary)
        cost_pred = model.predict(eff_workload, bisheng_res=eff_summary)
        if ideal_roofline_us > 0.0 and candidate.latency_us > 0.0:
            eta = min(1.0, float(ideal_roofline_us) / float(candidate.latency_us))
        elif cost_pred.pred_us > 0.0 and cost_pred.tier0_physics_us > 0.0:
            eta = min(1.0, float(cost_pred.tier0_physics_us) / float(cost_pred.pred_us))
        else:
            eta = max(0.0, 1.0 - min(1.0, float(candidate.residual.unexplained_gap_ratio)))
        eff_bound = dominant_bound.strip() or cost_pred.dominant_bound
        escalation = evaluate_track_escalation(
            current_tier=current_tier, roofline_eta=eta, target_eta=target_eta, stagnation_rounds=stagnation_rounds,
            dominant_bound=eff_bound, compiler_summary=eff_summary, structural_features=_project_features_with_ratios(candidate.features, case_ratios),
        )
        waterlines = getattr(candidate.ledger, "waterlines", {}) or {}
        l1_ov = bool((wl := waterlines.get("L1")) is not None and int(getattr(wl, "overflow_bytes", 0) or 0) > 0)
        ub_ov = bool((wl := waterlines.get("UB")) is not None and int(getattr(wl, "overflow_bytes", 0) or 0) > 0)
        relaxed_mask = relax_facets_for_bottleneck(
            current_mask if current_mask is not None else ConstraintRelaxationMask(), dominant_bound=eff_bound,
            residual=candidate.residual, compiler_summary=eff_summary, l1_overflow=l1_ov, ub_overflow=ub_ov,
        )
        return RoundEscalationReport(escalation=escalation, relaxed_mask=relaxed_mask, cost_prediction=cost_pred, roofline_eta=round(eta, 6), l1_overflow=l1_ov, ub_overflow=ub_ov)


__all__ = [
    "CREATIVE_FOCUS_HINTS", "PortfolioRoundReport", "PortfolioRunner", "RoundEscalationReport",
    "TILELANG_ASCEND_PRIMITIVE_MENU", "WorkerSpec", "build_creative_request", "build_exploit_request",
    "build_prescreen_summary_from_kwargs", "build_syntax_repair_request", "check_repair_monotonicity",
    "classify_stage_failure_for_repair", "format_sil_diagnostics_section",
    "parse_scoped_search_replace_blocks", "summarize_excluded_mechanisms",
]
