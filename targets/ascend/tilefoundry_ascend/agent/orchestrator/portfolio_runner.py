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
    format_prescreen_summary,
    generate_aligned_neighborhood,
    prescreen_tile_configs,
)
from tilefoundry_ascend.agent.orchestrator.twin_bridge import format_hir_tir_navigation_section
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import HirTirNavigationPacket

CREATIVE_FOCUS_HINTS: tuple[str, ...] = (
    "memory_hierarchy_and_tiling",
    "compute_pipeline_and_sync",
    "cross_core_dataflow_and_fusion",
)

TILELANG_ASCEND_PRIMITIVE_MENU: str = "\n".join(
    (
        "- T.MixedKernel(28, sids=2): coupled 1 AIC + 2 AIV execution (or pure-AIV sids=2)",
        "- T.Kernel(blocks) / T.Persistent(...): grid dispatch and persistent 2D swizzle",
        "- T.Pipelined(count, num_stages=...): multi-buffered loop pipelining",
        "- T.SimdVF(): native SIMD vector-function mode (256 KiB UB with PURE_SIMDVF_UBUF_FLAGS)",
        "- T.SimtVF(threads=128): SIMT vector-function mode for irregular/scalar-tail loops",
        "- T.Scope('C') / T.Scope('V'): explicit AIC Cube and AIV Vector core regions",
        "- T.gemm(a, b, c, ...): Cube MMAD on L1/L0A/L0B -> L0C",
        "- T.copy(src, dst) / T.dual_copy(src, dst, ...): GM/L1/L0C/UB DMA and 1C2V split copy",
        "- T.alloc_l1 / T.alloc_l0a / T.alloc_l0b / T.alloc_l0c: AIC hierarchical SRAM buffers",
        "- T.alloc_shared(...) / T.alloc_ss(...): AIV Unified Buffer (UB) and 3 KiB shared SSBUF",
        "- T.set_atomic(...): atomic reduction/accumulation mode on GM outputs",
        (
            "- PURE_SIMDVF_UBUF_FLAGS: unlock 256 KiB UB via _select_*_config()"
            "[\"pass_configs\"]/\"compile_flags\" or PrimFunc.attrs"
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


def build_prescreen_summary_from_kwargs(kwargs: Mapping[str, object] | None) -> str:
    """Build tile prescreen summary when tile dimensions appear in builder kwargs."""
    if not kwargs:
        return ""
    lowered = {str(k).lower(): v for k, v in kwargs.items()}

    def _pick_pos_int(keys: tuple[str, ...]) -> int | None:
        for key in keys:
            val = lowered.get(key)
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
    center = TileConfigCandidate(
        bm=bm_val or 64,
        bn=bn_val or 64,
        bk=bk_val or 64,
        stages=stages_val,
        dtype_bytes=dtype_bytes,
    )
    neighbors = generate_aligned_neighborhood(center)
    scores = prescreen_tile_configs(neighbors, top_k=4)
    if not scores:
        return ""
    return format_prescreen_summary(scores)


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


def _resolve_navigation_section(
    parent: BeamCandidate,
    navigation_context: str,
    nav_packet: HirTirNavigationPacket | None,
) -> str:
    if nav_packet is not None:
        nav_text = format_hir_tir_navigation_section(
            nav_packet,
            current_tl_ledger=parent.ledger,
        )
    else:
        nav_text = navigation_context.strip()
    return f"{nav_text}\n\n" if nav_text else ""


def build_exploit_request(
    parent: BeamCandidate,
    rule: AscendCausalRule,
    *,
    navigation_context: str = "",
    nav_packet: HirTirNavigationPacket | None = None,
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
    nav_section = _resolve_navigation_section(parent, navigation_context, nav_packet)
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
    nav_section = _resolve_navigation_section(parent, navigation_context, nav_packet)
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
    )


def build_syntax_repair_request(dsl_source: str, error_message: str) -> GenerationRequest:
    """Build bounded syntax/compile repair request routed to Hydrogen Medium."""
    sys_prompt = (
        "You are a TileLang Ascend compiler repair engineer. Fix the syntax, API, or "
        "memory-capacity failure reported below without altering tensor semantics. "
        "Output ONLY corrected Python code."
    )
    user_prompt = (
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
    ) -> None:
        self._backend = backend
        self._funnel = funnel
        self._tracker = tracker
        self._beam = beam or TwoLaneBeamSearch()
        self._k_rules = max(1, k_rule_workers)
        self._max_repairs = max(0, max_syntax_repairs)
        self._enable_track_router = enable_track_router

    @property
    def beam(self) -> TwoLaneBeamSearch:
        """Return active TwoLaneBeamSearch instance."""
        return self._beam

    def plan_workers(
        self,
        parent: BeamCandidate,
        round_idx: int,
        *,
        factory_kwargs: Mapping[str, object] | None = None,
        navigation_context: str = "",
        ideal_roofline_us: float = 0.0,
        nav_packet: HirTirNavigationPacket | None = None,
    ) -> tuple[WorkerSpec, ...]:
        """Select matching causal rules and fill remaining slots with creative explorers."""
        ctx = build_predicate_context(
            parent.ledger,
            features=parent.features,
            residual=parent.residual,
        )
        matched_rules = list(self._tracker.rank_applicable_rules(ctx, top_k=self._k_rules))
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
                    ),
                    parent_candidate=parent,
                    ideal_roofline_us=eff_roofline,
                )
            )

        excluded = summarize_excluded_mechanisms(
            matched_rules,
            self._beam.exploited_signatures,
        )
        prescreen_summary = build_prescreen_summary_from_kwargs(factory_kwargs)
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
    ) -> tuple[BeamCandidate | None, bool, tuple[str, str, str] | None]:
        try:
            raw_reply = await self._backend.generate(spec.request)
        except Exception as exc:
            return None, False, (spec.worker_id, "llm_transport", str(exc)[-400:])

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
                self._tracker.record_outcome(spec.rule.rule_id, postcondition_realized=False)
            return None, False, (spec.worker_id, "llm_parse", str(exc)[-400:])

        repaired = False
        evaluation: FunnelEvaluation = self._funnel.evaluate(
            dsl_source,
            factory_name=factory_name,
            factory_kwargs=factory_kwargs,
        )
        if not evaluation.passed and evaluation.failed_stage in ("stage_a", "stage_b"):
            for _ in range(self._max_repairs):
                repair_req = build_syntax_repair_request(dsl_source, evaluation.error_message)
                try:
                    repaired_raw = await self._backend.generate(repair_req)
                except Exception as exc:
                    return None, repaired, (spec.worker_id, "llm_transport", str(exc)[-400:])
                try:
                    dsl_source = extract_python_dsl(repaired_raw)
                except Exception as exc:
                    if spec.rule is not None:
                        self._tracker.record_outcome(
                            spec.rule.rule_id, postcondition_realized=False
                        )
                    return None, repaired, (spec.worker_id, "llm_parse", str(exc)[-400:])
                repaired = True
                evaluation = self._funnel.evaluate(
                    dsl_source,
                    factory_name=factory_name,
                    factory_kwargs=factory_kwargs,
                )
                if evaluation.passed:
                    break

        if not evaluation.passed or evaluation.stage_a.ledger is None:
            if spec.rule is not None:
                self._tracker.record_outcome(spec.rule.rule_id, postcondition_realized=False)
            fail_stage = evaluation.failed_stage or "unknown"
            return None, repaired, (spec.worker_id, fail_stage, evaluation.error_message)

        ledger = evaluation.stage_a.ledger
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
        if stage_c is not None:
            latency_us = stage_c.measured_latency_us
            acc_ok = stage_c.accuracy_passed
            tel = dict(stage_c.telemetry)
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
                residual = compute_static_dynamic_residual(ledger, tel)
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
            self._tracker.record_outcome(
                spec.rule.rule_id,
                postcondition_realized=realized,
                speed_gain=max(0.0, speed_gain),
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
            structural_signature=extract_structural_signature(dsl_source, track=track),
            hypothesis=hypothesis,
            parent_id=spec.parent_candidate.candidate_id,
            features=evaluation.stage_a.features or spec.parent_candidate.features,
        )
        return cand, repaired, None

    async def run_round(
        self,
        parent: BeamCandidate,
        *,
        round_idx: int = 1,
        factory_name: str | None = None,
        factory_kwargs: dict[str, object] | None = None,
        navigation_context: str = "",
        ideal_roofline_us: float = 0.0,
        nav_packet: HirTirNavigationPacket | None = None,
    ) -> PortfolioRoundReport:
        """Execute one parallel portfolio round across Rule and Creative workers."""
        specs = self.plan_workers(
            parent,
            round_idx,
            factory_kwargs=factory_kwargs,
            navigation_context=navigation_context,
            ideal_roofline_us=ideal_roofline_us,
            nav_packet=nav_packet,
        )
        tasks = [
            self._run_single_worker(
                spec,
                round_idx=round_idx,
                factory_name=factory_name,
                factory_kwargs=factory_kwargs,
            )
            for spec in specs
        ]
        outcomes = await asyncio.gather(*tasks)

        report = PortfolioRoundReport(round_idx=round_idx, spawned_workers=specs)
        for cand, repaired, fail in outcomes:
            if repaired:
                uid = cand.candidate_id if cand else (fail[0] if fail else "")
                report.repaired_worker_ids.append(uid)
            if cand is not None:
                report.evaluated_candidates.append(cand)
            elif fail is not None:
                report.failed_candidates.append(fail)

        pruned = self._tracker.prune_underperforming()
        report.pruned_rule_ids = pruned
        exploit, stone = self._beam.update([parent, *report.evaluated_candidates])
        report.exploit_beam = exploit
        report.stepping_stone_beam = stone
        return report


__all__ = [
    "CREATIVE_FOCUS_HINTS",
    "PortfolioRoundReport",
    "PortfolioRunner",
    "TILELANG_ASCEND_PRIMITIVE_MENU",
    "WorkerSpec",
    "build_creative_request",
    "build_exploit_request",
    "build_prescreen_summary_from_kwargs",
    "build_syntax_repair_request",
    "summarize_excluded_mechanisms",
]
