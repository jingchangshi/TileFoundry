"""End-to-end HIR-to-kernel dual-loop agent evolution driver and trajectory tracker."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json

from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    FunnelEvaluation,
    GateFunnel,
    StageAResult,
    StageCResult,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import (
    GenerationRequest,
    KernelLLMBackend,
    extract_python_dsl,
)
from tilefoundry_ascend.agent.orchestrator.portfolio_runner import (
    TILELANG_ASCEND_PRIMITIVE_MENU,
    PortfolioRoundReport,
    PortfolioRunner,
    build_syntax_repair_request,
)
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    BeamCandidate,
    TwoLaneBeamSearch,
    compute_static_dynamic_residual,
    extract_structural_signature,
)
from tilefoundry_ascend.agent.orchestrator.twin_bridge import format_hir_tir_navigation_section
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import (
    HirTirNavigationPacket,
    compile_hir_navigation_packet,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger
from tilefoundry_ascend.ascend.target import AscendTarget


@dataclass(frozen=True)
class WorkerStepRecord:
    """Single worker attempt and funnel telemetry inside one evolution round."""

    worker_id: str
    worker_kind: str
    rule_id: str | None
    repaired: bool
    passed: bool
    failed_stage: str | None
    latency_us: float
    estimated_cycles: float
    dominant_pipe: str
    unexplained_gap_ratio: float
    hypothesis: str
    dsl_sha256: str

    def to_dict(self) -> dict[str, object]:
        """Serialize worker step telemetry as a JSON-compatible dictionary."""
        return {
            "worker_id": self.worker_id,
            "worker_kind": self.worker_kind,
            "rule_id": self.rule_id,
            "repaired": self.repaired,
            "passed": self.passed,
            "failed_stage": self.failed_stage,
            "latency_us": round(self.latency_us, 6),
            "estimated_cycles": round(self.estimated_cycles, 3),
            "dominant_pipe": self.dominant_pipe,
            "unexplained_gap_ratio": round(self.unexplained_gap_ratio, 6),
            "hypothesis": self.hypothesis,
            "dsl_sha256": self.dsl_sha256,
        }


@dataclass(frozen=True)
class EvolutionRoundRecord:
    """Per-round beam, pruning, and latency convergence snapshot."""

    round_idx: int
    parent_id: str
    parent_latency_us: float
    parent_dominant_pipe: str
    workers: tuple[WorkerStepRecord, ...]
    exploit_beam_ids: tuple[str, ...]
    stepping_stone_beam_ids: tuple[str, ...]
    pruned_rule_ids: tuple[str, ...]
    best_round_latency_us: float
    best_so_far_latency_us: float

    def to_dict(self) -> dict[str, object]:
        """Serialize evolution round snapshot to a JSON-compatible dictionary."""
        return {
            "round_idx": self.round_idx,
            "parent_id": self.parent_id,
            "parent_latency_us": round(self.parent_latency_us, 6),
            "parent_dominant_pipe": self.parent_dominant_pipe,
            "workers": [w.to_dict() for w in self.workers],
            "exploit_beam_ids": list(self.exploit_beam_ids),
            "stepping_stone_beam_ids": list(self.stepping_stone_beam_ids),
            "pruned_rule_ids": list(self.pruned_rule_ids),
            "best_round_latency_us": round(self.best_round_latency_us, 6),
            "best_so_far_latency_us": round(self.best_so_far_latency_us, 6),
        }


@dataclass
class EvolutionTrajectory:
    """Complete multi-round convergence record from unscheduled HIR to evolved DSL."""

    problem_name: str
    track: str
    total_rounds: int
    baseline_candidate_id: str
    baseline_latency_us: float
    baseline_estimated_cycles: float
    best_candidate_id: str
    best_latency_us: float
    best_estimated_cycles: float
    speedup_vs_baseline: float
    best_dsl_source: str
    rounds: tuple[EvolutionRoundRecord, ...] = field(default_factory=tuple)
    active_rules_after: tuple[str, ...] = field(default_factory=tuple)
    feature_vector: dict[str, object] = field(default_factory=dict)
    hir_navigation_summary: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize complete evolution trajectory to a JSON-compatible dictionary."""
        return {
            "problem_name": self.problem_name,
            "track": self.track,
            "feature_vector": dict(self.feature_vector),
            "total_rounds": self.total_rounds,
            "baseline_candidate_id": self.baseline_candidate_id,
            "baseline_latency_us": round(self.baseline_latency_us, 6),
            "baseline_estimated_cycles": round(self.baseline_estimated_cycles, 3),
            "best_candidate_id": self.best_candidate_id,
            "best_latency_us": round(self.best_latency_us, 6),
            "best_estimated_cycles": round(self.best_estimated_cycles, 3),
            "speedup_vs_baseline": round(self.speedup_vs_baseline, 6),
            "best_dsl_source": self.best_dsl_source,
            "rounds": [r.to_dict() for r in self.rounds],
            "active_rules_after": list(self.active_rules_after),
            "hir_navigation_summary": dict(self.hir_navigation_summary),
        }

    def to_json(self) -> str:
        """Render formatted JSON representation of the evolution trajectory."""
        return json.dumps(self.to_dict(), indent=2)


def _ledger_estimated_cycles(ledger: ObjectivePhysicalLedger) -> float:
    if ledger.steady_state_ii is not None and ledger.steady_state_ii > 0:
        return float(ledger.steady_state_ii)
    return float(max(ledger.pipe_cycles.values(), default=0))


def _proxy_latency_from_ledger(ledger: ObjectivePhysicalLedger) -> float:
    max_cycles = max(ledger.pipe_cycles.values()) if ledger.pipe_cycles else 100
    return max(1.0, float(max_cycles) / 20.0)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _compile_nav_packet_safe(
    hir_source: str,
    *,
    scheduled_hir_source: str | None = None,
    anchor_dims: Mapping[str, int] | None = None,
    target: AscendTarget | None = None,
) -> HirTirNavigationPacket | None:
    """Compile authored HIR and optional scheduled HIR into a navigation packet."""
    try:
        return compile_hir_navigation_packet(
            hir_source,
            scheduled_hir_source=scheduled_hir_source,
            anchor_dims=anchor_dims,
            target=target,
        )
    except Exception:
        return None


def _summarize_nav_packet(packet: HirTirNavigationPacket | None) -> dict[str, object]:
    """Return a compact JSON-serializable summary of *packet* for trajectory logs."""
    if packet is None:
        return {}
    return {
        "entry_name": packet.entry_name,
        "roofline": packet.roofline.to_dict(),
        "candidate_op_count": len(packet.candidate_ops),
        "candidate_ops": [entry.to_dict() for entry in packet.candidate_ops],
        "has_tir_blueprint": packet.tir_script is not None,
        "z3_pipeline_summary": dict(packet.z3_pipeline_summary),
        "tir_ledger": packet.tir_ledger.to_dict() if packet.tir_ledger is not None else None,
        "lowering_diagnostics": list(packet.lowering_diagnostics),
    }


def build_hir_seed_request(
    hir_source: str,
    shape_contract: str,
    *,
    factory_name: str | None = None,
    nav_packet: HirTirNavigationPacket | None = None,
    navigation_context: str = "",
) -> GenerationRequest:
    """Build initial Tier-0 TileLang synthesis request from mathematical HIR source."""
    target_factory = factory_name or "make_kernel"
    sys_prompt = (
        "You are an Ascend 950PR TileLang compiler engineer. Translate the mathematical HIR "
        "module below into a clean, unscheduled standalone Python factory function named "
        f"'{target_factory}' returning a @T.prim_func named 'main'. "
        "Import 'import tilelang.ascend.language as T'. Do NOT include "
        "'from __future__ import annotations'. Output ONLY valid Python code."
    )
    nav_text = (
        format_hir_tir_navigation_section(nav_packet)
        if nav_packet is not None
        else navigation_context.strip()
    )
    nav_section = f"{nav_text}\n\n" if nav_text else ""
    primitive_section = (
        f"### TILELANG ASCEND PRIMITIVE REFERENCE\n{TILELANG_ASCEND_PRIMITIVE_MENU}\n\n"
    )
    user_prompt = (
        f"### SHAPE AND SIGNATURE CONTRACT\n{shape_contract.strip()}\n\n"
        f"{nav_section}"
        f"{primitive_section}"
        f"### MATHEMATICAL HIR SOURCE\n```python\n{hir_source.strip()}\n```\n"
    )
    return GenerationRequest(
        role="exploit_rule",
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        json_schema=None,
    )


async def synthesize_initial_dsl(
    *,
    hir_source: str,
    shape_contract: str,
    backend: KernelLLMBackend,
    funnel: GateFunnel,
    factory_name: str | None = None,
    factory_kwargs: Mapping[str, object] | None = None,
    max_seed_repairs: int = 2,
    target: AscendTarget | None = None,
    enable_track_router: bool = False,
    scheduled_hir_source: str | None = None,
    anchor_dims: Mapping[str, int] | None = None,
    nav_packet: HirTirNavigationPacket | None = None,
) -> tuple[str, FunnelEvaluation]:
    """Synthesize and funnel-validate an initial @T.prim_func seed from HIR source."""
    active_packet = nav_packet or _compile_nav_packet_safe(
        hir_source,
        scheduled_hir_source=scheduled_hir_source,
        anchor_dims=anchor_dims,
        target=target,
    )
    eval_funnel = (
        GateFunnel(
            target=target or funnel._target,
            skip_stage_b=funnel._skip_stage_b,
            compile_runner=funnel._compile_runner,
            device_runner=funnel._device_runner,
            enable_track_router=True,
        )
        if enable_track_router and not funnel._enable_track_router
        else funnel
    )
    req = build_hir_seed_request(
        hir_source,
        shape_contract,
        factory_name=factory_name,
        nav_packet=active_packet,
    )
    raw = await backend.generate(req)
    dsl_source = extract_python_dsl(raw)
    eval_res = eval_funnel.evaluate(
        dsl_source,
        factory_name=factory_name,
        factory_kwargs=factory_kwargs,
    )
    repairs = 0
    while not eval_res.passed and repairs < max(0, max_seed_repairs):
        repairs += 1
        repair_req = build_syntax_repair_request(dsl_source, eval_res.error_message)
        repaired_raw = await backend.generate(repair_req)
        dsl_source = extract_python_dsl(repaired_raw)
        eval_res = eval_funnel.evaluate(
            dsl_source,
            factory_name=factory_name,
            factory_kwargs=factory_kwargs,
        )
    return dsl_source, eval_res


def _build_round_record(
    report: PortfolioRoundReport,
    parent: BeamCandidate,
    best_so_far_latency: float,
) -> tuple[EvolutionRoundRecord, float]:
    by_id = {cand.candidate_id: cand for cand in report.evaluated_candidates}
    fail_by_id = {wid: (stage, err) for wid, stage, err in report.failed_candidates}
    repaired_set = set(report.repaired_worker_ids)

    worker_records: list[WorkerStepRecord] = []
    round_best_lat = best_so_far_latency
    for spec in report.spawned_workers:
        cand = by_id.get(spec.worker_id)
        if cand is not None:
            lat = cand.latency_us
            round_best_lat = min(round_best_lat, lat)
            worker_records.append(
                WorkerStepRecord(
                    worker_id=spec.worker_id,
                    worker_kind=spec.worker_kind,
                    rule_id=spec.rule.rule_id if spec.rule else None,
                    repaired=spec.worker_id in repaired_set,
                    passed=True,
                    failed_stage=None,
                    latency_us=lat,
                    estimated_cycles=_ledger_estimated_cycles(cand.ledger),
                    dominant_pipe=cand.ledger.dominant_pipe,
                    unexplained_gap_ratio=cand.residual.unexplained_gap_ratio,
                    hypothesis=cand.hypothesis,
                    dsl_sha256=_sha256_text(cand.dsl_source),
                )
            )
        else:
            stage, err = fail_by_id.get(spec.worker_id, ("unknown", "evaluation failed"))
            worker_records.append(
                WorkerStepRecord(
                    worker_id=spec.worker_id,
                    worker_kind=spec.worker_kind,
                    rule_id=spec.rule.rule_id if spec.rule else None,
                    repaired=spec.worker_id in repaired_set,
                    passed=False,
                    failed_stage=stage,
                    latency_us=0.0,
                    estimated_cycles=0.0,
                    dominant_pipe="unknown",
                    unexplained_gap_ratio=1.0,
                    hypothesis=err[:160],
                    dsl_sha256="",
                )
            )

    new_best_so_far = min(best_so_far_latency, round_best_lat)
    rec = EvolutionRoundRecord(
        round_idx=report.round_idx,
        parent_id=parent.candidate_id,
        parent_latency_us=parent.latency_us,
        parent_dominant_pipe=parent.ledger.dominant_pipe,
        workers=tuple(worker_records),
        exploit_beam_ids=tuple(c.candidate_id for c in report.exploit_beam),
        stepping_stone_beam_ids=tuple(c.candidate_id for c in report.stepping_stone_beam),
        pruned_rule_ids=tuple(report.pruned_rule_ids),
        best_round_latency_us=round_best_lat,
        best_so_far_latency_us=new_best_so_far,
    )
    return rec, new_best_so_far


def _derive_proportional_pmu_ratios(ledger: ObjectivePhysicalLedger) -> tuple[float, float, float]:
    """Derive proportional (aic_mad, aiv_vec, aiv_mte2) ratios from Stage-A pipe_cycles."""
    pipes = ledger.pipe_cycles or {}
    aic_cycles = sum(
        max(0, int(v))
        for k, v in pipes.items()
        if any(tok in k.upper() for tok in ("PIPE_M", "CUBE", "MAD", "FIX"))
    )
    vec_cycles = sum(
        max(0, int(v))
        for k, v in pipes.items()
        if any(tok in k.upper() for tok in ("PIPE_V", "VEC", "SIMT"))
    )
    mte_cycles = sum(
        max(0, int(v))
        for k, v in pipes.items()
        if any(tok in k.upper() for tok in ("MTE2", "MTE3", "MTE1", "DMA"))
    )
    subtotal = float(aic_cycles + vec_cycles + mte_cycles)
    if subtotal <= 0.0:
        all_cycles = float(sum(max(0, int(v)) for v in pipes.values()))
        if all_cycles <= 0.0:
            return 0.0, 0.0, 0.0
        return 0.0, 1.0, 0.0
    return (
        round(float(aic_cycles) / subtotal, 6),
        round(float(vec_cycles) / subtotal, 6),
        round(float(mte_cycles) / subtotal, 6),
    )


def _wrap_funnel_with_proxy(
    funnel: GateFunnel,
    target: AscendTarget | None,
    *,
    enable_track_router: bool = False,
    ideal_roofline_us: float = 0.0,
) -> GateFunnel:
    def _proxy_device(_dsl: str, stage_a: StageAResult) -> StageCResult:
        assert stage_a.ledger is not None
        lat = _proxy_latency_from_ledger(stage_a.ledger)
        aic_mad, aiv_vec, aiv_mte2 = _derive_proportional_pmu_ratios(stage_a.ledger)
        telemetry: dict[str, object] = {
            "dur_us": lat,
            "aic_mad": aic_mad,
            "aiv_vec": aiv_vec,
            "aiv_mte2": aiv_mte2,
            "is_analytical_proxy": True,
        }
        if ideal_roofline_us > 0.0 and lat > 0.0:
            telemetry["roofline_eta"] = min(1.0, ideal_roofline_us / max(1e-6, lat))
        return StageCResult(
            passed=True,
            accuracy_passed=True,
            measured_latency_us=lat,
            residual=compute_static_dynamic_residual(stage_a.ledger, telemetry),
            telemetry=telemetry,
        )

    eff_router = enable_track_router or funnel._enable_track_router
    if funnel._device_runner is not None:
        if eff_router == funnel._enable_track_router:
            return funnel
        return GateFunnel(
            target=target or funnel._target,
            skip_stage_b=funnel._skip_stage_b,
            compile_runner=funnel._compile_runner,
            device_runner=funnel._device_runner,
            enable_track_router=eff_router,
        )
    return GateFunnel(
        target=target or funnel._target,
        skip_stage_b=funnel._skip_stage_b,
        compile_runner=funnel._compile_runner,
        device_runner=_proxy_device,
        enable_track_router=eff_router,
    )


async def run_e2e_evolution(
    *,
    problem_name: str,
    hir_source: str,
    shape_contract: str,
    backend: KernelLLMBackend,
    funnel: GateFunnel,
    tracker: BayesianRuleTracker | None = None,
    beam: TwoLaneBeamSearch | None = None,
    initial_dsl_source: str | None = None,
    factory_name: str | None = None,
    factory_kwargs: Mapping[str, object] | None = None,
    num_rounds: int = 2,
    k_rule_workers: int = 2,
    max_syntax_repairs: int = 1,
    max_seed_repairs: int = 2,
    target: AscendTarget | None = None,
    explore_stepping_stone_on_stagnation: bool = True,
    enable_track_router: bool = False,
    scheduled_hir_source: str | None = None,
    anchor_dims: Mapping[str, int] | None = None,
) -> EvolutionTrajectory:
    """Drive dual-loop causal + creative evolution from unscheduled HIR to converged DSL."""
    active_tracker = tracker or BayesianRuleTracker.from_seed_yaml()
    active_beam = beam or TwoLaneBeamSearch()
    nav_packet = _compile_nav_packet_safe(
        hir_source,
        scheduled_hir_source=scheduled_hir_source,
        anchor_dims=anchor_dims,
        target=target,
    )
    ideal_roofline_us = nav_packet.roofline.ideal_roofline_us if nav_packet is not None else 0.0
    nav_context = format_hir_tir_navigation_section(nav_packet) if nav_packet is not None else ""
    active_funnel = _wrap_funnel_with_proxy(
        funnel,
        target,
        enable_track_router=enable_track_router,
        ideal_roofline_us=ideal_roofline_us,
    )

    if initial_dsl_source is not None:
        baseline_dsl = initial_dsl_source
        base_eval = active_funnel.evaluate(
            baseline_dsl,
            factory_name=factory_name,
            factory_kwargs=factory_kwargs,
        )
    else:
        baseline_dsl, base_eval = await synthesize_initial_dsl(
            hir_source=hir_source,
            shape_contract=shape_contract,
            backend=backend,
            funnel=active_funnel,
            factory_name=factory_name,
            factory_kwargs=factory_kwargs,
            max_seed_repairs=max_seed_repairs,
            target=target,
            enable_track_router=enable_track_router,
            scheduled_hir_source=scheduled_hir_source,
            anchor_dims=anchor_dims,
            nav_packet=nav_packet,
        )

    if not base_eval.passed or base_eval.stage_a is None or base_eval.stage_a.ledger is None:
        raise RuntimeError(
            f"Initial HIR baseline failed funnel ({base_eval.failed_stage}): "
            f"{base_eval.error_message}"
        )

    stage_a = base_eval.stage_a
    base_ledger = stage_a.ledger
    base_track = (
        stage_a.route.track
        if enable_track_router and stage_a.route is not None
        else (stage_a.route.track if stage_a.route is not None else "unconstrained")
    )
    base_features = (
        stage_a.features.to_dict()
        if enable_track_router and stage_a.features is not None
        else {}
    )
    if base_eval.stage_c and base_eval.stage_c.residual is not None:
        base_tel = dict(base_eval.stage_c.telemetry)
        base_dur = float(base_tel.get("dur_us", base_eval.stage_c.measured_latency_us))
        if ideal_roofline_us > 0.0 and base_dur > 0.0 and "roofline_eta" not in base_tel:
            base_tel["roofline_eta"] = min(1.0, ideal_roofline_us / max(1e-6, base_dur))
            base_residual = compute_static_dynamic_residual(base_ledger, base_tel)
        else:
            base_residual = base_eval.stage_c.residual
    else:
        proxy_lat = _proxy_latency_from_ledger(base_ledger)
        aic_mad, aiv_vec, aiv_mte2 = _derive_proportional_pmu_ratios(base_ledger)
        proxy_tel: dict[str, object] = {
            "dur_us": proxy_lat,
            "aic_mad": aic_mad,
            "aiv_vec": aiv_vec,
            "aiv_mte2": aiv_mte2,
            "is_analytical_proxy": True,
        }
        if ideal_roofline_us > 0.0 and proxy_lat > 0.0:
            proxy_tel["roofline_eta"] = min(1.0, ideal_roofline_us / max(1e-6, proxy_lat))
        base_residual = compute_static_dynamic_residual(base_ledger, proxy_tel)
    base_lat = (
        base_eval.stage_c.measured_latency_us
        if base_eval.stage_c and base_eval.stage_c.measured_latency_us > 0.0
        else _proxy_latency_from_ledger(base_ledger)
    )

    baseline_cand = BeamCandidate(
        candidate_id="r0_baseline",
        round_idx=0,
        worker_kind="hir_baseline",
        rule_id=None,
        dsl_source=baseline_dsl,
        track=base_track,
        ledger=base_ledger,
        residual=base_residual,
        latency_us=base_lat,
        accuracy_passed=True,
        structural_signature=extract_structural_signature(baseline_dsl, track=base_track),
        hypothesis="unscheduled_hir_baseline",
        parent_id=None,
        features=stage_a.features if enable_track_router else None,
    )
    active_beam.update([baseline_cand])

    runner = PortfolioRunner(
        backend=backend,
        funnel=active_funnel,
        tracker=active_tracker,
        beam=active_beam,
        k_rule_workers=k_rule_workers,
        max_syntax_repairs=max_syntax_repairs,
        enable_track_router=enable_track_router,
    )

    rounds: list[EvolutionRoundRecord] = []
    best_cand = baseline_cand
    best_so_far_lat = baseline_cand.latency_us
    stagnated = False
    factory_kw_dict = dict(factory_kwargs) if factory_kwargs is not None else None

    for round_idx in range(1, max(1, num_rounds) + 1):
        if (
            explore_stepping_stone_on_stagnation
            and stagnated
            and active_beam.stepping_stone_beam
        ):
            parent = active_beam.stepping_stone_beam[0]
        elif active_beam.exploit_beam:
            parent = active_beam.exploit_beam[0]
        else:
            parent = baseline_cand

        active_beam.mark_parent_selected(parent)
        prev_best_lat = best_so_far_lat
        report = await runner.run_round(
            parent,
            round_idx=round_idx,
            factory_name=factory_name,
            factory_kwargs=factory_kw_dict,
            navigation_context=nav_context,
            ideal_roofline_us=ideal_roofline_us,
            nav_packet=nav_packet,
        )
        round_rec, best_so_far_lat = _build_round_record(report, parent, best_so_far_lat)
        rounds.append(round_rec)
        if active_beam.exploit_beam and active_beam.exploit_beam[0].latency_us <= best_cand.latency_us:
            best_cand = active_beam.exploit_beam[0]
        stagnated = best_so_far_lat >= prev_best_lat - 1e-9

    speedup = (
        baseline_cand.latency_us / best_cand.latency_us
        if best_cand.latency_us > 0.0
        else 1.0
    )
    return EvolutionTrajectory(
        problem_name=problem_name,
        track=base_track,
        feature_vector=base_features,
        total_rounds=len(rounds),
        baseline_candidate_id=baseline_cand.candidate_id,
        baseline_latency_us=baseline_cand.latency_us,
        baseline_estimated_cycles=_ledger_estimated_cycles(baseline_cand.ledger),
        best_candidate_id=best_cand.candidate_id,
        best_latency_us=best_cand.latency_us,
        best_estimated_cycles=_ledger_estimated_cycles(best_cand.ledger),
        speedup_vs_baseline=speedup,
        best_dsl_source=best_cand.dsl_source,
        rounds=tuple(rounds),
        active_rules_after=tuple(r.rule_id for r in active_tracker.active_rules()),
        hir_navigation_summary=_summarize_nav_packet(nav_packet),
    )


__all__ = [
    "EvolutionRoundRecord",
    "EvolutionTrajectory",
    "WorkerStepRecord",
    "build_hir_seed_request",
    "run_e2e_evolution",
    "synthesize_initial_dsl",
]
