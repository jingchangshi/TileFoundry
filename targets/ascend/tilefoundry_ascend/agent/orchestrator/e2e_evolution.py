"""End-to-end HIR-to-kernel dual-loop agent evolution driver and trajectory tracker."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import tempfile

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
from tilefoundry_ascend.agent.router.capability_gate import (
    extract_structural_features,
    route_execution_track,
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
    feature_vector: dict[str, object]
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


def _inspect_hir_for_seed(
    hir_source: str,
    target: AscendTarget | None = None,
) -> tuple[str | None, str | None]:
    """Run L0 HIR pre-routing and optional TIR blueprint lowering on CPU."""
    try:
        from tilefoundry.analysis import analyze, check_program
        from tilefoundry.cli.source import load_authored_ir
        from tilefoundry.inspection import as_script
        from tilefoundry.schedule.finalize import finalize

        resolved_target = target or AscendTarget("ascend.950pr")
        with tempfile.TemporaryDirectory(prefix="tf_hir_seed_") as tmp_dir:
            hir_path = Path(tmp_dir) / "authored_hir.py"
            hir_path.write_text(hir_source, encoding="utf-8")
            mod = load_authored_ir(str(hir_path))
        entry_fn = mod.entry_function()
        features = extract_structural_features(entry_fn, target=resolved_target)
        route = route_execution_track(features)
        prims = ", ".join(route.recommended_primitives)
        route_summary = f"{route.track}: {prims}" if prims else route.track

        tir_blueprint: str | None = None
        try:
            checked_mod, checked_fn = check_program(mod, entry_fn)
            analyzed = analyze(checked_mod, checked_fn, analysis=("memory",))
            lowered = finalize(analyzed, authored=entry_fn)
            tir_blueprint = as_script(lowered.entry_function())
        except Exception:
            tir_blueprint = None
        return route_summary, tir_blueprint
    except Exception:
        return None, None


def build_hir_seed_request(
    hir_source: str,
    shape_contract: str,
    *,
    factory_name: str | None = None,
    recommended_track: str | None = None,
    recommended_primitives: Sequence[str] = (),
    tir_blueprint: str | None = None,
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
    route_section = ""
    if recommended_track:
        prims_text = ", ".join(recommended_primitives) if recommended_primitives else "default"
        route_section = (
            f"### L0 RECOMMENDED EXECUTION TRACK\n"
            f"Track: {recommended_track}\n"
            f"Recommended Primitives: {prims_text}\n\n"
        )
    blueprint_section = ""
    if tir_blueprint:
        blueprint_section = (
            f"### LOWERED ASCEND TIR BLUEPRINT\n"
            f"```python\n{tir_blueprint[:4000]}\n```\n\n"
        )
    user_prompt = (
        f"### SHAPE AND SIGNATURE CONTRACT\n{shape_contract.strip()}\n\n"
        f"{route_section}"
        f"{blueprint_section}"
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
) -> tuple[str, FunnelEvaluation]:
    """Synthesize and funnel-validate an initial @T.prim_func seed from HIR source."""
    route_summary, tir_blueprint = _inspect_hir_for_seed(hir_source, target=target)
    rec_track: str | None = None
    rec_prims: tuple[str, ...] = ()
    if route_summary:
        if ":" in route_summary:
            raw_track, raw_prims = route_summary.split(":", 1)
            rec_track = raw_track.strip()
            rec_prims = tuple(p.strip() for p in raw_prims.split(",") if p.strip())
        else:
            rec_track = route_summary.strip()
    req = build_hir_seed_request(
        hir_source,
        shape_contract,
        factory_name=factory_name,
        recommended_track=rec_track,
        recommended_primitives=rec_prims,
        tir_blueprint=tir_blueprint,
    )
    raw = await backend.generate(req)
    dsl_source = extract_python_dsl(raw)
    eval_res = funnel.evaluate(
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
        eval_res = funnel.evaluate(
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


def _wrap_funnel_with_proxy(funnel: GateFunnel, target: AscendTarget | None) -> GateFunnel:
    def _proxy_device(_dsl: str, stage_a: StageAResult) -> StageCResult:
        assert stage_a.ledger is not None
        lat = _proxy_latency_from_ledger(stage_a.ledger)
        telemetry = {"dur_us": lat, "aic_mad": 0.4, "aiv_vec": 0.4, "aiv_mte2": 0.2}
        return StageCResult(
            passed=True,
            accuracy_passed=True,
            measured_latency_us=lat,
            residual=compute_static_dynamic_residual(stage_a.ledger, telemetry),
            telemetry=telemetry,
        )

    if funnel._device_runner is not None:
        return funnel
    return GateFunnel(
        target=target or funnel._target,
        skip_stage_b=funnel._skip_stage_b,
        compile_runner=funnel._compile_runner,
        device_runner=_proxy_device,
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
) -> EvolutionTrajectory:
    """Drive dual-loop causal + creative evolution from unscheduled HIR to converged DSL."""
    active_tracker = tracker or BayesianRuleTracker.from_seed_yaml()
    active_beam = beam or TwoLaneBeamSearch()
    active_funnel = _wrap_funnel_with_proxy(funnel, target)

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
        )

    if not base_eval.passed or base_eval.stage_a is None or base_eval.stage_a.ledger is None:
        raise RuntimeError(f"Initial HIR baseline failed funnel ({base_eval.failed_stage}): {base_eval.error_message}")

    stage_a = base_eval.stage_a
    base_ledger = stage_a.ledger
    base_track = stage_a.route.track if stage_a.route else "developer"
    base_features = stage_a.features.to_dict() if stage_a.features else {}
    base_residual = (
        base_eval.stage_c.residual
        if base_eval.stage_c and base_eval.stage_c.residual is not None
        else compute_static_dynamic_residual(base_ledger, {"dur_us": _proxy_latency_from_ledger(base_ledger)})
    )
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
        features=stage_a.features,
    )
    active_beam.update([baseline_cand])

    runner = PortfolioRunner(
        backend=backend,
        funnel=active_funnel,
        tracker=active_tracker,
        beam=active_beam,
        k_rule_workers=k_rule_workers,
        max_syntax_repairs=max_syntax_repairs,
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
    )


__all__ = [
    "EvolutionRoundRecord",
    "EvolutionTrajectory",
    "WorkerStepRecord",
    "build_hir_seed_request",
    "run_e2e_evolution",
    "synthesize_initial_dsl",
]
