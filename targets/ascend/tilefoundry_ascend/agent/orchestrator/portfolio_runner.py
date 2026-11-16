"""Portfolio evolution runner: K=2 causal rule workers + 1 rule-blind creative worker."""

from __future__ import annotations

import asyncio
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
    StaticDynamicResidual,
    TwoLaneBeamSearch,
    extract_structural_signature,
)


@dataclass(frozen=True)
class WorkerSpec:
    """Spawned worker specification for one optimization portfolio round."""

    worker_id: str
    worker_kind: str
    rule: AscendCausalRule | None
    request: GenerationRequest
    parent_candidate: BeamCandidate


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


def build_exploit_request(
    parent: BeamCandidate,
    rule: AscendCausalRule,
) -> GenerationRequest:
    """Build causal-rule worker request routed to exploit model (Hydrogen Medium)."""
    sys_prompt = (
        "You are an Ascend 950PR TileLang DSL optimization engineer. Apply the given causal rule "
        "faithfully to eliminate the bottleneck identified in the physical ledger while "
        "preserving exact numerical equivalence. Output ONLY valid Python code."
    )
    user_prompt = (
        f"### TARGET TRACK\n{parent.track}\n\n"
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


def build_creative_request(parent: BeamCandidate) -> GenerationRequest:
    """Build rule-blind creative search request routed to Hydrogen High without rules."""
    sys_prompt = (
        "You are a first-principles Ascend 950PR (dav-3510) systems architect. Propose an "
        "architectural kernel optimization directly from hardware limits (1 AIC + 2 AIV coupled "
        "block, 3 KiB SSBUF, L0C->UB direct path, UB->L1 direct path, 16-entry MTE2 queue, "
        "128 B/cycle FixPipe) and the physical telemetry below. Return JSON matching the schema."
    )
    user_prompt = (
        f"### EXECUTION TRACK\n{parent.track}\n\n"
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
    ) -> None:
        self._backend = backend
        self._funnel = funnel
        self._tracker = tracker
        self._beam = beam or TwoLaneBeamSearch()
        self._k_rules = max(1, k_rule_workers)
        self._max_repairs = max(0, max_syntax_repairs)

    @property
    def beam(self) -> TwoLaneBeamSearch:
        """Return active TwoLaneBeamSearch instance."""
        return self._beam

    def plan_workers(self, parent: BeamCandidate, round_idx: int) -> tuple[WorkerSpec, ...]:
        """Select top-K matching causal rules plus one rule-blind creative explorer."""
        ctx = build_predicate_context(
            parent.ledger,
            residual=parent.residual,
        )
        matched_rules = list(self._tracker.rank_applicable_rules(ctx, top_k=self._k_rules))
        if len(matched_rules) < self._k_rules:
            already = {r.rule_id for r in matched_rules}
            for rule in self._tracker.active_rules():
                if rule.rule_id not in already:
                    matched_rules.append(rule)
                    if len(matched_rules) >= self._k_rules:
                        break

        specs: list[WorkerSpec] = []
        for idx, rule in enumerate(matched_rules[: self._k_rules]):
            specs.append(
                WorkerSpec(
                    worker_id=f"r{round_idx}_rule_{idx}",
                    worker_kind="rule_exploit",
                    rule=rule,
                    request=build_exploit_request(parent, rule),
                    parent_candidate=parent,
                )
            )
        specs.append(
            WorkerSpec(
                worker_id=f"r{round_idx}_creative",
                worker_kind="creative_explore",
                rule=None,
                request=build_creative_request(parent),
                parent_candidate=parent,
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
            return None, False, (spec.worker_id, "llm_generate", str(exc)[-400:])

        repaired = False
        evaluation: FunnelEvaluation = self._funnel.evaluate(
            dsl_source,
            factory_name=factory_name,
            factory_kwargs=factory_kwargs,
        )
        if not evaluation.passed and evaluation.failed_stage in ("stage_a", "stage_b"):
            for _ in range(self._max_repairs):
                repair_req = build_syntax_repair_request(dsl_source, evaluation.error_message)
                repaired_raw = await self._backend.generate(repair_req)
                dsl_source = extract_python_dsl(repaired_raw)
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
        track = evaluation.stage_a.route.track if evaluation.stage_a.route else spec.parent_candidate.track
        residual = (
            evaluation.stage_c.residual
            if evaluation.stage_c and evaluation.stage_c.residual is not None
            else spec.parent_candidate.residual
        )
        latency_us = (
            evaluation.stage_c.measured_latency_us
            if evaluation.stage_c
            else spec.parent_candidate.latency_us
        )
        acc_ok = evaluation.stage_c.accuracy_passed if evaluation.stage_c else True

        if spec.rule is not None:
            post_ctx = build_predicate_context(
                ledger,
                features=evaluation.stage_a.features,
                residual=residual,
                prev_ledger=spec.parent_candidate.ledger,
            )
            realized = spec.rule.postcondition_realized(post_ctx)
            parent_us = spec.parent_candidate.latency_us
            speed_gain = ((parent_us - latency_us) / parent_us) if parent_us > 0.0 and latency_us > 0.0 else 0.0
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
        )
        return cand, repaired, None

    async def run_round(
        self,
        parent: BeamCandidate,
        *,
        round_idx: int = 1,
        factory_name: str | None = None,
        factory_kwargs: dict[str, object] | None = None,
    ) -> PortfolioRoundReport:
        """Execute one parallel portfolio round across Rule and Creative workers."""
        specs = self.plan_workers(parent, round_idx)
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
    "PortfolioRoundReport",
    "PortfolioRunner",
    "WorkerSpec",
    "build_creative_request",
    "build_exploit_request",
    "build_syntax_repair_request",
]
