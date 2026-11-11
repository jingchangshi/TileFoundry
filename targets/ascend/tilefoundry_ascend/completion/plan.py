"""Orchestration: resolved plan -> obligations -> events -> trace -> verified completion plan.

THE ORDER IS THE SPEC's ORDER (3.5 before 3.6 before 3.7), and each stage can refuse on its
own:

  1. derive the logical obligations from the plan's real accesses;
  2. build the instance-ordered completion trace (initial slots, short loops, guards, drain);
  3. allocate finite event ids per real namespace, from the trace's real lifetimes;
  4. emit the trace into the ACTUAL TIR and verify the ACTUAL result.

Nothing here reads a latency, a cycle count, or a "verified" flag it wrote earlier. A
completion plan that does not verify is RETURNED as a refused record with its named code
rather than raised, so the chain can record the specific gap;
require_completion_verified is the raising form used by the export path.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from .emit import emit_trace_tir
from .events import EventAssignment, allocate_events
from .model import CompletionError, TargetEventFacts, event_facts_for
from .obligations import ObligationSet, derive_obligations
from .schedule import CompletionTrace, build_completion_trace
from .verify import CompletionReport, verify_completion

#: Where a verified completion plan travels on Module.metadata, and the key the export gate
#: reads. It is a declared place a dataclass is stored, not a private side channel.
COMPLETION_PLAN_KEY = "ascend.completion_plan"

SCHEMA = "ascend.completion/1"


@dataclass(frozen=True)
class CompletionPlan:
    """The completion program plus everything needed to re-judge it."""

    schema: str
    obligations: tuple
    obligation_set: ObligationSet
    trace: CompletionTrace
    events: EventAssignment
    channel_map: dict
    facts: TargetEventFacts
    report: CompletionReport
    rewritten_tir: object = None
    diagnostics: tuple = ()

    @property
    def status(self) -> str:
        return self.report.status

    @property
    def verified(self) -> bool:
        return self.report.verified

    @property
    def code(self) -> str:
        return self.report.code

    def as_dict(self):
        return {"schema": self.schema,
                "status": self.report.status, "code": self.report.code,
                "required": self.report.required, "covered": self.report.covered,
                "required_by_kind": dict(self.report.required_by_kind),
                "covered_by_kind": dict(self.report.covered_by_kind),
                "trip": self.trace.trip, "depth": self.trace.depth,
                "channels": list(self.trace.channels),
                "records": len(self.trace.records),
                "events": self.events.as_dict(),
                "namespaces": dict(self.report.namespaces),
                "cross_checks": [dict(item) for item in self.report.cross_checks],
                "obligations": [obligation.as_dict() for obligation in self.obligations],
                "channel_map": dict(self.channel_map),
                "diagnostics": list(self.diagnostics),
                "result_digest": self.report.result_digest}


def _channel_map(obligations, trace) -> dict:
    """obligation -> the channel that carries it. Derived, so it cannot drift from the trace."""
    mapping = {}
    for record in trace.records:
        if record.obligation_id and record.role in ("set", "wait"):
            mapping.setdefault(record.obligation_id, record.channel)
    for obligation in obligations:
        mapping.setdefault(obligation.obligation_id, "")
    return {key: value for key, value in mapping.items() if value}


def build_completion_plan(plan, tir, *, facts: TargetEventFacts = None,
                          capacities=None, profile_id: Optional[str] = None,
                          skip_emit: bool = False) -> CompletionPlan:
    """Build and verify the completion program for ONE resolved plan.

    'tir' is the ACTUAL module the completion statements are injected into; verification then
    reads them back from the injected module. Passing skip_emit is for diagnostic runs only and
    produces a REFUSED plan, because coverage cannot be judged without the emitted program.
    """
    facts = event_facts_for(profile_id, facts=facts)
    obligation_set = derive_obligations(plan, facts=facts)
    trace = build_completion_trace(plan, obligation_set, facts=facts)
    assignment = allocate_events(trace.channel_requests(), facts=facts,
                                 capacities=capacities)
    channel_map = _channel_map(obligation_set.obligations, trace)
    if skip_emit:
        return CompletionPlan(
            schema=SCHEMA, obligations=obligation_set.obligations,
            obligation_set=obligation_set, trace=trace, events=assignment,
            channel_map=channel_map, facts=facts,
            report=CompletionReport("FAILED", code="COMPLETION_NOT_VERIFIED",
                                    detail="skip_emit was requested, so no actual program"
                                           " exists to verify"),
            rewritten_tir=None,
            diagnostics=("emission skipped",))
    rewritten, bound = emit_trace_tir(tir, trace, events=assignment)
    candidate = CompletionPlan(
        schema=SCHEMA, obligations=obligation_set.obligations,
        obligation_set=obligation_set, trace=bound, events=assignment,
        channel_map=channel_map, facts=facts, report=CompletionReport("PENDING"),
        rewritten_tir=rewritten)
    report = verify_completion(plan, candidate, actual_tir=rewritten, facts=facts)
    return replace(candidate, report=report)


def attach_completion_plan(plan, completion_plan: CompletionPlan):
    """Attach a VERIFIED completion plan to a ResolvedPlan (P4-01's half of the R5 boundary)."""
    from ..plan.resolved_plan import PlanError, attach_completion

    if completion_plan is None or not completion_plan.verified:
        code = "" if completion_plan is None else completion_plan.code
        raise PlanError("COMPLETION_NOT_VERIFIED",
                        "the completion program did not verify: " + (code or "no plan"),
                        "completion", "a VERIFIED CompletionPlan", code or None)
    return attach_completion(plan, completion_plan.report)


def require_completion_verified(completion_plan: CompletionPlan) -> CompletionPlan:
    """The raising form: a caller that needs a verified plan gets one or a named refusal."""
    if completion_plan is None or not completion_plan.verified:
        code = "" if completion_plan is None else (completion_plan.code or "")
        detail = ("" if completion_plan is None else completion_plan.report.detail)
        raise CompletionError(code or "COMPLETION_NOT_VERIFIED",
                              "the completion program is not verified: " + detail,
                              "completion", "VERIFIED", code or None)
    return completion_plan


__all__ = ["COMPLETION_PLAN_KEY", "SCHEMA", "CompletionPlan", "build_completion_plan",
           "attach_completion_plan", "require_completion_verified"]
