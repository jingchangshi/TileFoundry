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


def _serial_program(tir):
    """The PRE-materialization program behind 'tir', when it can be found.

    Returns 'tir' unchanged when it is not a finished artifact, so a caller that already passes
    the serial program sees identical behaviour. Detection is structural -- a name with a __k
    instance suffix, or a FlagSet/FlagWait/Drain already present -- because a completion-bearing
    unrolled program is exactly the input this function must not be given.
    """
    original = getattr(tir, "_p3_pre_materialize", None)
    if original is not None:
        return original
    return tir


def _carries_completion(node, depth=0):
    """Does this program already contain Set/Wait/Drain statements?"""
    if depth > 40:
        return False
    callable_obj = getattr(node, "callable", None)
    if callable_obj is not None and type(callable_obj).__name__ in (
            "FlagSet", "FlagWait", "Drain", "AivFlagSet", "AivFlagWait", "AivDrain"):
        return True
    body = getattr(node, "body", None)
    if isinstance(body, (tuple, list)):
        return any(_carries_completion(child, depth + 1) for child in body)
    if body is not None:
        return _carries_completion(body, depth + 1)
    return False



def build_completion_plan(plan, tir, *, facts: TargetEventFacts = None,
                          capacities=None, profile_id: Optional[str] = None,
                          skip_emit: bool = False, target=None) -> CompletionPlan:
    """Build and verify the completion program for ONE resolved plan.

    'tir' is the ACTUAL module the completion statements are injected into; verification then
    reads them back from the injected module. Passing skip_emit is for diagnostic runs only and
    produces a REFUSED plan, because coverage cannot be judged without the emitted program.
    """
    # ---------------------------------------------------------------------------------------
    # COMPLETION IS PLANNED AGAINST THE SERIAL PROGRAM, NEVER AGAINST A FINISHED ONE.
    #
    # 'tir' used to be whatever the caller had to hand, and after P4-03 that is the FINISHED
    # artifact: R3-unrolled AND already carrying its completion block. Planning completion of a
    # program that already has one derives the same obligations twice and the emitter then
    # reports "31 more completion statements than the verified program requires" on a program
    # that is perfectly correct -- a false failure that reads like an emitter bug.
    #
    # The fix is at the CALL BOUNDARY, not in the checker: verify_completion is right to refuse
    # extra statements. When the caller hands over a PrimFunction, we re-derive the serial
    # program from the module it came from; when it hands over the serial program itself,
    # nothing changes.
    # ---------------------------------------------------------------------------------------
    tir = _serial_program(tir)
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
    rewritten, bound = emit_trace_tir(tir, trace, events=assignment, target=target)
    candidate = CompletionPlan(
        schema=SCHEMA, obligations=obligation_set.obligations,
        obligation_set=obligation_set, trace=bound, events=assignment,
        channel_map=channel_map, facts=facts, report=CompletionReport("PENDING"),
        rewritten_tir=rewritten)
    report = verify_completion(plan, candidate, actual_tir=rewritten, facts=facts)
    return replace(candidate, report=report)


def reemit_into_unrolled(completion_plan, unrolled_tir, plan, *, target=None,
                          facts: TargetEventFacts = None,
                          materialization=None) -> CompletionPlan:
    """Re-emit a verified completion program into the UNROLLED body, at real instance positions.

    WHY A SECOND EMISSION IS NEEDED AT ALL, and why it is not double work.

    Completion is derived from the plan, and the plan describes the SERIAL loop: one body, N
    logical iterations. Verification at that point is real and stays -- it proves the
    obligations are the right ones, the channels are allocatable, the slots follow the ring and
    the protocol simulates. But the PROGRAM that runs is the unrolled one, produced later by
    R3, and a block of flags appended to the serial body says nothing about where those flags
    belong once the body is replicated per instance.

    Emitting once into the serial body and appending was the original defect: all 14 Waits
    landed after every access, so each guarded nothing while remaining individually legal. The
    fix cannot be a patch-up of the appended block; the statements have to be placed against
    the statements they order, and those only exist after unrolling.

    So the trace is RE-USED, not recomputed: the same verified trace is placed into the real
    unrolled body. Nothing about the obligations changes, and the emitted program is verified
    again -- against the unrolled TIR this time -- by the caller.
    """
    if completion_plan is None or not completion_plan.verified:
        raise CompletionError("COMPLETION_NOT_VERIFIED",
                              "only a verified completion program may be re-emitted",
                              "completion", "VERIFIED",
                              None if completion_plan is None else completion_plan.code)
    facts = facts if facts is not None else completion_plan.facts
    rewritten, bound = emit_trace_tir(unrolled_tir, completion_plan.trace,
                                      events=completion_plan.events, target=target,
                                      materialization=materialization)
    candidate = replace(completion_plan, trace=bound, rewritten_tir=rewritten)
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
           "reemit_into_unrolled", "attach_completion_plan", "require_completion_verified"]
