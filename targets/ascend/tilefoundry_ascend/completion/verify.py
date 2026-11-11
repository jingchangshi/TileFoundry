"""Independent verification of a completion program.

THE RULE. The verifier does not trust what the builder said. It performs FOUR independent
checks, and each can fail on its own with its own code:

  A. RE-DERIVATION. The obligation set is derived from the plan a second time, by this module,
     from the plan's own accesses. An obligation the builder omitted or invented is refused
     here, so "the set is derived, not templated" is checked rather than asserted.

  B. ALLOCATION RE-CHECK. The event assignment is re-walked against the target's namespaces:
     same id in independent namespaces is fine, overlapping lifetimes inside one namespace are
     FLAG_RESOURCE_CONFLICT, and a namespace holding more ids than its capacity is
     FLAG_POOL_EXHAUSTED.

  C. PROGRAM-LEVEL PROTOCOL REPLAY. The completion program itself -- the instance-ordered
     trace, whose access points come from the plan's REAL accesses and task instances -- is
     replayed through the ring protocol state machine. This is where early FREE, a read before
     READY, an overwrite before FREE, a replayed old generation, a wait with no Set, an
     incompatible guard, a disagreeing cross-core handoff and a live slot at region exit are
     decided. It works on instances because ONLY the plan knows where the real reads and
     writes are; the emitted flag statements do not carry that.

  D. EMISSION FIDELITY. The ACTUAL TIR is read back and compared with the program verified in
     C, statement by statement and in order: same role, channel, kind, generation, slot,
     versions, event id, and the same sequence. This is what closes the gap C leaves: C
     verifies a program, D proves the program that will run IS that program. A dropped
     statement, a changed slot, a changed generation, a moved Wait, a forged channel or an
     unreachable (statically dead) statement is refused here.

C and D compose: C is a proof about a program, D is a proof that the emitted program is that
program. Neither is sufficient alone, and saying so is the point -- a coverage COUNT would
have claimed both while proving neither.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .emit import read_trace_from_tir
from .events import check_assignment
from .model import (
    KIND_EXIT,
    CompletionError,
    digest,
)
from .obligations import derive_obligations
from .protocol import simulate_trace

STATUS_VERIFIED = "VERIFIED"
STATUS_FAILED = "FAILED"

#: The attributes that make one emitted completion statement the statement the program asked
#: for. A difference in ANY of them is a different program, not a formatting difference.
_STATEMENT_ATTRS = ("role", "channel", "kind", "generation", "slot", "versions", "event_id")


@dataclass(frozen=True)
class CompletionReport:
    status: str
    code: str = ""
    detail: str = ""
    required: int = 0
    covered: int = 0
    missing: tuple = ()
    required_by_kind: dict = field(default_factory=dict)
    covered_by_kind: dict = field(default_factory=dict)
    protocol: dict = field(default_factory=dict)
    namespaces: dict = field(default_factory=dict)
    cross_checks: tuple = ()
    records: int = 0
    statements: int = 0
    result_digest: str = ""

    @property
    def verified(self) -> bool:
        return self.status == STATUS_VERIFIED

    def as_dict(self):
        return {"status": self.status, "code": self.code, "detail": self.detail,
                "required": self.required, "covered": self.covered,
                "missing": list(self.missing),
                "required_by_kind": dict(self.required_by_kind),
                "covered_by_kind": dict(self.covered_by_kind),
                "protocol": dict(self.protocol), "namespaces": dict(self.namespaces),
                "cross_checks": [dict(item) for item in self.cross_checks],
                "records": self.records, "statements": self.statements,
                "result_digest": self.result_digest}


def _counts(obligations):
    counts = {}
    for obligation in obligations:
        counts[obligation.kind] = counts.get(obligation.kind, 0) + 1
    return counts


def _failed(code, detail, required=0, covered=0, missing=(), required_by_kind=None,
            covered_by_kind=None, protocol=None, namespaces=None, cross=(), records=0,
            statements=0):
    return CompletionReport(STATUS_FAILED, code=code, detail=detail, required=required,
                            covered=covered, missing=tuple(missing),
                            required_by_kind=dict(required_by_kind or {}),
                            covered_by_kind=dict(covered_by_kind or {}),
                            protocol=dict(protocol or {}), namespaces=dict(namespaces or {}),
                            cross_checks=tuple(cross), records=records,
                            statements=statements)


def _expected_statements(trace):
    """The completion statements the program consists of, in program order."""
    return [record for record in trace.records if record.role in ("set", "wait", "drain")]


def _signature(record):
    """The identity of one emitted statement, over the attributes its OP actually carries.

    A Drain carries none of channel/kind/generation/slot (see the core Drain op), so comparing
    those would compare two absences and report a false difference for every correct program.
    The set of compared attributes is therefore taken from the record's role, which is the only
    honest way to compare "the statement in the program" with "the statement the plan wanted".
    """
    if getattr(record, "role", None) == "drain":
        return ("drain",)
    return tuple(getattr(record, name, None) for name in _STATEMENT_ATTRS)


def verify_completion(plan, completion_plan, *, actual_tir=None, facts=None) -> CompletionReport:
    """Verify a completion plan against the plan it claims to serve and the ACTUAL TIR."""
    from .plan import CompletionPlan  # local import: avoids a cycle at module load

    if not isinstance(completion_plan, CompletionPlan):
        raise CompletionError("COMPLETION_NOT_VERIFIED",
                              "the object under test is not a CompletionPlan", "input",
                              "CompletionPlan", type(completion_plan).__name__)
    assignment = completion_plan.events
    if assignment is None:
        return _failed("COMPLETION_NOT_VERIFIED", "no physical event allocation was produced")
    facts = facts or completion_plan.facts

    # ---- A. re-derive the obligations from the plan ITSELF.
    try:
        expected = derive_obligations(plan, facts=facts)
    except CompletionError as error:
        return _failed(error.code, error.message)
    required = tuple(expected.obligations)
    required_by_kind = _counts(required)
    supplied_ids = {o.obligation_id for o in completion_plan.obligations}
    expected_ids = {o.obligation_id for o in required}
    missing_obligations = tuple(sorted(expected_ids - supplied_ids))
    if missing_obligations:
        return _failed("COMPLETION_SOURCE_COVERAGE_MISSING",
                       "the completion plan omits %d obligation(s) the plan's own accesses"
                       " require: %r" % (len(missing_obligations),
                                         list(missing_obligations)[:6]),
                       required=len(required), missing=missing_obligations,
                       required_by_kind=required_by_kind)
    invented = tuple(sorted(supplied_ids - expected_ids))
    if invented:
        return _failed("COMPLETION_OBLIGATION_UNSUPPORTED",
                       "the completion plan carries %d obligation(s) no plan access supports:"
                       " %r" % (len(invented), list(invented)[:6]),
                       required=len(required), missing=invented,
                       required_by_kind=required_by_kind)

    # ---- B. re-check the event allocation.
    try:
        namespaces = check_assignment(assignment, facts=facts)
    except CompletionError as error:
        return _failed(error.code, error.message, required=len(required),
                       required_by_kind=required_by_kind)

    trace = completion_plan.trace
    if trace is None:
        return _failed("COMPLETION_NOT_VERIFIED", "no completion program was built",
                       required=len(required), required_by_kind=required_by_kind)

    # ---- C. COVERAGE against an INDEPENDENTLY built program.
    #       This layer exists because the other two cannot see this defect: the trace under
    #       test may have been produced from a program that simply omits a wait, in which case
    #       the protocol replay verifies the omission and the fidelity check compares the
    #       omission with itself. Coverage asks the plan, once more and separately, which
    #       statements its obligations require, and demands the ACTUAL program carry them.
    expected_trace = _independent_program(plan, expected, assignment, facts)
    if actual_tir is None:
        return _failed("COMPLETION_NOT_VERIFIED",
                       "no actual TIR was supplied, so coverage cannot be judged",
                       required=len(required), required_by_kind=required_by_kind,
                       namespaces=namespaces["namespaces"])
    actual = read_trace_from_tir(actual_tir)
    covered, missing, covered_by_kind = _coverage(expected_trace, actual)
    if missing:
        return _failed("COMPLETION_NOT_VERIFIED",
                       "the actual program does not carry %d statement(s) the plan's"
                       " obligations require: %r" % (len(missing), missing[:6]),
                       required=len(required), covered=covered, missing=missing,
                       required_by_kind=required_by_kind, covered_by_kind=covered_by_kind,
                       namespaces=namespaces["namespaces"], records=len(trace.records),
                       statements=len(actual))

    # ---- D. replay the PROGRAM through the ring protocol.
    verdict = simulate_trace(trace, obligations=required)

    # ---- E. emission fidelity: the ACTUAL TIR must BE the verified program.
    fidelity = _emission_fidelity(trace, actual)
    if fidelity is not None:
        code, detail = fidelity
        return _failed(code, detail, required=len(required), covered=0,
                       required_by_kind=required_by_kind,
                       covered_by_kind=covered_by_kind, protocol=verdict.as_dict(),
                       namespaces=namespaces["namespaces"],
                       cross=verdict.cross_checks, records=len(trace.records),
                       statements=len(actual))
    if verdict.status != STATUS_VERIFIED:
        return _failed(verdict.code or "COMPLETION_NOT_VERIFIED", verdict.detail,
                       required=len(required), covered=0,
                       required_by_kind=required_by_kind,
                       covered_by_kind={}, protocol=verdict.as_dict(),
                       namespaces=namespaces["namespaces"],
                       cross=verdict.cross_checks, records=len(trace.records),
                       statements=len(actual))
    payload = {"required": required_by_kind, "covered": covered_by_kind,
               "namespaces": namespaces["namespaces"],
               "statements": len(actual), "trip": expected.trip,
               "protocol": verdict.counts}
    return CompletionReport(STATUS_VERIFIED, required=len(required),
                            covered=len(required), required_by_kind=required_by_kind,
                            covered_by_kind=covered_by_kind,
                            protocol=verdict.as_dict(),
                            namespaces=namespaces["namespaces"],
                            cross_checks=verdict.cross_checks, records=len(trace.records),
                            statements=int(len(actual)),
                            result_digest=digest(payload))


def _independent_program(plan, expected, assignment, facts):
    """A SECOND, separate program built from the plan's obligations alone.

    Built by re-running the schedule builder from scratch rather than by reusing the trace under
    test, so this is an independent statement of what the plan requires. It is the coverage
    base: if the program under test omits a wait, this still contains it.
    """
    from .schedule import build_completion_trace

    return build_completion_trace(plan, expected, facts=facts)


def _coverage(expected_trace, actual):
    """Which of the plan's REQUIRED statements does the ACTUAL program carry?

    Identity is (channel base, kind, generation, role). The Drain is keyed on its role alone
    because the core Drain op carries no channel or generation -- comparing those would compare
    two absences.
    """
    present = {}
    for record in actual:
        if record.role not in ("set", "wait", "drain"):
            continue
        if not getattr(record, "reachable", True):
            continue
        key = _coverage_key(record)
        present[key] = present.get(key, 0) + 1
    missing = []
    covered = 0
    for record in expected_trace.records:
        if record.role not in ("set", "wait", "drain"):
            continue
        key = _coverage_key(record)
        if present.get(key, 0) < 1:
            missing.append(_describe(record))
        else:
            covered += 1
    return covered, missing, {}


def _coverage_key(record):
    if record.role == "drain":
        return ("", "", -1, "drain")
    return (str(record.channel).split("#")[0], str(record.kind).lower(),
            int(record.generation), record.role)


def _emission_fidelity(trace, actual):
    """Does the ACTUAL program consist of exactly the verified statements, in order?

    Returns None when it does, or (code, detail) naming the FIRST difference. The comparison is
    per statement and positional, so a dropped statement shifts everything after it and is
    caught at its own position rather than by a count that happens to still match.
    """
    expected = _expected_statements(trace)
    unreachable = [record for record in actual if not getattr(record, "reachable", True)]
    if unreachable:
        first = unreachable[0]
        return ("COMPLETION_UNREACHABLE_NODE",
                "the emitted program contains a completion statement that can never execute"
                " (%s %s generation %s): a statically dead Set/Wait is not coverage"
                % (first.role, first.channel, first.generation))
    if len(actual) < len(expected):
        missing = _first_missing(expected, actual)
        return ("COMPLETION_NOT_VERIFIED",
                "the emitted program carries %d completion statement(s) for the %d the"
                " verified program requires; first missing: %s"
                % (len(actual), len(expected),
                   _describe(missing) if missing is not None else "<unknown>"))
    if len(actual) > len(expected):
        extra = actual[len(expected)]
        return ("COMPLETION_OBLIGATION_UNSUPPORTED",
                "the emitted program carries %d more completion statement(s) than the verified"
                " program requires; first extra: %s %s generation %s"
                % (len(actual) - len(expected), extra.role, extra.channel, extra.generation))
    for index, (wanted, got) in enumerate(zip(expected, actual)):
        if _signature(wanted) != _signature(got):
            return ("COMPLETION_MAP_MISMATCH",
                    "emitted statement %d is %s but the verified program requires %s"
                    % (index, _describe(got), _describe(wanted)))
    return None


def _first_missing(expected, actual):
    for index, wanted in enumerate(expected):
        if index >= len(actual):
            return wanted
        if _signature(wanted) != _signature(actual[index]):
            return wanted
    return None


def _describe(record):
    if record is None:
        return "<none>"
    return "%s %s kind=%s generation=%s slot=%s versions=%s event=%s" % (
        record.role, record.channel or "<no-channel>", record.kind or "-",
        record.generation, record.slot, record.versions, record.event_id)


__all__ = ["STATUS_VERIFIED", "STATUS_FAILED", "CompletionReport", "verify_completion"]
