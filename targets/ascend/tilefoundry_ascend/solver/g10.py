"""G10: S-side consumption of the POC gate's complete expected obligations.

The input is the task-82 ExpectedObligationSet (RAW/SSA/WAR/FREE/WAW/REUSE/FINAL),
not the BYTES-only subset from s_hard_constraint_dependencies. The output keeps the
L5 s-problem/1 contract fields and four explicit solver states. A critical unknown or
a missing expected obligation makes the input NOT_READY and can never yield a SAT
assignment usable by R.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional

SCHEMA_INPUT = "poc-solver-input/1"
SCHEMA_RESULT = "poc-solve-result/1"
S_PROBLEM_SCHEMA = "s-problem/1"


class G10Error(RuntimeError):
    def __init__(self, code, message, source_id="", expected=None, actual=None, phase="g10"):
        self.code = code
        self.phase = phase
        self.source_id = source_id
        self.expected = expected
        self.actual = actual
        super().__init__("%s: %s (source=%r expected=%r actual=%r)"
                         % (code, message, source_id, expected, actual))


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _sha(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _obligations(expected):
    if expected is None:
        return []
    value = _get(expected, "all", None)
    if value is None:
        value = expected
    if isinstance(value, dict):
        value = value.get("all", value.get("obligations", []))
    return list(value or [])


def _oid(obligation):
    return str(_get(obligation, "oid", _get(obligation, "obligation_id", "")))


def _kind(obligation):
    return str(_get(obligation, "kind", ""))


def _discharged(obligation):
    return _get(obligation, "discharged_by", "unknown") is not None


@dataclass(frozen=True)
class PocInput:
    schema: str
    s_problem_schema: str
    problem_digest: str
    required_obligation_ids: tuple
    required_count: int
    categories: dict
    supported_required_unknown: int
    critical_unknown: int
    expected_digest: str
    status: str
    code: Optional[str] = None
    detail: str = ""

    def as_dict(self):
        return dict(self.__dict__)


@dataclass(frozen=True)
class PocSolveResult:
    schema: str
    s_problem_schema: str
    s_problem_digest: str
    required_count: int
    supported_required_unknown: int
    critical_unknown: int
    state: str
    r_usable: bool
    reason: str
    code: Optional[str]
    search_scope: dict
    assignment: dict
    fallback_candidate: object
    expected_ids: tuple
    result_digest: str = ""

    def as_dict(self):
        return dict(self.__dict__)


def consume_poc_expected(expected, s_problem, *, authoritative=None,
                         critical_unknown=0) -> PocInput:
    """Validate the complete expected set against L5 s_problem/1, no BYTES filtering."""
    if s_problem is None or str(_get(s_problem, "schema", "")) != S_PROBLEM_SCHEMA:
        return PocInput(SCHEMA_INPUT, str(_get(s_problem, "schema", "")), "", (), 0, {},
                        0, 0, "", "NOT_READY", "POC_SOLVER_INPUT_NOT_READY",
                        "s_problem schema is not s-problem/1")
    contract = _get(s_problem, "s_contract", {}) or {}
    if contract.get("unknown_not_a_hard_constraint") is not True:
        return PocInput(SCHEMA_INPUT, S_PROBLEM_SCHEMA, str(_get(s_problem, "problem_digest", "")),
                        (), 0, {}, 0, 0, "", "NOT_READY", "POC_SOLVER_INPUT_NOT_READY",
                        "s_problem s_contract is missing")
    items = _obligations(expected)
    if not items:
        return PocInput(SCHEMA_INPUT, S_PROBLEM_SCHEMA, str(_get(s_problem, "problem_digest", "")),
                        (), 0, {}, 0, 0, "", "NOT_READY", "POC_SOLVER_INPUT_NOT_READY",
                        "expected obligation set is empty")
    ids = tuple(_oid(item) for item in items)
    if authoritative is not None:
        authoritative_ids = {_oid(item) for item in _obligations(authoritative)}
        missing = sorted(authoritative_ids - set(ids))
        extra = sorted(set(ids) - authoritative_ids)
        if missing or extra:
            return PocInput(SCHEMA_INPUT, S_PROBLEM_SCHEMA,
                            str(_get(s_problem, "problem_digest", "")), ids, len(items),
                            {}, 0, 0, _sha(ids), "NOT_READY", "POC_SOLVER_INPUT_NOT_READY",
                            "expected set differs from authoritative: missing=%r extra=%r"
                            % (missing[:3], extra[:3]))
    categories = {}
    for item in items:
        categories[_kind(item)] = categories.get(_kind(item), 0) + 1
    unknown = sum(1 for item in items if not _discharged(item))
    if int(critical_unknown) > 0 or unknown > 0:
        code = "POC_SOLVER_INPUT_NOT_READY"
        detail = "supported_required_unknown=%d critical_unknown=%d" % (unknown, int(critical_unknown))
        status = "NOT_READY"
    else:
        code = None
        detail = "complete supported set"
        status = "READY"
    return PocInput(SCHEMA_INPUT, S_PROBLEM_SCHEMA, str(_get(s_problem, "problem_digest", "")),
                    ids, len(items), categories, unknown, int(critical_unknown),
                    _sha(ids), status, code, detail)


def classify_poc_result(expected, s_problem, raw_core=None, *, authoritative=None,
                        critical_unknown=0, search_domain=None) -> PocSolveResult:
    """Four states: SAT only for a READY, fully discharged, core-sat input."""
    poc = consume_poc_expected(expected, s_problem, authoritative=authoritative,
                               critical_unknown=critical_unknown)
    raw_status = str(_get(raw_core, "status", "missing") if raw_core else "missing")
    search_scope = {"solver": "POC_SUPPORTED_DOMAIN", "search_complete": False,
                    "domain": dict(search_domain or {})}
    base = dict(schema=SCHEMA_RESULT, s_problem_schema=S_PROBLEM_SCHEMA,
                s_problem_digest=poc.problem_digest, required_count=poc.required_count,
                supported_required_unknown=poc.supported_required_unknown,
                critical_unknown=poc.critical_unknown, expected_ids=poc.required_obligation_ids)
    if poc.status != "READY":
        result = PocSolveResult(**base, state="UNKNOWN", r_usable=False,
                                reason=poc.detail or "POC solver input not ready",
                                code="POC_SOLVER_INPUT_NOT_READY", search_scope=search_scope,
                                assignment={}, fallback_candidate=None)
        return PocSolveResult(**{**result.__dict__, "result_digest": _sha(result.__dict__)})
    if raw_core is None or raw_status == "missing":
        result = PocSolveResult(**base, state="UNKNOWN", r_usable=False,
                                reason="no raw core result supplied", code="POC_SOLVER_INPUT_NOT_READY",
                                search_scope=search_scope, assignment={}, fallback_candidate=None)
        return PocSolveResult(**{**result.__dict__, "result_digest": _sha(result.__dict__)})
    if raw_status == "sat":
        result = PocSolveResult(**base, state="SAT", r_usable=True,
                                reason="complete supported-domain obligations; core SAT",
                                code=None, search_scope={**search_scope, "search_complete": True},
                                assignment=dict(_get(raw_core, "assignment", {}) or {}),
                                fallback_candidate=None)
    elif raw_status == "fallback":
        result = PocSolveResult(**base, state="FALLBACK", r_usable=False,
                                reason="core fallback is not SAT and is not R-usable",
                                code="POC_SOLVER_FALLBACK", search_scope=search_scope,
                                assignment={}, fallback_candidate=_get(raw_core, "fallback_candidate", raw_core))
    elif raw_status == "unsat":
        complete = bool(_get(raw_core, "search_scope_complete", False)) and bool(search_domain)
        state = "UNSAT" if complete else "UNKNOWN"
        code = None if complete else "POC_SOLVER_INPUT_NOT_READY"
        result = PocSolveResult(**base, state=state, r_usable=False,
                                reason=("finite domain exhausted" if complete else
                                        "UNSAT claimed without a complete search domain"),
                                code=code, search_scope={**search_scope, "search_complete": complete,
                                                        "domain": dict(search_domain or {})},
                                assignment={}, fallback_candidate=None)
    else:
        result = PocSolveResult(**base, state="UNKNOWN", r_usable=False,
                                reason="core status=%s" % raw_status,
                                code="POC_SOLVER_INPUT_NOT_READY", search_scope=search_scope,
                                assignment={}, fallback_candidate=None)
    return PocSolveResult(**{**result.__dict__, "result_digest": _sha(result.__dict__)})


def solve_poc_gate(expected, s_problem, raw_core=None, **kwargs) -> PocSolveResult:
    return classify_poc_result(expected, s_problem, raw_core, **kwargs)


__all__ = ["G10Error", "PocInput", "PocSolveResult", "consume_poc_expected",
           "classify_poc_result", "solve_poc_gate"]
