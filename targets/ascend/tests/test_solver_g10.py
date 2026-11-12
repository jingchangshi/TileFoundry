"""G10 tests: complete POC ExpectedObligationSet -> four solver states.

Uses task-82's real expected set (47 obligations) and L5 s-problem/1 document.
Run: PYTHONPATH=src:targets/ascend python targets/ascend/tests/test_solver_g10.py
CPU only; no device.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "targets" / "ascend"))
sys.path.insert(0, str(REPO / "targets" / "ascend" / "tests"))

from tilefoundry_ascend.solver import g10
from tilefoundry_ascend.schedule.poc_gate import build_expected_obligations
from tilefoundry_ascend.schedule.ledger import build_s_problem_document, s_hard_constraint_dependencies
import test_task82_g0_g9 as T82


def built():
    _hm, _hf, _tf, ledger, problem, expansion = T82._built()
    return build_expected_obligations(ledger, problem, expansion), build_s_problem_document(problem)


def test_g10_positive_uses_complete_47_obligation_set_and_sat():
    expected, s_problem = built()
    result = g10.solve_poc_gate(expected, s_problem,
                                {"status": "sat", "assignment": {"tasks": {"t0": {"start_time": 0}}}})
    assert result.state == "SAT" and result.r_usable is True
    assert result.required_count == 47 and result.supported_required_unknown == 0
    assert result.critical_unknown == 0
    assert result.s_problem_schema == "s-problem/1" and result.s_problem_digest
    assert result.assignment["tasks"]["t0"]["start_time"] == 0
    # the expected set is categorically complete, not the BYTES-only hard subset
    assert result.required_count == 47 and len(s_problem["problem"]["dependencies"]) < 47


def test_g10_missing_l_obligation_is_not_ready():
    expected, s_problem = built()
    shortened = {"all": expected["all"][:-1]}
    result = g10.solve_poc_gate(shortened, s_problem, {"status": "sat"}, authoritative=expected)
    assert result.state == "UNKNOWN" and result.r_usable is False
    assert result.code == "POC_SOLVER_INPUT_NOT_READY"


def test_g10_critical_unknown_is_not_ready():
    expected, s_problem = built()
    mutated = list(expected["all"])
    mutated[0] = dataclasses.replace(mutated[0], discharged_by=None)
    mutated_set = {"all": mutated}
    result = g10.solve_poc_gate(mutated_set, s_problem, {"status": "sat"}, authoritative=expected)
    assert result.state == "UNKNOWN" and result.r_usable is False
    assert result.code == "POC_SOLVER_INPUT_NOT_READY"
    assert result.supported_required_unknown == 1


def test_g10_fallback_is_not_sat_and_unscoped_unsat_is_unknown():
    expected, s_problem = built()
    fallback = g10.solve_poc_gate(expected, s_problem, {"status": "fallback",
                                                      "fallback_candidate": {"minimal_II": 1}})
    assert fallback.state == "FALLBACK" and fallback.r_usable is False
    assert fallback.fallback_candidate is not None
    unscoped = g10.solve_poc_gate(expected, s_problem, {"status": "unsat"})
    assert unscoped.state == "UNKNOWN" and unscoped.code == "POC_SOLVER_INPUT_NOT_READY"
    scoped = g10.solve_poc_gate(expected, s_problem, {"status": "unsat",
                                                     "search_scope_complete": True},
                                search_domain={"num_stages": 4, "versions": [4, 2]})
    assert scoped.state == "UNSAT"
    assert scoped.search_scope["domain"] == {"num_stages": 4, "versions": [4, 2]}
    assert scoped.search_scope["search_complete"] is True


def test_g10_hard_bytes_subset_alone_is_not_enough():
    expected, s_problem = built()
    # Passing only the BYTES-filtered family (12 raw) must not be accepted as 47.
    subset = {"all": expected["raw"]}
    result = g10.solve_poc_gate(subset, s_problem, {"status": "sat"}, authoritative=expected)
    assert result.state == "UNKNOWN" and result.code == "POC_SOLVER_INPUT_NOT_READY"


if __name__ == "__main__":
    failed = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print("PASS", _name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL", _name, type(exc).__name__, exc)
    raise SystemExit(1 if failed else 0)
