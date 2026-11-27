"""Tests for L3/L4 AgentGates execution, receipt verification, classification, and scoping."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from tests.ascend.benchmarks import test_npu_acceptance as npu_acceptance
from tests.ascend.benchmarks.l3_contract import (
    EvidenceRecorder,
    NodeContract,
    TargetContract,
    comparison_failure,
    coverage_failure,
    infrastructure_failure,
)
from tilefoundry_ascend.agent.npu import ag_transport, l3_l4_gates

_L3_GATE = ag_transport.AGENT_GATES_L3_PROMOTION_GATE
_L4_GATE = ag_transport.AGENT_GATES_L4_CANNBENCH_GATE
_L4_PREP_GATE = ag_transport.AGENT_GATES_L4_CANNBENCH_PREP_GATE


class _JumpClock:
    def __init__(self, step: float = 1000.0) -> None:
        self._step = step
        self._now = 0.0

    def __call__(self) -> float:
        cur = self._now
        self._now += self._step
        return cur


def _completed(cmd: list[str], payload: dict[str, object], rc: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(cmd, rc, stdout=json.dumps(payload), stderr="")


def _ag_config(root: Path) -> Path:
    ev_root = root / "v2-evidence"
    cfg = root / "agent-gates.toml"
    cfg.write_text(
        f'evidence_root = "{ev_root}"\n\n'
        "[workspaces.tilefoundry]\n"
        f'source_root = "{root.resolve()}"\n'
        'adapters = ["tilefoundry-npu"]\n',
        encoding="utf-8",
    )
    return cfg


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified(root: Path, *, satisfied: bool = True, failures: list[str] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "satisfied": satisfied,
        "status": "PASS" if satisfied else "FAIL",
        "current_source_identity": {"root": str(root.resolve())},
    }
    if failures:
        payload["failures"] = failures
    return payload


def _l3_candidate_failure(
    *, node: str = "test_case_20", case_id: str = "20", role: str = "candidate"
) -> dict[str, object]:
    return {
        "kind": "candidate" if role == "candidate" else "infrastructure",
        "category": "numeric" if role == "candidate" else "baseline_mismatch",
        "stage": "comparison",
        "node": node,
        "case": case_id,
        "role": role,
        "comparator": "torch.testing.assert_close",
        "references": ["oracle"],
        "comparisons": [{"output": 0, "passed": role != "candidate", "message": "max_abs_error exceeded"}],
        "exceptionType": None,
        "detail": "candidate numeric comparison failed",
    }


def _l3_infra_failure(
    *,
    kind: str = "infrastructure",
    category: str = "exception",
    stage: str = "run",
    node: str | None = "test_case_20",
    case_id: str | None = "20",
    detail: str = "device runtime error",
) -> dict[str, object]:
    return {
        "kind": kind,
        "category": category,
        "stage": stage,
        "node": node,
        "case": case_id,
        "role": None,
        "comparator": None,
        "references": [],
        "comparisons": [],
        "exceptionType": "RuntimeError" if category == "exception" else None,
        "detail": detail,
    }


def _l3_node(
    *, node: str = "test_case_20", case_id: str = "20", status: str = "failed", failure: Mapping[str, object] | None = None
) -> dict[str, object]:
    return {
        "name": node,
        "nodeId": f"{node}::0",
        "status": status,
        "requiredCases": [case_id],
        "cases": {case_id: status},
        "details": {},
        "failure": dict(failure) if failure is not None else None,
    }


def _l3_coverage(kind: str) -> dict[str, object]:
    if kind == "qualified_complete":
        return {
            "qualified": True,
            "nodes": {"expected": 2, "completed": 2, "missing": [], "failed": [], "skipped": [], "blocked": []},
            "cases": {"expected": 20, "completed": 20, "missing": [], "unexpected": []},
            "referenceCacheProblem": None,
        }
    if kind == "complete":
        return {
            "qualified": False,
            "nodes": {"expected": 2, "completed": 2, "missing": [], "failed": [], "skipped": [], "blocked": []},
            "cases": {"expected": 20, "completed": 20, "missing": [], "unexpected": []},
            "referenceCacheProblem": None,
        }
    if kind == "missing_node":
        return {
            "qualified": False,
            "nodes": {"expected": 2, "completed": 0, "missing": ["test_case_20"], "failed": [], "skipped": [], "blocked": []},
            "cases": {"expected": 20, "completed": 0, "missing": [], "unexpected": []},
            "referenceCacheProblem": None,
        }
    skipped = kind == "skipped"
    return {
        "qualified": False,
        "nodes": {
            "expected": 2,
            "completed": 0 if skipped else 1,
            "missing": [],
            "failed": [] if skipped else ["test_case_20"],
            "skipped": ["test_case_20"] if skipped else [],
            "blocked": [],
        },
        "cases": {"expected": 20, "completed": 0 if skipped else 19, "missing": ["test_case_20:20"], "unexpected": []},
        "referenceCacheProblem": None,
    }


def _l3_report(
    *,
    candidate: Path,
    status: str = "PASS",
    coverage: dict[str, object] | None = None,
    failure_category: str = "",
    failure: Mapping[str, object] | None = None,
    nodes: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    cov = coverage or _l3_coverage("qualified_complete")
    digest = _digest(candidate)
    summary: dict[str, object] = {
        "schema": "tilefoundry-l3-evidence/v1",
        "profile": "l3-promotion",
        "task": "level4/attention_residual",
        "operator": "attention_residual",
        "target": {"operator": "attention_residual", "task": "level4/attention_residual", "modulePath": str(candidate)},
        "candidate": {"modulePath": str(candidate), "relativePath": candidate.name, "sha256": digest},
        "interpreter": {"sysExecutable": "/adapter/python3", "expected": "/adapter/python3", "match": True},
        "coverage": cov,
        "referenceCache": {},
        "writeErrors": [],
        "interruption": {"interrupted": False, "errorCategory": None, "signal": None},
        "phase": "finished",
        "nodes": list(nodes or []),
    }
    if failure is not None:
        summary["failure"] = dict(failure)
        summary["failures"] = [dict(failure)]
    obs: dict[str, object] = {"profile": "l3-promotion", "coverage": cov, "evidence": summary}
    if failure_category:
        obs["failureCategory"] = failure_category
    return {
        "status": status,
        "runId": "run-l3",
        "receiptPath": "/receipt-l3.json",
        "recordedStatus": status,
        "gates": [{"gateId": _L3_GATE, "status": status, "recordedStatus": status, "observations": obs}],
    }


def _l4_report(
    *,
    candidate: str,
    flags: Mapping[str, object] | None = None,
    status: str = "PASS",
    failure: Mapping[str, object] | None = None,
    gate_failures: tuple[str, ...] | None = None,
    prep_status: str = "PASS",
    prep_failures: tuple[str, ...] = (),
    prep_observations: Mapping[str, object] | None = None,
) -> dict[str, object]:
    obs: dict[str, object] = {
        "avgSpeedup": 4.25,
        "score": 99.5,
        "officialReportValid": True,
        "allCasesPassed": True,
        "allTimingsValid": True,
        "caseCount": 1,
        "passedCases": 1,
        "timedCases": 1,
        "results": [
            {
                "task": "level4/attention_residual",
                "candidate": candidate,
                "reportDir": "/signed-report",
                "reportValid": True,
                "cases": [{"case_id": "level4/attention_residual_2", "speedup": 3.5, "elapsed_us": 12.5}],
            }
        ],
    }
    obs.update(flags or {})
    prep_gate: dict[str, object] = {
        "gateId": _L4_PREP_GATE,
        "status": prep_status,
        "recordedStatus": prep_status,
        "observations": dict(prep_observations or {}),
    }
    if prep_failures:
        prep_gate["failures"] = list(prep_failures)
    gate: dict[str, object] = {"gateId": _L4_GATE, "status": status, "recordedStatus": status, "observations": obs}
    if gate_failures is not None:
        gate["failures"] = list(gate_failures)
    elif failure is not None:
        gate["failures"] = [dict(failure)]
    return {"status": status, "runId": "run-l4", "receiptPath": "/receipt-l4.json", "recordedStatus": status, "gates": [prep_gate, gate]}


def _v2_handler(
    *,
    report: Mapping[str, object],
    root: Path,
    verification: Mapping[str, object] | None = None,
    submit_payload: Mapping[str, object] | None = None,
    submit_rc: int = 0,
    wait_payloads: list[dict[str, object]] | None = None,
    doctor: Mapping[str, object] | None = None,
    doctor_rc: int = 0,
):
    calls: list[list[str]] = []
    waits = list(
        wait_payloads
        or [{"execution_id": "exec-1", "timed_out": False, "status": {"state": "FINISHED", "run_id": str(report.get("runId"))}}]
    )
    verify = verification if verification is not None else _verified(root)
    doc = dict(doctor or {"status": "PASS"})

    def _fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        cmd_list = [str(p) for p in cmd]
        if cmd_list[0] == "git":
            return subprocess.CompletedProcess(cmd_list, 1, "", "")
        if cmd_list[0] == "npu-smi":
            return subprocess.CompletedProcess(cmd_list, 1, "", "")
        calls.append(cmd_list)
        sub = cmd_list[1]
        if sub == "doctor":
            return _completed(cmd_list, doc, rc=doctor_rc)
        if sub == "submit":
            return _completed(cmd_list, dict(submit_payload or {"execution_id": "exec-1"}), rc=submit_rc)
        if sub == "wait":
            return _completed(cmd_list, waits.pop(0) if len(waits) > 1 else waits[0])
        if sub == "report":
            return _completed(cmd_list, dict(report))
        if sub == "verify-requirements":
            return _completed(cmd_list, dict(verify))
        raise AssertionError(cmd_list)

    return _fake_run, calls


def _setup_env(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(_ag_config(root)))
    ag_bin = root / "agent-gates"
    ag_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("TILEFOUNDRY_NPU_AGENT_GATES_BIN", str(ag_bin))


def test_l3_promotion_execution_resume_and_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute L3 promotion, resume without resubmit, and keep execution_id on timeout."""
    _setup_env(monkeypatch, tmp_path)
    cand = tmp_path / "kernels.py"
    cand.write_text("x = 1\n", encoding="utf-8")
    report = _l3_report(candidate=cand)
    handler, calls = _v2_handler(
        report=report,
        root=tmp_path,
        submit_payload={"execution_id": "exec-l3", "device_id": "5"},
        wait_payloads=[
            {"execution_id": "exec-l3", "timed_out": True, "status": {"state": "RUNNING"}},
            {"execution_id": "exec-l3", "timed_out": False, "status": {"state": "FINISHED", "run_id": "run-l3"}},
        ],
    )
    monkeypatch.setattr(subprocess, "run", handler)
    started: list[tuple[str, str]] = []
    res = l3_l4_gates.run_l3_full_contract_via_agent_gates(
        operator_name="attention_residual",
        candidate_module_path=cand,
        source_root=tmp_path,
        device_id="3",
        on_execution_started=lambda eid, did: started.append((eid, did)),
    )
    assert res.passed is True
    assert res.run_id == "run-l3"
    assert res.execution_id == "exec-l3"
    assert res.device_id == "5"
    assert res.candidate_sha256 == _digest(cand)
    assert started == [("exec-l3", "5")]
    assert [c[1] for c in calls] == ["doctor", "submit", "wait", "wait", "report", "verify-requirements"]
    assert calls[1][calls[1].index("--gate") + 1] == _L3_GATE
    assert calls[1][calls[1].index("--candidate") + 1] == "kernels.py"

    h2, calls2 = _v2_handler(report=report, root=tmp_path)
    monkeypatch.setattr(subprocess, "run", h2)
    res_exec = l3_l4_gates.run_l3_full_contract_via_agent_gates(
        operator_name="attention_residual",
        candidate_module_path=cand,
        source_root=tmp_path,
        resume_execution_id="exec-saved",
    )
    assert res_exec.passed is True
    assert [c[1] for c in calls2] == ["wait", "report", "verify-requirements"]

    h3, calls3 = _v2_handler(report=report, root=tmp_path)
    monkeypatch.setattr(subprocess, "run", h3)
    res_run = l3_l4_gates.run_l3_full_contract_via_agent_gates(
        operator_name="attention_residual",
        candidate_module_path=cand,
        source_root=tmp_path,
        resume_execution_id="exec-saved",
        resume_run_id="run-l3",
    )
    assert res_run.passed is True
    assert [c[1] for c in calls3] == ["report", "verify-requirements"]

    seen_to: list[str] = []

    def _timeout_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[0] == "git":
            return subprocess.CompletedProcess(cmd, 1, "", "")
        seen_to.append(cmd[1])
        return _completed(cmd, {"status": "PASS"} if cmd[1] == "doctor" else {"execution_id": "exec-to"})

    monkeypatch.setattr(subprocess, "run", _timeout_run)
    monkeypatch.setattr(time, "monotonic", _JumpClock())
    res_to = l3_l4_gates.run_l3_full_contract_via_agent_gates(
        operator_name="attention_residual",
        candidate_module_path=cand,
        source_root=tmp_path,
        timeout_sec=1,
    )
    assert res_to.passed is False
    assert res_to.execution_id == "exec-to"
    assert res_to.execution_terminal is False
    assert "timed out" in res_to.error_message
    assert seen_to == ["doctor", "submit"]


@pytest.mark.parametrize(
    ("cov_kind", "fail_cat", "intr_err", "marker_kind", "expected_kind"),
    [
        ("failed", "", None, "none", l3_l4_gates.FAILURE_KIND_INFRASTRUCTURE),
        ("failed", "numeric", None, "none", l3_l4_gates.FAILURE_KIND_INFRASTRUCTURE),
        ("skipped", "coverage", None, "none", l3_l4_gates.FAILURE_KIND_INFRASTRUCTURE),
        ("missing_node", "", None, "none", l3_l4_gates.FAILURE_KIND_INFRASTRUCTURE),
        ("failed", "numeric", "reference_cache_error", "candidate", l3_l4_gates.FAILURE_KIND_INFRASTRUCTURE),
        ("failed", "numeric", None, "candidate", l3_l4_gates.FAILURE_KIND_CANDIDATE),
        ("failed", "numeric", None, "candidate_no_nodes", l3_l4_gates.FAILURE_KIND_INFRASTRUCTURE),
        ("failed", "numeric", None, "candidate_baseline_role", l3_l4_gates.FAILURE_KIND_INFRASTRUCTURE),
        ("failed", "numeric", None, "candidate_all_passed", l3_l4_gates.FAILURE_KIND_INFRASTRUCTURE),
        ("failed", "numeric", None, "candidate_case_mismatch", l3_l4_gates.FAILURE_KIND_INFRASTRUCTURE),
        ("failed", "numeric", None, "candidate_with_infra_row", l3_l4_gates.FAILURE_KIND_INFRASTRUCTURE),
        ("failed", "candidate", None, "kernel_compile", l3_l4_gates.FAILURE_KIND_CANDIDATE),
    ],
)
def test_l3_failure_classification_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cov_kind: str,
    fail_cat: str,
    intr_err: str | None,
    marker_kind: str,
    expected_kind: str,
) -> None:
    """Classify L3 failures as candidate only on unambiguous producer numeric/kernel evidence."""
    _setup_env(monkeypatch, tmp_path)
    cand = tmp_path / "kernels.py"
    cand.write_text("x = 1\n", encoding="utf-8")

    marker: dict[str, object] | None = None
    nodes: list[dict[str, object]] = []
    if marker_kind == "candidate":
        marker = _l3_candidate_failure()
        nodes = [_l3_node(failure=marker)]
    elif marker_kind == "candidate_no_nodes":
        marker = _l3_candidate_failure()
    elif marker_kind == "candidate_baseline_role":
        marker = _l3_candidate_failure(role="baseline")
        nodes = [_l3_node(failure=marker)]
    elif marker_kind == "candidate_all_passed":
        marker = _l3_candidate_failure()
        marker["comparisons"] = [{"output": 0, "passed": True, "message": ""}]
        nodes = [_l3_node(failure=marker)]
    elif marker_kind == "candidate_case_mismatch":
        marker = _l3_candidate_failure()
        nodes = [_l3_node(case_id="21", failure=marker)]
    elif marker_kind == "candidate_with_infra_row":
        marker = _l3_candidate_failure()
        nodes = [_l3_node(failure=marker)]
    elif marker_kind == "kernel_compile":
        marker = {
            "kind": "candidate",
            "category": "candidate",
            "stage": "compile",
            "node": "test_case_20",
            "case": "20",
            "role": "candidate",
            "exceptionType": "RuntimeError",
            "detail": "exceeded vf stack size",
        }
        nodes = [_l3_node(failure=marker)]

    report = _l3_report(
        candidate=cand,
        status="FAIL",
        coverage=_l3_coverage(cov_kind),
        failure_category=fail_cat,
        failure=marker,
        nodes=nodes,
    )
    ev = report["gates"][0]["observations"]["evidence"]
    if marker_kind == "candidate_with_infra_row":
        ev["failures"] = [dict(marker or {}), _l3_infra_failure()]
        ev["failure"] = _l3_infra_failure()
    ev["interruption"]["errorCategory"] = intr_err
    handler, _calls = _v2_handler(
        report=report, root=tmp_path, verification=_verified(tmp_path, satisfied=False, failures=["gate failed"])
    )
    monkeypatch.setattr(subprocess, "run", handler)
    res = l3_l4_gates.run_l3_full_contract_via_agent_gates(
        operator_name="attention_residual", candidate_module_path=cand, source_root=tmp_path
    )
    assert res.passed is False
    assert res.failure_kind == expected_kind
    assert res.pending is (expected_kind != l3_l4_gates.FAILURE_KIND_CANDIDATE)


def test_l3_counterexample_extraction_collateral_missing_cases_and_identity_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Extract failed_case_nums, keep collateral missing cases, and reject tampered receipts."""
    rec = EvidenceRecorder(
        tmp_path / "l3_ev.json",
        TargetContract(
            operator="mla",
            task="level4/mla",
            task_dir=tmp_path,
            nodes=(
                NodeContract(name="test_op_cases", required_cases=("0", "3", "4")),
                NodeContract(name="test_other", required_cases=("0",)),
            ),
            case_ids=("0", "3", "4"),
        ),
    )
    rec.case_finished("test_op_cases", "0", "passed")
    cf = comparison_failure(
        node="test_op_cases",
        case_id="3",
        role="candidate",
        comparator="torch.testing.assert_close",
        references=["oracle"],
        comparisons=[{"output": 0, "passed": False, "message": "max_abs_error"}],
        detail="case 3 mismatch",
    )
    rec.case_finished("test_op_cases", "3", "failed", failure=cf)
    rec.node_finished("test_op_cases", "failed")
    rec.case_finished("test_other", "0", "passed")
    rec.node_finished("test_other", "passed")
    rec.finalize()
    ev = dict(rec.payload())
    ev["timing"] = {"finishedAt": "2026-11-25T00:00:05Z"}
    ev["failures"] = [cf, coverage_failure(node="test_op_cases", case_id="4", category="missing_case", detail="skipped")]
    mapped = l3_l4_gates.l3_failure_result_from_observation(
        {"failureCategory": "numeric", "evidence": ev},
        run_id="run-x",
        execution_id="exec-x",
        device_id="2",
    )
    assert mapped.failure_kind == l3_l4_gates.FAILURE_KIND_CANDIDATE
    assert mapped.pending is False
    assert mapped.failed_case_nums == (3, 4)
    assert mapped.failure_stage == "comparison"

    cross_ev = dict(ev)
    cross_ev["failures"] = [cf, coverage_failure(node="test_other", case_id="0", category="missing_case", detail="gap")]
    assert l3_l4_gates.classify_l3_observation_failure({"failureCategory": "numeric", "evidence": cross_ev}) == (
        l3_l4_gates.FAILURE_KIND_INFRASTRUCTURE,
        True,
    )
    infra_ev = dict(ev)
    infra_ev["failures"] = [cf, infrastructure_failure(node="test_op_cases", case_id="3", stage="run", category="exception", detail="offline")]
    assert l3_l4_gates.classify_l3_observation_failure({"failureCategory": "numeric", "evidence": infra_ev}) == (
        l3_l4_gates.FAILURE_KIND_INFRASTRUCTURE,
        True,
    )

    _setup_env(monkeypatch, tmp_path)
    cand = tmp_path / "kernels.py"
    cand.write_text("x = 1\n", encoding="utf-8")
    other = tmp_path / "other.py"
    other.write_text("x = 2\n", encoding="utf-8")

    bad_path_rep = _l3_report(candidate=other)
    bad_path_rep["gates"][0]["observations"]["evidence"]["target"]["modulePath"] = str(cand)
    h, _ = _v2_handler(report=bad_path_rep, root=tmp_path)
    monkeypatch.setattr(subprocess, "run", h)
    r_path = l3_l4_gates.run_l3_full_contract_via_agent_gates(
        operator_name="attention_residual", candidate_module_path=cand, source_root=tmp_path
    )
    assert r_path.passed is False and "candidate path" in r_path.error_message

    bad_dig_rep = _l3_report(candidate=other)
    bad_dig_rep["gates"][0]["observations"]["evidence"]["target"]["modulePath"] = str(cand)
    bad_dig_rep["gates"][0]["observations"]["evidence"]["candidate"]["modulePath"] = str(cand)
    h2, _ = _v2_handler(report=bad_dig_rep, root=tmp_path)
    monkeypatch.setattr(subprocess, "run", h2)
    r_dig = l3_l4_gates.run_l3_full_contract_via_agent_gates(
        operator_name="attention_residual", candidate_module_path=cand, source_root=tmp_path
    )
    assert r_dig.passed is False and "digest" in r_dig.error_message

    unverified_h, _ = _v2_handler(
        report=_l3_report(candidate=cand), root=tmp_path, verification=_verified(tmp_path, satisfied=False, failures=["stale"])
    )
    monkeypatch.setattr(subprocess, "run", unverified_h)
    r_unv = l3_l4_gates.run_l3_full_contract_via_agent_gates(
        operator_name="attention_residual", candidate_module_path=cand, source_root=tmp_path
    )
    assert r_unv.passed is False and "did not verify" in r_unv.error_message

    inc_cov_rep = _l3_report(
        candidate=cand,
        coverage={
            "qualified": True,
            "nodes": {"expected": 2, "completed": 2, "missing": [], "failed": [], "skipped": [], "blocked": []},
            "cases": {"expected": 20, "completed": 19, "missing": [], "unexpected": []},
            "referenceCacheProblem": None,
        },
    )
    h_inc, _ = _v2_handler(report=inc_cov_rep, root=tmp_path)
    monkeypatch.setattr(subprocess, "run", h_inc)
    r_inc = l3_l4_gates.run_l3_full_contract_via_agent_gates(
        operator_name="attention_residual", candidate_module_path=cand, source_root=tmp_path
    )
    assert r_inc.passed is False and "case coverage incomplete" in r_inc.error_message


def test_l3_occupancy_backoff_and_card_rotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rotate pool cards on L3 occupancy and backoff across passes before succeeding."""
    _setup_env(monkeypatch, tmp_path)
    cand = tmp_path / "kernels.py"
    cand.write_text("x = 1\n", encoding="utf-8")
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(float(s)))

    attempts: list[str] = []
    cur = {"dev": ""}

    def _run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[0] in ("git", "npu-smi"):
            return subprocess.CompletedProcess(cmd, 1, "", "")
        sub = cmd[1]
        if sub == "doctor":
            return _completed(cmd, {"status": "PASS"})
        if sub == "submit":
            dev = cmd[cmd.index("--device-id") + 1]
            cur["dev"] = dev
            attempts.append(dev)
            return _completed(cmd, {"execution_id": f"exec-{len(attempts)}", "device_id": dev})
        if sub == "wait":
            return _completed(cmd, {"status": {"state": "FINISHED", "run_id": f"run-{len(attempts)}"}})
        if sub == "report":
            if len(attempts) <= 2:
                return _completed(
                    cmd,
                    {
                        "status": "BLOCKED_ENV",
                        "receiptPath": "/r.json",
                        "gates": [{"gateId": _L3_GATE, "status": "BLOCKED_ENV", "observations": {"errorCategory": "occupancy"}}],
                    },
                )
            return _completed(cmd, _l3_report(candidate=cand))
        return _completed(cmd, _verified(tmp_path))

    monkeypatch.setattr(subprocess, "run", _run)
    res = l3_l4_gates.run_l3_full_contract_via_agent_gates(
        operator_name="attention_residual",
        candidate_module_path=cand,
        source_root=tmp_path,
        device_id="0",
        device_pool=("0", "1"),
    )
    assert res.passed is True
    assert attempts == ["0", "1", "0"]
    assert sleeps == [15.0]


def test_l4_official_cannbench_receipts_occupancy_and_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validate L4 official metrics, doctor blocks, occupancy, and device refusal."""
    assert l3_l4_gates.DEFAULT_L4_CANNBENCH_WAIT_SEC >= 4500
    _setup_env(monkeypatch, tmp_path)
    cand = tmp_path / "kernels.py"
    cand.write_text("x = 1\n", encoding="utf-8")

    h_ok, calls_ok = _v2_handler(report=_l4_report(candidate="kernels.py"), root=tmp_path)
    monkeypatch.setattr(subprocess, "run", h_ok)
    res_ok = l3_l4_gates.run_l4_official_cannbench_via_agent_gates(
        operator_name="attention_residual", candidate_module_path=cand, source_root=tmp_path, device_id="7"
    )
    assert res_ok.passed is True
    assert res_ok.avg_speedup == pytest.approx(4.25)
    assert res_ok.score == pytest.approx(99.5)
    assert res_ok.per_case_speedups == {2: 3.5}
    assert res_ok.per_case_elapsed_us == {2: 12.5}
    assert [c[1] for c in calls_ok] == ["doctor", "submit", "wait", "report", "verify-requirements"]

    h_doc, calls_doc = _v2_handler(
        report=_l4_report(candidate="kernels.py"),
        root=tmp_path,
        doctor={"status": "BLOCKED_ENV", "failures": ["CONFIG_CHECK_FAILED: device"]},
        doctor_rc=1,
    )
    monkeypatch.setattr(subprocess, "run", h_doc)
    res_doc = l3_l4_gates.run_l4_official_cannbench_via_agent_gates(
        operator_name="attention_residual", candidate_module_path=cand, source_root=tmp_path
    )
    assert res_doc.passed is False
    assert res_doc.failure_kind == l3_l4_gates.FAILURE_KIND_INFRASTRUCTURE
    assert [c[1] for c in calls_doc] == ["doctor"]

    h_rej, calls_rej = _v2_handler(
        report=_l4_report(candidate="kernels.py"),
        root=tmp_path,
        submit_payload={"code": "DEVICE_NOT_AUTHORIZED", "message": "excluded"},
        submit_rc=2,
    )
    monkeypatch.setattr(subprocess, "run", h_rej)
    res_rej = l3_l4_gates.run_l4_official_cannbench_via_agent_gates(
        operator_name="attention_residual", candidate_module_path=cand, source_root=tmp_path, device_id="7"
    )
    assert res_rej.passed is False
    assert res_rej.device_rejected is True
    assert [c[1] for c in calls_rej] == ["doctor", "submit"]

    prep_occ_rep = _l4_report(
        candidate="kernels.py",
        status="NOT_RUN",
        gate_failures=("DEPENDENCY_NOT_PASSED",),
        prep_status="BLOCKED_ENV",
        prep_failures=("BLOCKED_EXIT_CODE: 75",),
        prep_observations={"errorCategory": "occupancy"},
    )
    h_prep, _ = _v2_handler(report=prep_occ_rep, root=tmp_path)
    monkeypatch.setattr(subprocess, "run", h_prep)
    res_prep = l3_l4_gates.run_l4_official_cannbench_via_agent_gates(
        operator_name="attention_residual", candidate_module_path=cand, source_root=tmp_path
    )
    assert res_prep.passed is False
    assert res_prep.blocked_env_category == "occupancy"

    cand_fail_rep = _l4_report(
        candidate="kernels.py",
        status="FAIL",
        failure={"message": "candidate numeric mismatch"},
        flags={"errorCategory": "occupancy"},
    )
    h_cf, _ = _v2_handler(report=cand_fail_rep, root=tmp_path)
    monkeypatch.setattr(subprocess, "run", h_cf)
    res_cf = l3_l4_gates.run_l4_official_cannbench_via_agent_gates(
        operator_name="attention_residual", candidate_module_path=cand, source_root=tmp_path
    )
    assert res_cf.passed is False
    assert res_cf.failure_kind == l3_l4_gates.FAILURE_KIND_CANDIDATE
    assert res_cf.blocked_env_category == ""


@pytest.mark.parametrize(
    "flags",
    [
        {"officialReportValid": False},
        {"allCasesPassed": False},
        {"allTimingsValid": False},
        {"passedCases": 0},
        {"timedCases": 0},
        {"caseCount": 0},
    ],
)
def test_l4_rejects_incomplete_official_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: dict[str, object]
) -> None:
    """Reject missing official report flags, zero cases, or mismatched candidate paths."""
    _setup_env(monkeypatch, tmp_path)
    cand = tmp_path / "kernels.py"
    cand.write_text("x = 1\n", encoding="utf-8")
    h, _ = _v2_handler(report=_l4_report(candidate="kernels.py", flags=flags), root=tmp_path)
    monkeypatch.setattr(subprocess, "run", h)
    res = l3_l4_gates.run_l4_official_cannbench_via_agent_gates(
        operator_name="attention_residual", candidate_module_path=cand, source_root=tmp_path
    )
    assert res.passed is False
    assert res.per_case_speedups == {}


def test_l3_acceptance_scoping_and_tuned_only_guard() -> None:
    """Scope L3 acceptance tests by operator name and keep tuned-only execution guard."""
    assert npu_acceptance._normalize_target_op_short("") == ""
    assert npu_acceptance._normalize_target_op_short("level4/mla") == "mla"
    assert npu_acceptance._normalize_target_op_short("level2/softmax") == "softmax"

    for kind, expected in (
        ("attention_residual", True),
        ("mla", True),
        ("dynamic_cache", True),
        ("generic_cannbench", False),
    ):
        assert npu_acceptance._should_run_acceptance_test(kind, "") is expected

    assert npu_acceptance._should_run_acceptance_test("attention_residual", "level4/attention_residual") is True
    assert npu_acceptance._should_run_acceptance_test("mla", "level4/attention_residual") is False
    assert npu_acceptance._should_run_acceptance_test("mla", "level4/mla") is True
    assert npu_acceptance._should_run_acceptance_test("generic_cannbench", "level2/softmax") is True
    assert npu_acceptance._should_run_acceptance_test("mla", "level2/softmax") is False
