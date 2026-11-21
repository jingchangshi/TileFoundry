"""CPU-only contract tests for campaign AgentGates L2/L3/L4 routes and receipts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from tilefoundry_ascend.agent.campaign import surrogate_cannbench
from tilefoundry_ascend.agent.campaign.anchor_specs import AnchorCaseSpec

_FAST_GATE = "tilefoundry-npu-device-fast"
_L3_GATE = "tilefoundry-npu-l3-promotion"
_L4_GATE = "tilefoundry-npu-cannbench"


def _completed(
    cmd: list[str], parsed: dict[str, object], returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=cmd,
        returncode=returncode,
        stdout=json.dumps(parsed),
        stderr="",
    )


def _env_file(tmp_path: Path) -> Path:
    path = tmp_path / "env.sh"
    path.write_text(
        "TILEFOUNDRY_NPU_CANNBENCH_REPO=/configured/cannbench\n"
        "TILEFOUNDRY_NPU_WORK_ROOT=/configured/work\n",
        encoding="utf-8",
    )
    return path


def _patch_common(monkeypatch, tmp_path: Path) -> Path:
    env_file = _env_file(tmp_path)
    cache = tmp_path / "cache"
    tl_cache = tmp_path / "tl-cache"
    monkeypatch.setattr(surrogate_cannbench, "_resolve_env_file_path", lambda _p: env_file)
    monkeypatch.setattr(surrogate_cannbench, "_resolve_agent_gates_bin", lambda _r: "agent-gates")
    monkeypatch.setattr(surrogate_cannbench, "resolve_cannbench_repo", lambda **_kw: tmp_path)
    monkeypatch.setattr(surrogate_cannbench, "_default_surrogate_cache_dir", lambda: cache)
    monkeypatch.setattr(surrogate_cannbench, "_default_surrogate_tl_cache_dir", lambda: tl_cache)
    monkeypatch.setattr(
        surrogate_cannbench, "prewarm_anchor_reference_cache", lambda *_a, **_kw: None
    )
    monkeypatch.setattr(
        surrogate_cannbench,
        "prewarm_surrogate_tilelang_cache",
        lambda *_a, **_kw: tl_cache,
    )
    monkeypatch.setattr(surrogate_cannbench, "_prewarm_micro_probe_cache", lambda *_a, **_kw: None)
    return env_file


def _anchor(case_num: int = 2) -> AnchorCaseSpec:
    return AnchorCaseSpec("level4/attention_residual", case_num, "small_decode", 10.0, 3.0)


def _agent_gates_config(tmp_path: Path) -> Path:
    evidence_root = tmp_path / "v2-evidence"
    config = tmp_path / "agent-gates.toml"
    config.write_text(
        f'evidence_root = "{evidence_root}"\n'
        "\n"
        "[workspaces.tilefoundry]\n"
        f'source_root = "{tmp_path.resolve()}"\n'
        'adapters = ["tilefoundry-npu"]\n',
        encoding="utf-8",
    )
    return config


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified(
    root: Path,
    *,
    satisfied: bool = True,
    failures: list[str] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "satisfied": satisfied,
        "status": "PASS" if satisfied else "FAIL",
        "current_source_identity": {"root": str(root.resolve())},
    }
    if failures:
        payload["failures"] = failures
    return payload


def _l3_candidate_failure(
    *,
    node: str = "test_case_20",
    case_id: str = "20",
    role: str = "candidate",
) -> dict[str, object]:
    """Producer marker for an explicit numeric comparison failure."""
    return {
        "kind": "candidate" if role == "candidate" else "infrastructure",
        "category": "numeric" if role == "candidate" else "baseline_mismatch",
        "stage": "comparison",
        "node": node,
        "case": case_id,
        "role": role,
        "comparator": "torch.testing.assert_close",
        "references": ["oracle"],
        "comparisons": [
            {"output": 0, "passed": role != "candidate", "message": "max_abs_error exceeded"},
        ],
        "exceptionType": None,
        "detail": "candidate numeric comparison failed",
    }


def _l3_infrastructure_failure(
    *,
    kind: str = "infrastructure",
    category: str = "exception",
    stage: str = "run",
    node: str | None = "test_case_20",
    case_id: str | None = "20",
    role: str | None = None,
    comparator: str | None = None,
    detail: str = "device runtime error",
) -> dict[str, object]:
    """Producer marker for a runtime, baseline, or coverage fault."""
    return {
        "kind": kind,
        "category": category,
        "stage": stage,
        "node": node,
        "case": case_id,
        "role": role,
        "comparator": comparator,
        "references": [],
        "comparisons": [],
        "exceptionType": "RuntimeError" if category == "exception" else None,
        "detail": detail,
    }


def _l3_node(
    *,
    node: str = "test_case_20",
    case_id: str = "20",
    status: str = "failed",
    failure: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Signed node/case record matching a producer failure marker."""
    return {
        "name": node,
        "nodeId": f"{node}::0",
        "status": status,
        "requiredCases": [case_id],
        "cases": {case_id: status},
        "details": {},
        "failure": dict(failure) if failure is not None else None,
        "error": None,
        "startedAt": None,
        "endedAt": None,
    }


def _l3_coverage(kind: str) -> dict[str, object]:
    """Signed coverage counters for one L3 classification scenario."""
    if kind == "qualified_complete":
        return {
            "qualified": True,
            "nodes": {
                "expected": 2,
                "completed": 2,
                "missing": [],
                "failed": [],
                "skipped": [],
                "blocked": [],
            },
            "cases": {"expected": 20, "completed": 20, "missing": [], "unexpected": []},
            "referenceCacheProblem": None,
        }
    if kind == "complete":
        return {
            "qualified": False,
            "nodes": {
                "expected": 2,
                "completed": 2,
                "missing": [],
                "failed": [],
                "skipped": [],
                "blocked": [],
            },
            "cases": {"expected": 20, "completed": 20, "missing": [], "unexpected": []},
            "referenceCacheProblem": None,
        }
    if kind == "missing_node":
        return {
            "qualified": False,
            "nodes": {
                "expected": 2,
                "completed": 0,
                "missing": ["test_case_20"],
                "failed": [],
                "skipped": [],
                "blocked": [],
            },
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
        "cases": {
            "expected": 20,
            "completed": 0 if skipped else 19,
            "missing": ["test_case_20:20"],
            "unexpected": [],
        },
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
    coverage_payload: dict[str, object] = coverage or {
        "qualified": True,
        "nodes": {
            "expected": 2,
            "completed": 2,
            "missing": [],
            "failed": [],
            "skipped": [],
            "blocked": [],
        },
        "cases": {
            "expected": 20,
            "completed": 20,
            "missing": [],
            "unexpected": [],
        },
        "referenceCacheProblem": None,
    }
    candidate_digest = _digest(candidate)
    summary: dict[str, object] = {
        "schema": "tilefoundry-l3-evidence/v1",
        "profile": "l3-promotion",
        "task": "level4/attention_residual",
        "operator": "attention_residual",
        "target": {
            "operator": "attention_residual",
            "task": "level4/attention_residual",
            "taskDir": None,
            "modulePath": str(candidate),
            "cases": [],
        },
        "targetCaseCount": 0,
        "candidate": {
            "modulePath": str(candidate),
            "relativePath": candidate.name,
            "declaredPath": str(candidate),
            "sha256": candidate_digest,
        },
        "expectedIdentity": {
            "target": {"modulePath": str(candidate)},
            "candidate": {
                "modulePath": str(candidate),
                "relativePath": candidate.name,
                "sha256": candidate_digest,
            },
        },
        "interpreter": {
            "sysExecutable": "/adapter/python3",
            "expected": "/adapter/python3",
            "match": True,
        },
        "coverage": coverage_payload,
        "referenceCache": {},
        "writeErrors": [],
        "interruption": {
            "interrupted": False,
            "errorCategory": None,
            "signal": None,
        },
        "phase": "finished",
        "nodes": list(nodes or []),
    }
    if failure is not None:
        summary["failure"] = dict(failure)
        summary["failures"] = [dict(failure)]
    observations: dict[str, object] = {
        "profile": "l3-promotion",
        "task": "level4/attention_residual",
        "operator": "attention_residual",
        "candidate": {
            "path": candidate.name,
            "absolutePath": str(candidate),
            "sha256": candidate_digest,
            "sizeBytes": candidate.stat().st_size,
        },
        "coverage": coverage_payload,
        "evidence": summary,
    }
    if failure_category:
        observations["failureCategory"] = failure_category
    return {
        "status": status,
        "runId": "run-l3",
        "receiptPath": "/receipt-l3.json",
        "recordedStatus": status,
        "gates": [
            {
                "gateId": _L3_GATE,
                "status": status,
                "recordedStatus": status,
                "observations": observations,
            }
        ],
    }


def _l4_report(
    *, candidate: str, flags: Mapping[str, object] | None = None
) -> dict[str, object]:
    observations: dict[str, object] = {
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
                "cases": [
                    {
                        "case_id": "level4/attention_residual_2",
                        "speedup": 3.5,
                        "elapsed_us": 12.5,
                    }
                ],
            }
        ],
    }
    observations.update(flags or {})
    return {
        "status": "PASS",
        "runId": "run-l4",
        "receiptPath": "/receipt-l4.json",
        "recordedStatus": "PASS",
        "gates": [
            {
                "gateId": _L4_GATE,
                "status": "PASS",
                "recordedStatus": "PASS",
                "observations": observations,
            }
        ],
    }


def _v2_handler(
    *,
    report: Mapping[str, object],
    root: Path,
    verification: Mapping[str, object] | None = None,
    submit_payload: Mapping[str, object] | None = None,
    wait_payloads: list[dict[str, object]] | None = None,
    doctor: Mapping[str, object] | None = None,
    doctor_returncode: int = 0,
):
    calls: list[list[str]] = []
    waits = list(
        wait_payloads
        or [
            {
                "execution_id": "exec-1",
                "timed_out": False,
                "status": {"state": "FINISHED", "run_id": str(report.get("runId"))},
            }
        ]
    )
    verify = verification if verification is not None else _verified(root)
    doctor_payload = dict(doctor or {"status": "PASS"})

    def _fake_run(cmd, *, cwd, env, timeout, **kwargs):
        _ = (cwd, timeout, kwargs)
        cmd_list = [str(part) for part in cmd]
        calls.append(cmd_list)
        command = cmd_list[1]
        if command == "doctor":
            return _completed(cmd_list, doctor_payload, returncode=doctor_returncode)
        if command == "submit":
            return _completed(
                cmd_list, dict(submit_payload or {"execution_id": "exec-1"})
            )
        if command == "wait":
            payload = waits.pop(0) if len(waits) > 1 else waits[0]
            return _completed(cmd_list, payload)
        if command == "report":
            return _completed(cmd_list, dict(report))
        if command == "verify-requirements":
            return _completed(cmd_list, dict(verify))
        raise AssertionError(cmd_list)

    return _fake_run, calls


def _fast_response(
    request: Mapping[str, object],
    *,
    request_sha256: str,
    tamper: str = "",
) -> dict[str, object]:
    batches = request.get("batches") or []
    targets = request.get("msprof_targets") or []
    assert isinstance(batches, list)
    assert isinstance(targets, list)
    operators: dict[str, dict[str, object]] = {}
    bindings: dict[str, dict[str, dict[str, str]]] = {}
    for batch in batches:
        assert isinstance(batch, Mapping)
        op_name = str(batch["operator_name"])
        candidates = batch["candidates"]
        assert isinstance(candidates, list)
        operators[op_name] = {}
        for candidate in candidates:
            assert isinstance(candidate, Mapping)
            candidate_id = str(candidate["candidate_id"])
            operators[op_name][candidate_id] = {
                "candidate_id": candidate_id,
                "operator_name": op_name,
                "all_passed": True,
                "avg_speedup": 2.0,
                "min_case_speedup": 1.5,
                "avg_roofline_eta": 0.5,
                "surrogate_score": 0.75,
                "anchors": [],
                "error_message": "",
            }
            bindings.setdefault(op_name, {})[candidate_id] = {
                "module_path": str(candidate["module_path"]),
                "sha256": str(candidate["sha256"]),
                "post_sha256": str(candidate["sha256"]),
            }
    telemetry: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    for row in targets:
        assert isinstance(row, Mapping)
        op_name = str(row["operator_name"])
        candidate_id = str(row["candidate_id"])
        bindings.setdefault(op_name, {})[candidate_id] = {
            "module_path": str(row["module_path"]),
            "sha256": str(row["sha256"]),
            "post_sha256": str(row["sha256"]),
        }
        telemetry.setdefault(op_name, {}).setdefault(candidate_id, {})[
            str(row["case_num"])
        ] = {"aicore_time_us": 12.5, "mte2_ratio": 0.25}
    unique_candidates = len(
        {(op, cid) for op, cands in bindings.items() for cid in cands}
    )
    coverage = {
        "expected_candidates": unique_candidates,
        "evaluated_candidates": unique_candidates,
        "expected_msprof_targets": len(targets),
        "observed_msprof_targets": len(targets),
    }
    status = "ok"
    error = ""
    if tamper == "sha":
        for op_bindings in bindings.values():
            for binding in op_bindings.values():
                binding["sha256"] = "0" * 64
    elif tamper == "post_sha":
        for op_bindings in bindings.values():
            for binding in op_bindings.values():
                binding["post_sha256"] = "0" * 64
    elif tamper == "coverage":
        coverage["observed_msprof_targets"] = 0
    elif tamper == "request_error":
        status = "request_error"
        error = "staged candidate sha256 mismatch"
    elif tamper == "infrastructure":
        status = "infrastructure_error"
        error = "device lost"
    return {
        "schema": surrogate_cannbench.FAST_RESPONSE_SCHEMA,
        "request_id": request["request_id"],
        "request_sha256": request_sha256,
        "mode": request["mode"],
        "status": status,
        "error": error,
        "operators": operators,
        "msprof_telemetry": telemetry,
        "candidates": bindings,
        "coverage": coverage,
    }


def _fast_gate_run(
    root: Path,
    monkeypatch,
    *,
    tamper: str = "",
    satisfied: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(_agent_gates_config(root)))
    monkeypatch.setattr(surrogate_cannbench, "_resolve_agent_gates_bin", lambda _r: "agent-gates")
    monkeypatch.setattr(
        surrogate_cannbench,
        "_agent_gates_evidence_root",
        lambda: root / "v2-evidence",
    )
    monkeypatch.setattr(
        surrogate_cannbench,
        "verify_agent_gates_receipt",
        lambda *_a, **_kw: (True, ""),
    )
    calls: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    env_texts: list[str] = []

    def _fake_run(cmd, *, cwd, env, timeout, **kwargs):
        _ = (cwd, timeout, kwargs)
        cmd_list = [str(part) for part in cmd]
        calls.append({"cmd": cmd_list, "env": dict(env)})
        command = cmd_list[1]
        if command == "doctor":
            return _completed(cmd_list, {"status": "PASS"})
        if command == "submit":
            candidate_relative = cmd_list[cmd_list.index("--candidate") + 1]
            request_path = root / candidate_relative
            request_bytes = request_path.read_bytes()
            request = json.loads(request_bytes)
            assert isinstance(request, dict)
            requests.append(request)
            response_path = root / str(request["response_path"])
            response_path.write_text(
                json.dumps(
                    _fast_response(
                        request,
                        request_sha256=hashlib.sha256(request_bytes).hexdigest(),
                        tamper=tamper,
                    )
                ),
                encoding="utf-8",
            )
            return _completed(cmd_list, {"execution_id": "exec-fast"})
        if command == "wait":
            return _completed(
                cmd_list,
                {
                    "timed_out": False,
                    "status": {"state": "FINISHED", "run_id": "run-fast"},
                },
            )
        if command == "report":
            return _completed(
                cmd_list,
                {
                    "status": "PASS",
                    "runId": "run-fast",
                    "receiptPath": "/receipt-fast.json",
                    "gates": [
                        {
                            "gateId": _FAST_GATE,
                            "observations": {"deviceWindow": {"heldSeconds": 2.5}},
                        }
                    ],
                },
            )
        if command == "verify-requirements":
            return _completed(cmd_list, _verified(root, satisfied=satisfied))
        if command == "run":
            env_file = Path(cmd_list[cmd_list.index("--env-file") + 1])
            env_texts.append(env_file.read_text(encoding="utf-8"))
            request_path = Path(str(env["TF_MICRO_PROBE_REQUEST"]))
            request = json.loads(request_path.read_text(encoding="utf-8"))
            Path(str(request["response_path"])).write_text(
                '{"metrics": {"speedup": 1.25}}', encoding="utf-8"
            )
            return _completed(
                cmd_list,
                {
                    "status": "PASS",
                    "runId": "run-probe",
                    "receiptPath": "/receipt-probe.json",
                },
            )
        raise AssertionError(cmd_list)

    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", _fake_run)
    return calls, requests, env_texts


class _JumpClock:
    """Monotonic clock that jumps forward so a bounded wait expires immediately."""

    def __init__(self, step: float = 1000.0) -> None:
        self._step = step
        self._now = 0.0

    def __call__(self) -> float:
        current = self._now
        self._now += self._step
        return current


def test_l2_fast_contract_is_workspace_relative_and_sha_bound(
    tmp_path: Path, monkeypatch
) -> None:
    """Batch and profile requests stay immutable, workspace-relative, and content-bound."""
    env_file = _patch_common(monkeypatch, tmp_path)
    reference_prewarms: list[dict[str, Any]] = []
    tilelang_prewarms: list[dict[str, Any]] = []

    def _record_reference(*_args, **kwargs):
        reference_prewarms.append(dict(kwargs))

    def _record_tilelang(*_args, **kwargs):
        tilelang_prewarms.append(dict(kwargs))
        return Path(str(kwargs["tl_cache_dir"]))

    monkeypatch.setattr(
        surrogate_cannbench, "prewarm_anchor_reference_cache", _record_reference
    )
    monkeypatch.setattr(
        surrogate_cannbench, "prewarm_surrogate_tilelang_cache", _record_tilelang
    )
    calls, requests, env_texts = _fast_gate_run(tmp_path, monkeypatch)
    candidate = tmp_path / "candidate.py"
    candidate.write_text("x = 1\n", encoding="utf-8")
    anchors = [_anchor(), _anchor(case_num=3)]
    mla_anchor = AnchorCaseSpec("level4/mla", 4, "small_decode", 10.0, 3.0)
    by_op, run_id, receipt_path, held = (
        surrogate_cannbench.run_multi_operator_surrogate_batch_via_agent_gates(
            [
                ("attention_residual", [("cand", candidate)], anchors),
                ("mla", [("cand", candidate)], [mla_anchor]),
            ],
            source_root=tmp_path,
            env_file=env_file,
            prewarm_cache=False,
        )
    )
    assert run_id == "run-fast"
    assert receipt_path == "/receipt-fast.json"
    assert held == 2.5
    assert by_op["attention_residual"]["cand"].all_passed is True
    assert by_op["mla"]["cand"].all_passed is True

    telemetry = surrogate_cannbench.collect_bottleneck_msprof_via_agent_gates(
        "attention_residual",
        candidate,
        anchors,
        source_root=tmp_path,
        env_file=env_file,
    )
    assert telemetry == {"small_decode": {"aicore_time_us": 12.5, "mte2_ratio": 0.25}}

    probe = SimpleNamespace(
        probe_id="probe",
        probe_kind="vector",
        factory_name="entry",
        baseline_probe_source="x = 1\n",
        optimized_probe_source="x = 2\n",
        tensor_shape=(16, 128),
        tensor_dtype="float32",
    )
    metrics = surrogate_cannbench.run_micro_probe_via_agent_gates(
        probe,
        source_root=tmp_path,
        env_file=env_file,
        prewarm_cache=False,
    )
    assert metrics["speedup"] == 1.25

    batch_commands = calls[:5]
    profile_commands = calls[5:10]
    probe_commands = calls[10:]
    assert [call["cmd"][1] for call in batch_commands] == [
        "doctor",
        "submit",
        "wait",
        "report",
        "verify-requirements",
    ]
    for group in (batch_commands, profile_commands):
        for call in group:
            assert "--env-file" not in call["cmd"]
            assert surrogate_cannbench.FAST_REQUEST_ENV not in call["env"]
            assert "TF_SURROGATE_BATCH_REQUEST" not in call["env"]
            assert "TILEFOUNDRY_NPU_MSPROF_LOCK" not in call["env"]
        for index in (0, 1, 4):
            cmd = group[index]["cmd"]
            assert cmd[cmd.index("--gate") + 1] == _FAST_GATE
    assert len(probe_commands) == 1
    probe_cmd = probe_commands[0]["cmd"]
    assert probe_cmd[1] == "run"
    assert probe_cmd[probe_cmd.index("--gate") + 1] == _FAST_GATE
    assert "TF_MICRO_PROBE_REQUEST" in probe_commands[0]["env"]
    assert len(env_texts) == 1
    assert "TILEFOUNDRY_NPU_DEVICE_TESTS" in env_texts[0]
    assert "TILEFOUNDRY_NPU_CPU_TESTS" not in env_texts[0]

    batch_request, profile_request = requests
    request_root = tmp_path / "build" / "agent_gates" / "fast"
    batch_id = str(batch_request["request_id"])
    batch_submit = batch_commands[1]["cmd"]
    assert batch_submit[batch_submit.index("--target") + 1] == f"fast/{batch_id}"
    assert (
        batch_submit[batch_submit.index("--candidate") + 1]
        == f"build/agent_gates/fast/{batch_id}.request.json"
    )
    assert batch_request["schema"] == surrogate_cannbench.FAST_REQUEST_SCHEMA
    assert batch_request["mode"] == "batch"
    for machine_field in (
        "source_root",
        "cannbench_repo",
        "tilelang_cache_dir",
        "reference_cache_dir",
    ):
        assert machine_field not in batch_request
    assert (
        batch_request["response_path"]
        == f"build/agent_gates/fast/{batch_id}.response.json"
    )
    assert not Path(str(batch_request["response_path"])).is_absolute()
    assert batch_request["msprof_targets"] == []
    assert {
        Path(str(call["cache_dir"])) for call in reference_prewarms
    } == {Path("/configured/work/surrogate/reference").resolve()}
    assert {
        Path(str(call["tl_cache_dir"])) for call in tilelang_prewarms
    } == {Path("/configured/work/surrogate/tilelang").resolve()}
    batch_entries = {
        str(entry["operator_name"]): entry for entry in batch_request["batches"]
    }
    assert set(batch_entries) == {"attention_residual", "mla"}
    candidate_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
    for operator_name, entry in batch_entries.items():
        assert set(entry) == {"operator_name", "entrypoint", "candidates", "anchors"}
        for anchor in entry["anchors"]:
            assert set(anchor) == {
                "task",
                "case_num",
                "regime_tag",
                "ref_perf_us",
                "t_hw_us",
            }
        staged_entry = entry["candidates"][0]
        assert staged_entry["candidate_id"] == "cand"
        assert not Path(str(staged_entry["module_path"])).is_absolute()
        assert str(staged_entry["module_path"]).startswith(
            f"build/agent_gates/fast/{batch_id}/candidates/{operator_name}/"
        )
        staged_path = tmp_path / str(staged_entry["module_path"])
        assert staged_path.is_file()
        assert staged_path.stat().st_mode & 0o777 == 0o444
        assert staged_entry["sha256"] == candidate_sha256
        assert hashlib.sha256(staged_path.read_bytes()).hexdigest() == candidate_sha256
    assert (request_root / f"{batch_id}.request.json").stat().st_mode & 0o777 == 0o444

    profile_id = str(profile_request["request_id"])
    profile_submit = profile_commands[1]["cmd"]
    assert profile_submit[profile_submit.index("--target") + 1] == f"fast/{profile_id}"
    assert profile_request["mode"] == "profile"
    assert profile_request["batches"] == []
    for machine_field in (
        "source_root",
        "cannbench_repo",
        "tilelang_cache_dir",
        "reference_cache_dir",
    ):
        assert machine_field not in profile_request
    profile_targets = profile_request["msprof_targets"]
    assert [row["case_num"] for row in profile_targets] == [2, 3]
    for row in profile_targets:
        assert set(row) == {
            "operator_name",
            "candidate_id",
            "module_path",
            "sha256",
            "case_num",
            "warmup",
            "repeat",
        }
        assert not Path(str(row["module_path"])).is_absolute()
        assert row["sha256"] == hashlib.sha256(
            (tmp_path / str(row["module_path"])).read_bytes()
        ).hexdigest()

    env_file.write_text(
        "TILEFOUNDRY_NPU_CANNBENCH_REPO=/configured/cannbench\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="TILEFOUNDRY_NPU_WORK_ROOT"):
        surrogate_cannbench.run_multi_operator_surrogate_batch_via_agent_gates(
            [("attention_residual", [("cand", candidate)], anchors)],
            source_root=tmp_path,
            env_file=env_file,
            prewarm_cache=False,
        )
    env_file.write_text(
        "TILEFOUNDRY_NPU_CANNBENCH_REPO=/configured/cannbench\n"
        f"TILEFOUNDRY_NPU_WORK_ROOT={tmp_path / 'work'}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outside the source root"):
        surrogate_cannbench.run_multi_operator_surrogate_batch_via_agent_gates(
            [("attention_residual", [("cand", candidate)], anchors)],
            source_root=tmp_path,
            env_file=env_file,
            prewarm_cache=False,
        )


def test_l2_fast_contract_rejects_unverified_or_tampered_responses(
    tmp_path: Path, monkeypatch
) -> None:
    """L2 routes fail closed on unverified receipts, tampered bindings, and coverage gaps."""
    env_file = _patch_common(monkeypatch, tmp_path)
    candidate = tmp_path / "candidate.py"
    candidate.write_text("x = 1\n", encoding="utf-8")
    anchors = [_anchor()]

    _fast_gate_run(tmp_path, monkeypatch, satisfied=False)
    with pytest.raises(RuntimeError, match="did not verify"):
        surrogate_cannbench.run_multi_operator_surrogate_batch_via_agent_gates(
            [("attention_residual", [("cand", candidate)], anchors)],
            source_root=tmp_path,
            env_file=env_file,
            prewarm_cache=False,
        )
    with pytest.raises(RuntimeError, match="did not verify"):
        surrogate_cannbench.collect_bottleneck_msprof_via_agent_gates(
            "attention_residual",
            candidate,
            anchors,
            source_root=tmp_path,
            env_file=env_file,
        )

    _fast_gate_run(tmp_path, monkeypatch, tamper="sha")
    with pytest.raises(RuntimeError, match="does not match the staged candidate"):
        surrogate_cannbench.run_multi_operator_surrogate_batch_via_agent_gates(
            [("attention_residual", [("cand", candidate)], anchors)],
            source_root=tmp_path,
            env_file=env_file,
            prewarm_cache=False,
        )

    _fast_gate_run(tmp_path, monkeypatch, tamper="coverage")
    with pytest.raises(RuntimeError, match="msprof coverage"):
        surrogate_cannbench.collect_bottleneck_msprof_via_agent_gates(
            "attention_residual",
            candidate,
            anchors,
            source_root=tmp_path,
            env_file=env_file,
        )

    _fast_gate_run(tmp_path, monkeypatch, tamper="infrastructure")
    with pytest.raises(RuntimeError, match="infrastructure_error"):
        surrogate_cannbench.run_multi_operator_surrogate_batch_via_agent_gates(
            [("attention_residual", [("cand", candidate)], anchors)],
            source_root=tmp_path,
            env_file=env_file,
            prewarm_cache=False,
        )


def test_l2_fast_contract_classifies_post_execution_mutation_as_infrastructure(
    tmp_path: Path, monkeypatch
) -> None:
    """Post-timing mutation is an infrastructure error, not a request or candidate fault."""
    env_file = _patch_common(monkeypatch, tmp_path)
    candidate = tmp_path / "candidate.py"
    candidate.write_text("x = 1\n", encoding="utf-8")
    anchors = [_anchor()]

    _fast_gate_run(tmp_path, monkeypatch, tamper="request_error")
    with pytest.raises(RuntimeError, match="fast gate request_error"):
        surrogate_cannbench.run_multi_operator_surrogate_batch_via_agent_gates(
            [("attention_residual", [("cand", candidate)], anchors)],
            source_root=tmp_path,
            env_file=env_file,
            prewarm_cache=False,
        )

    _fast_gate_run(tmp_path, monkeypatch, tamper="post_sha")
    with pytest.raises(
        RuntimeError,
        match=r"infrastructure_error: candidate attention_residual/\S+ changed during evaluation",
    ):
        surrogate_cannbench.collect_bottleneck_msprof_via_agent_gates(
            "attention_residual",
            candidate,
            anchors,
            source_root=tmp_path,
            env_file=env_file,
        )


def test_l3_promotion_uses_v2_target_and_candidate_without_injection(
    tmp_path: Path, monkeypatch
) -> None:
    """L3 submits the promotion gate with canonical target and relative candidate only."""
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(_agent_gates_config(tmp_path)))
    monkeypatch.setattr(surrogate_cannbench, "_resolve_agent_gates_bin", lambda _r: "agent-gates")
    candidate = tmp_path / "kernels.py"
    candidate.write_text("x = 1\n", encoding="utf-8")
    report = _l3_report(candidate=candidate)
    handler, calls = _v2_handler(
        report=report,
        root=tmp_path,
        submit_payload={"execution_id": "exec-l3"},
        wait_payloads=[
            {
                "execution_id": "exec-l3",
                "timed_out": True,
                "status": {"state": "RUNNING"},
            },
            {
                "execution_id": "exec-l3",
                "timed_out": False,
                "status": {"state": "FINISHED", "run_id": "run-l3"},
            },
        ],
    )
    original = handler

    def _fake_run(cmd, *, cwd, env, timeout, **kwargs):
        submit = [str(part) for part in cmd][1] == "submit"
        if submit:
            cmd_list = [str(part) for part in cmd]
            assert "--target" in cmd_list
            assert cmd_list[cmd_list.index("--target") + 1] == "level4/attention_residual"
            assert cmd_list[cmd_list.index("--candidate") + 1] == "kernels.py"
            assert cmd_list[cmd_list.index("--gate") + 1] == _L3_GATE
            assert "--env-file" not in cmd_list
        assert "PYTHONPATH" not in env
        return original(cmd, cwd=cwd, env=env, timeout=timeout, **kwargs)

    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        surrogate_cannbench,
        "_agent_gates_evidence_root",
        lambda: tmp_path / "v2-evidence",
    )
    result = surrogate_cannbench.run_l3_full_contract_via_agent_gates(
        operator_name="attention_residual",
        candidate_module_path=candidate,
        source_root=tmp_path,
        env_file=tmp_path / "unused-env.sh",
        device_id="7",
    )
    assert result.passed is True
    assert result.run_id == "run-l3"
    assert result.execution_id == "exec-l3"
    assert result.candidate_sha256 == _digest(candidate)
    assert result.target_path == str(candidate.resolve())
    assert result.failure_kind == ""
    doctor = calls[0]
    assert doctor[1] == "doctor"
    assert doctor[doctor.index("--gate") + 1] == _L3_GATE
    assert doctor[doctor.index("--target") + 1] == "level4/attention_residual"
    assert doctor[doctor.index("--candidate") + 1] == "kernels.py"
    wait_cmd = [cmd for cmd in calls if cmd[1] == "wait"]
    assert len(wait_cmd) == 2
    assert {cmd[cmd.index("--execution-id") + 1] for cmd in wait_cmd} == {"exec-l3"}
    verify_cmd = [cmd for cmd in calls if cmd[1] == "verify-requirements"][0]
    assert verify_cmd[verify_cmd.index("--gate") + 1] == _L3_GATE
    assert verify_cmd[verify_cmd.index("--target") + 1] == "level4/attention_residual"
    assert verify_cmd[verify_cmd.index("--candidate") + 1] == "kernels.py"


def test_l3_promotion_accepts_qualified_and_rejects_numeric_or_coverage(
    tmp_path: Path, monkeypatch
) -> None:
    """One adapter summary accepts a qualified run and distinguishes numeric from coverage."""
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(_agent_gates_config(tmp_path)))
    monkeypatch.setattr(surrogate_cannbench, "_resolve_agent_gates_bin", lambda _r: "agent-gates")
    monkeypatch.setattr(
        surrogate_cannbench,
        "_agent_gates_evidence_root",
        lambda: tmp_path / "v2-evidence",
    )
    candidate = tmp_path / "kernels.py"
    candidate.write_text("x = 1\n", encoding="utf-8")
    qualified = _l3_report(candidate=candidate)
    numeric_failure = _l3_candidate_failure()
    numeric = _l3_report(
        candidate=candidate,
        status="FAIL",
        failure_category="numeric",
        failure=numeric_failure,
        nodes=[_l3_node(failure=numeric_failure)],
        coverage={
            "qualified": False,
            "nodes": {
                "expected": 2,
                "completed": 1,
                "missing": [],
                "failed": ["test_case_20"],
                "skipped": [],
                "blocked": [],
            },
            "cases": {
                "expected": 20,
                "completed": 19,
                "missing": ["test_case_20:20"],
                "unexpected": [],
            },
            "referenceCacheProblem": None,
        },
    )
    coverage = _l3_report(
        candidate=candidate,
        status="FAIL",
        failure_category="coverage",
        coverage={
            "qualified": False,
            "nodes": {
                "expected": 2,
                "completed": 0,
                "missing": [],
                "failed": [],
                "skipped": ["test_case_20"],
                "blocked": [],
            },
            "cases": {
                "expected": 20,
                "completed": 0,
                "missing": ["test_case_20:20"],
                "unexpected": [],
            },
            "referenceCacheProblem": None,
        },
    )

    outcomes: list[tuple[bool, str, str, bool]] = []
    for report in (qualified, numeric, coverage):
        handler, _calls = _v2_handler(report=report, root=tmp_path)
        monkeypatch.setattr(surrogate_cannbench.subprocess, "run", handler)
        result = surrogate_cannbench.run_l3_full_contract_via_agent_gates(
            operator_name="attention_residual",
            candidate_module_path=candidate,
            source_root=tmp_path,
        )
        outcomes.append(
            (result.passed, result.failure_kind, result.failure_category, result.pending)
        )

    assert outcomes == [
        (True, "", "", False),
        (False, surrogate_cannbench.FAILURE_KIND_CANDIDATE, "numeric", False),
        (False, surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE, "coverage", True),
    ]


@pytest.mark.parametrize(
    (
        "coverage_kind",
        "failure_category",
        "interruption_error",
        "marker_kind",
        "expected_kind",
    ),
    [
        ("failed", "", None, "none", surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE),
        ("failed", "numeric", None, "none", surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE),
        ("skipped", "coverage", None, "none", surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE),
        ("skipped", "", None, "none", surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE),
        (
            "missing_node",
            "",
            None,
            "none",
            surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE,
        ),
        (
            "skipped",
            "",
            "reference_cache_unavailable",
            "none",
            surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE,
        ),
        (
            "qualified_complete",
            "reference_cache_expectation_cold",
            None,
            "none",
            surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE,
        ),
        ("complete", "numeric", None, "none", surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE),
        (
            "failed",
            "numeric",
            "reference_cache_error",
            "candidate",
            surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE,
        ),
        ("failed", "numeric", None, "candidate", surrogate_cannbench.FAILURE_KIND_CANDIDATE),
        (
            "failed",
            "numeric",
            None,
            "candidate_no_nodes",
            surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE,
        ),
        ("failed", "", None, "candidate", surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE),
        (
            "failed",
            "numeric",
            None,
            "candidate_baseline_role",
            surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE,
        ),
        (
            "failed",
            "numeric",
            None,
            "candidate_all_passed",
            surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE,
        ),
        (
            "failed",
            "numeric",
            None,
            "candidate_case_mismatch",
            surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE,
        ),
        (
            "failed",
            "numeric",
            None,
            "candidate_with_infra_row",
            surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE,
        ),
        (
            "failed",
            "baseline_mismatch",
            None,
            "baseline_mismatch",
            surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE,
        ),
        (
            "failed",
            "coverage",
            None,
            "coverage_case_identity",
            surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE,
        ),
        (
            "failed",
            "exception",
            None,
            "runtime_exception",
            surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE,
        ),
    ],
)
def test_l3_promotion_classifies_coverage_failure(
    tmp_path: Path,
    monkeypatch,
    coverage_kind,
    failure_category,
    interruption_error,
    marker_kind,
    expected_kind,
) -> None:
    """Only explicit candidate evidence rolls back; other failures preserve pending."""
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(_agent_gates_config(tmp_path)))
    monkeypatch.setattr(surrogate_cannbench, "_resolve_agent_gates_bin", lambda _r: "agent-gates")
    candidate = tmp_path / "kernels.py"
    candidate.write_text("x = 1\n", encoding="utf-8")

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
        marker["comparisons"] = [
            {"output": 0, "passed": True, "message": ""},
        ]
        nodes = [_l3_node(failure=marker)]
    elif marker_kind == "candidate_case_mismatch":
        marker = _l3_candidate_failure()
        nodes = [_l3_node(case_id="21", failure=marker)]
    elif marker_kind == "candidate_with_infra_row":
        marker = _l3_candidate_failure()
        nodes = [_l3_node(failure=marker)]
    elif marker_kind == "baseline_mismatch":
        marker = _l3_infrastructure_failure(
            category="baseline_mismatch",
            stage="comparison",
        )
        nodes = [_l3_node(failure=marker)]
    elif marker_kind == "coverage_case_identity":
        marker = _l3_infrastructure_failure(
            kind="coverage",
            category="case_identity",
            stage="contract",
            case_id=None,
        )
        nodes = [_l3_node(failure=marker)]
    elif marker_kind == "runtime_exception":
        marker = _l3_infrastructure_failure()
        nodes = [_l3_node(failure=marker)]

    report = _l3_report(
        candidate=candidate,
        status="FAIL",
        coverage=_l3_coverage(coverage_kind),
        failure_category=failure_category,
        failure=marker,
        nodes=nodes,
    )
    evidence_block = report["gates"][0]["observations"]["evidence"]
    assert isinstance(evidence_block, dict)
    if marker_kind == "candidate_with_infra_row":
        evidence_block["failures"] = [
            dict(marker or {}),
            _l3_infrastructure_failure(category="exception", detail="kernel crashed"),
        ]
        evidence_block["failure"] = _l3_infrastructure_failure(
            category="exception", detail="kernel crashed"
        )
    interruption = evidence_block["interruption"]
    assert isinstance(interruption, dict)
    interruption["errorCategory"] = interruption_error
    handler, _calls = _v2_handler(
        report=report,
        root=tmp_path,
        verification=_verified(tmp_path, satisfied=False, failures=["gate failed"]),
    )
    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", handler)
    result = surrogate_cannbench.run_l3_full_contract_via_agent_gates(
        operator_name="attention_residual",
        candidate_module_path=candidate,
        source_root=tmp_path,
    )
    assert result.passed is False
    assert result.failure_kind == expected_kind
    assert result.pending is (expected_kind != surrogate_cannbench.FAILURE_KIND_CANDIDATE)
    assert result.run_id == "run-l3"
    assert result.execution_id == "exec-1"


def test_l3_promotion_treats_incomplete_case_count_as_infrastructure(
    tmp_path: Path, monkeypatch
) -> None:
    """A qualified report whose case count is short is a coverage fault, not a candidate miss."""
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(_agent_gates_config(tmp_path)))
    monkeypatch.setattr(surrogate_cannbench, "_resolve_agent_gates_bin", lambda _r: "agent-gates")
    candidate = tmp_path / "kernels.py"
    candidate.write_text("x = 1\n", encoding="utf-8")
    report = _l3_report(
        candidate=candidate,
        coverage={
            "qualified": True,
            "nodes": {
                "expected": 2,
                "completed": 2,
                "missing": [],
                "failed": [],
                "skipped": [],
                "blocked": [],
            },
            "cases": {
                "expected": 20,
                "completed": 19,
                "missing": [],
                "unexpected": [],
            },
            "referenceCacheProblem": None,
        },
    )
    handler, _calls = _v2_handler(report=report, root=tmp_path)
    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", handler)
    result = surrogate_cannbench.run_l3_full_contract_via_agent_gates(
        operator_name="attention_residual",
        candidate_module_path=candidate,
        source_root=tmp_path,
    )
    assert result.passed is False
    assert result.failure_kind == surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE
    assert "case coverage incomplete" in result.error_message


@pytest.mark.parametrize("marker", ["write_errors", "unfinished_phase"])
def test_l3_promotion_untrustworthy_numeric_evidence_is_infrastructure(
    tmp_path: Path, monkeypatch, marker: str
) -> None:
    """Evidence-write failure or a stale phase overrides a claimed numeric failure."""
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(_agent_gates_config(tmp_path)))
    monkeypatch.setattr(surrogate_cannbench, "_resolve_agent_gates_bin", lambda _r: "agent-gates")
    candidate = tmp_path / "kernels.py"
    candidate.write_text("x = 1\n", encoding="utf-8")
    candidate_failure = _l3_candidate_failure()
    report = _l3_report(
        candidate=candidate,
        status="FAIL",
        coverage=_l3_coverage("failed"),
        failure_category="numeric",
        failure=candidate_failure,
        nodes=[_l3_node(failure=candidate_failure)],
    )
    evidence_block = report["gates"][0]["observations"]["evidence"]
    assert isinstance(evidence_block, dict)
    if marker == "write_errors":
        evidence_block["writeErrors"] = ["evidence write failed"]
    else:
        evidence_block["phase"] = "verifying"
    handler, _calls = _v2_handler(
        report=report,
        root=tmp_path,
        verification=_verified(tmp_path, satisfied=False, failures=["gate failed"]),
    )
    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", handler)
    result = surrogate_cannbench.run_l3_full_contract_via_agent_gates(
        operator_name="attention_residual",
        candidate_module_path=candidate,
        source_root=tmp_path,
    )
    assert result.passed is False
    assert result.failure_kind == surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE
    assert result.execution_terminal is True


def test_l3_observation_mapping_matches_acceptance_contract(tmp_path: Path) -> None:
    """The pure observation mapping separates producer evidence from failure state."""
    candidate = tmp_path / "kernels.py"
    candidate.write_text("x = 1\n", encoding="utf-8")
    candidate_digest = _digest(candidate)
    candidate_failure = _l3_candidate_failure()
    runtime_failure = _l3_infrastructure_failure()
    coverage_failure = _l3_infrastructure_failure(
        kind="coverage",
        category="case_identity",
        stage="contract",
        case_id=None,
    )

    def _observation(
        failure: Mapping[str, object],
        *,
        category: str,
        failures: list[Mapping[str, object]] | None = None,
        write_errors: list[str] | None = None,
    ) -> dict[str, object]:
        evidence = {
            "phase": "finished",
            "failure": dict(failure),
            "failures": [dict(row) for row in failures or [failure]],
            "writeErrors": list(write_errors or []),
            "nodes": [_l3_node(failure=failure)],
            "interruption": {
                "interrupted": False,
                "errorCategory": None,
                "signal": None,
            },
        }
        return {"failureCategory": category, "evidence": evidence}

    observations = [
        _observation(runtime_failure, category="exception"),
        _observation(candidate_failure, category="numeric"),
        _observation(coverage_failure, category="coverage"),
        _observation(
            runtime_failure,
            category="numeric",
            failures=[candidate_failure, runtime_failure],
        ),
        _observation(
            candidate_failure,
            category="numeric",
            write_errors=["evidence write failed"],
        ),
    ]
    mapped = [
        surrogate_cannbench.l3_failure_result_from_observation(
            observation,
            run_id="run-l3",
            receipt_path="/receipt-l3.json",
            execution_id="exec-1",
            candidate_sha256=candidate_digest,
            target_path=str(candidate),
            device_id="4",
            state="FINISHED",
            execution_terminal=True,
        )
        for observation in observations
    ]

    assert [
        (result.failure_kind, result.failure_category, result.pending)
        for result in mapped
    ] == [
        (surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE, "exception", True),
        (surrogate_cannbench.FAILURE_KIND_CANDIDATE, "numeric", False),
        (surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE, "coverage", True),
        (surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE, "numeric", True),
        (surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE, "numeric", True),
    ]
    assert all(result.run_id == "run-l3" for result in mapped)
    assert all(result.execution_id == "exec-1" for result in mapped)
    assert all(result.device_id == "4" for result in mapped)
    assert all(result.state == "FINISHED" for result in mapped)
    assert all(result.candidate_sha256 == candidate_digest for result in mapped)
    assert all(result.target_path == str(candidate) for result in mapped)
    assert all(result.execution_terminal is True for result in mapped)


def test_l3_promotion_rejects_candidate_identity_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    """The signed L3 evidence must bind the exact target, candidate path and digest."""
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(_agent_gates_config(tmp_path)))
    monkeypatch.setattr(surrogate_cannbench, "_resolve_agent_gates_bin", lambda _r: "agent-gates")
    candidate = tmp_path / "kernels.py"
    candidate.write_text("x = 1\n", encoding="utf-8")
    other = tmp_path / "other.py"
    other.write_text("x = 2\n", encoding="utf-8")

    report = _l3_report(candidate=other)
    evidence_block = report["gates"][0]["observations"]["evidence"]
    assert isinstance(evidence_block, dict)
    evidence_block["target"]["modulePath"] = str(candidate)  # type: ignore[index]
    handler, _calls = _v2_handler(report=report, root=tmp_path)
    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", handler)
    path_mismatch = surrogate_cannbench.run_l3_full_contract_via_agent_gates(
        operator_name="attention_residual",
        candidate_module_path=candidate,
        source_root=tmp_path,
    )
    assert path_mismatch.passed is False
    assert path_mismatch.failure_kind == surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE
    assert "candidate path" in path_mismatch.error_message

    report = _l3_report(candidate=other)
    evidence_block = report["gates"][0]["observations"]["evidence"]
    assert isinstance(evidence_block, dict)
    evidence_block["target"]["modulePath"] = str(candidate)  # type: ignore[index]
    declared = evidence_block["candidate"]
    assert isinstance(declared, dict)
    declared["modulePath"] = str(candidate)
    handler, _calls = _v2_handler(report=report, root=tmp_path)
    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", handler)
    digest_mismatch = surrogate_cannbench.run_l3_full_contract_via_agent_gates(
        operator_name="attention_residual",
        candidate_module_path=candidate,
        source_root=tmp_path,
    )
    assert digest_mismatch.passed is False
    assert digest_mismatch.failure_kind == surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE
    assert "digest" in digest_mismatch.error_message


def test_l3_promotion_rejects_unverified_receipt(tmp_path: Path, monkeypatch) -> None:
    """A receipt that verify-requirements does not satisfy is infrastructure, not a candidate."""
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(_agent_gates_config(tmp_path)))
    monkeypatch.setattr(surrogate_cannbench, "_resolve_agent_gates_bin", lambda _r: "agent-gates")
    candidate = tmp_path / "kernels.py"
    candidate.write_text("x = 1\n", encoding="utf-8")
    handler, _calls = _v2_handler(
        report=_l3_report(candidate=candidate),
        root=tmp_path,
        verification=_verified(
            tmp_path,
            satisfied=False,
            failures=["RECEIPT_INVALID: signature"],
        ),
    )
    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", handler)
    stale = surrogate_cannbench.run_l3_full_contract_via_agent_gates(
        operator_name="attention_residual",
        candidate_module_path=candidate,
        source_root=tmp_path,
    )
    assert stale.passed is False
    assert stale.failure_kind == surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE
    assert "did not verify" in stale.error_message
    assert stale.run_id == "run-l3"
    assert stale.receipt_path == "/receipt-l3.json"


def test_l3_promotion_timeout_preserves_execution_id(tmp_path: Path, monkeypatch) -> None:
    """A bounded deadline reports the execution id without resubmitting the recipe."""
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(_agent_gates_config(tmp_path)))
    monkeypatch.setattr(surrogate_cannbench, "_resolve_agent_gates_bin", lambda _r: "agent-gates")
    candidate = tmp_path / "kernels.py"
    candidate.write_text("x = 1\n", encoding="utf-8")
    commands: list[str] = []

    def _fake_run(cmd, *, cwd, env, timeout, **kwargs):
        _ = (cwd, env, timeout, kwargs)
        cmd_list = [str(part) for part in cmd]
        commands.append(cmd_list[1])
        if cmd_list[1] == "doctor":
            return _completed(cmd_list, {"status": "PASS"})
        return _completed(cmd_list, {"execution_id": "exec-timeout"})

    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", _fake_run)
    monkeypatch.setattr(surrogate_cannbench.time, "monotonic", _JumpClock())
    result = surrogate_cannbench.run_l3_full_contract_via_agent_gates(
        operator_name="attention_residual",
        candidate_module_path=candidate,
        source_root=tmp_path,
        timeout_sec=1,
    )
    assert result.passed is False
    assert result.failure_kind == surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE
    assert result.execution_id == "exec-timeout"
    assert result.execution_terminal is False
    assert "timed out" in result.error_message
    assert commands == ["doctor", "submit"]


def test_l3_promotion_submits_requested_device_and_reports_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    """The requested card is sent on submit; the callback persists the resolved identity."""
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(_agent_gates_config(tmp_path)))
    monkeypatch.setattr(surrogate_cannbench, "_resolve_agent_gates_bin", lambda _r: "agent-gates")
    candidate = tmp_path / "kernels.py"
    candidate.write_text("x = 1\n", encoding="utf-8")
    handler, calls = _v2_handler(
        report=_l3_report(candidate=candidate),
        root=tmp_path,
        submit_payload={"execution_id": "exec-dev", "device_id": "5"},
    )
    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", handler)
    started: list[tuple[str, str]] = []
    result = surrogate_cannbench.run_l3_full_contract_via_agent_gates(
        operator_name="attention_residual",
        candidate_module_path=candidate,
        source_root=tmp_path,
        device_id="3",
        on_execution_started=lambda execution_id, device_id: started.append(
            (execution_id, device_id)
        ),
    )
    assert result.passed is True
    assert result.execution_id == "exec-dev"
    assert result.device_id == "5"
    assert calls[0][1] == "doctor"
    assert calls[0][calls[0].index("--device-id") + 1] == "3"
    submit = [cmd for cmd in calls if cmd[1] == "submit"][0]
    assert submit[submit.index("--device-id") + 1] == "3"
    assert started == [("exec-dev", "5")]


def test_l3_promotion_resumes_saved_execution_without_resubmit(
    tmp_path: Path, monkeypatch
) -> None:
    """A saved execution id is waited on directly; no duplicate submit is made."""
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(_agent_gates_config(tmp_path)))
    monkeypatch.setattr(surrogate_cannbench, "_resolve_agent_gates_bin", lambda _r: "agent-gates")
    monkeypatch.setattr(
        surrogate_cannbench,
        "_agent_gates_evidence_root",
        lambda: tmp_path / "v2-evidence",
    )
    candidate = tmp_path / "kernels.py"
    candidate.write_text("x = 1\n", encoding="utf-8")
    handler, calls = _v2_handler(report=_l3_report(candidate=candidate), root=tmp_path)
    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", handler)
    result = surrogate_cannbench.run_l3_full_contract_via_agent_gates(
        operator_name="attention_residual",
        candidate_module_path=candidate,
        source_root=tmp_path,
        resume_execution_id="exec-saved",
    )
    assert result.passed is True
    assert [cmd[1] for cmd in calls] == ["wait", "report", "verify-requirements"]
    wait_cmd = calls[0]
    assert wait_cmd[wait_cmd.index("--execution-id") + 1] == "exec-saved"


def test_l3_promotion_resumes_finished_run_without_wait_or_submit(
    tmp_path: Path, monkeypatch
) -> None:
    """A finished run whose report failed is recovered from the same run id."""
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(_agent_gates_config(tmp_path)))
    monkeypatch.setattr(surrogate_cannbench, "_resolve_agent_gates_bin", lambda _r: "agent-gates")
    monkeypatch.setattr(
        surrogate_cannbench,
        "_agent_gates_evidence_root",
        lambda: tmp_path / "v2-evidence",
    )
    candidate = tmp_path / "kernels.py"
    candidate.write_text("x = 1\n", encoding="utf-8")
    handler, calls = _v2_handler(report=_l3_report(candidate=candidate), root=tmp_path)
    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", handler)
    result = surrogate_cannbench.run_l3_full_contract_via_agent_gates(
        operator_name="attention_residual",
        candidate_module_path=candidate,
        source_root=tmp_path,
        resume_execution_id="exec-saved",
        resume_run_id="run-l3",
    )
    assert result.passed is True
    assert [cmd[1] for cmd in calls] == ["report", "verify-requirements"]


def test_l3_promotion_resumed_wait_timeout_keeps_execution_id(
    tmp_path: Path, monkeypatch
) -> None:
    """An expired continuation reports the original execution id and never resubmits."""
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(_agent_gates_config(tmp_path)))
    monkeypatch.setattr(surrogate_cannbench, "_resolve_agent_gates_bin", lambda _r: "agent-gates")
    candidate = tmp_path / "kernels.py"
    candidate.write_text("x = 1\n", encoding="utf-8")
    commands: list[str] = []

    def _fake_run(cmd, *, cwd, env, timeout, **kwargs):
        _ = (cwd, env, timeout, kwargs)
        commands.append([str(part) for part in cmd][1])
        raise AssertionError("resume must not submit")

    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", _fake_run)
    monkeypatch.setattr(surrogate_cannbench.time, "monotonic", _JumpClock())
    result = surrogate_cannbench.run_l3_full_contract_via_agent_gates(
        operator_name="attention_residual",
        candidate_module_path=candidate,
        source_root=tmp_path,
        timeout_sec=1,
        resume_execution_id="exec-saved",
    )
    assert result.passed is False
    assert result.execution_id == "exec-saved"
    assert "timed out" in result.error_message
    assert commands == []


def _v2_l4_run(
    tmp_path: Path,
    monkeypatch,
    *,
    report: Mapping[str, object],
    verification: Mapping[str, object] | None = None,
    submit_payload: Mapping[str, object] | None = None,
    resume_execution_id: str = "",
    resume_run_id: str = "",
    on_execution_started=None,
) -> tuple[Any, list[list[str]], list[dict[str, str]]]:
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(_agent_gates_config(tmp_path)))
    monkeypatch.setattr(surrogate_cannbench, "_resolve_agent_gates_bin", lambda _r: "agent-gates")
    monkeypatch.setattr(
        surrogate_cannbench,
        "_agent_gates_evidence_root",
        lambda: tmp_path / "v2-evidence",
    )
    envs: list[dict[str, str]] = []
    handler, calls = _v2_handler(
        report=report,
        root=tmp_path,
        verification=verification,
        submit_payload=submit_payload,
    )

    def _fake_run(cmd, *, cwd, env, timeout, **kwargs):
        envs.append(dict(env))
        return handler(cmd, cwd=cwd, env=env, timeout=timeout, **kwargs)

    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", _fake_run)
    candidate = tmp_path / "kernels.py"
    candidate.write_text("x = 1\n", encoding="utf-8")
    result = surrogate_cannbench.run_l4_official_cannbench_via_agent_gates(
        operator_name="attention_residual",
        candidate_module_path=candidate,
        source_root=tmp_path,
        env_file=tmp_path / "ignored-env.sh",
        device_id="7",
        resume_execution_id=resume_execution_id,
        resume_run_id=resume_run_id,
        on_execution_started=on_execution_started,
    )
    return result, calls, envs


def test_l4_always_uses_v2_and_requires_official_final_binding(
    tmp_path: Path, monkeypatch
) -> None:
    """L4 ignores legacy env/device overrides and accepts only bound official evidence."""
    report = _l4_report(candidate="kernels.py")
    result, calls, envs = _v2_l4_run(tmp_path, monkeypatch, report=report)
    assert result.passed is True
    assert result.run_id == "run-l4"
    assert result.avg_speedup == 4.25
    assert result.score == 99.5
    assert result.per_case_speedups == {2: 3.5}
    assert result.per_case_elapsed_us == {2: 12.5}
    assert result.report_dir == "/signed-report"
    assert result.official_report_valid is True
    assert result.all_cases_passed is True
    assert result.all_timings_valid is True
    assert result.case_count == 1
    assert result.passed_cases == 1
    doctor = calls[0]
    assert doctor[1] == "doctor"
    assert doctor[doctor.index("--source") + 1] == str(tmp_path.resolve())
    assert doctor[doctor.index("--gate") + 1] == _L4_GATE
    assert doctor[doctor.index("--target") + 1] == "level4/attention_residual"
    assert doctor[doctor.index("--candidate") + 1] == "kernels.py"
    assert "--workspace" not in doctor
    assert "--env-file" not in doctor
    submit = [cmd for cmd in calls if cmd[1] == "submit"][0]
    assert submit[submit.index("--gate") + 1] == _L4_GATE
    assert submit[submit.index("--target") + 1] == "level4/attention_residual"
    assert submit[submit.index("--candidate") + 1] == "kernels.py"
    assert "--env-file" not in submit
    assert envs and all("PYTHONPATH" not in env for env in envs)
    verify = [cmd for cmd in calls if cmd[1] == "verify-requirements"][0]
    assert verify[verify.index("--target") + 1] == "level4/attention_residual"
    assert verify[verify.index("--candidate") + 1] == "kernels.py"


def test_l4_submits_requested_device_and_reports_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    """L4 sends the authorized card and exposes the resolved execution identity."""
    report = _l4_report(candidate="kernels.py")
    started: list[tuple[str, str]] = []
    result, calls, _envs = _v2_l4_run(
        tmp_path,
        monkeypatch,
        report=report,
        submit_payload={"execution_id": "exec-l4", "device_id": "8"},
        on_execution_started=lambda execution_id, device_id: started.append(
            (execution_id, device_id)
        ),
    )
    assert result.passed is True
    assert result.execution_id == "exec-l4"
    assert result.device_id == "8"
    assert calls[0][1] == "doctor"
    assert calls[0][calls[0].index("--device-id") + 1] == "7"
    submit = [cmd for cmd in calls if cmd[1] == "submit"][0]
    assert submit[submit.index("--device-id") + 1] == "7"
    assert started == [("exec-l4", "8")]


def test_l4_resumes_execution_without_duplicate_submit(
    tmp_path: Path, monkeypatch
) -> None:
    """A saved L4 execution is continued on its original card; no fresh submit occurs."""
    report = _l4_report(candidate="kernels.py")
    result, calls, _envs = _v2_l4_run(
        tmp_path,
        monkeypatch,
        report=report,
        resume_execution_id="exec-saved",
    )
    assert result.passed is True
    assert result.execution_id == "exec-saved"
    assert result.device_id == "7"
    assert [cmd[1] for cmd in calls] == ["wait", "report", "verify-requirements"]
    wait = calls[0]
    assert wait[wait.index("--execution-id") + 1] == "exec-saved"

    result, calls, _envs = _v2_l4_run(
        tmp_path,
        monkeypatch,
        report=report,
        resume_execution_id="exec-saved",
        resume_run_id="run-l4",
    )
    assert result.passed is True
    assert [cmd[1] for cmd in calls] == ["report", "verify-requirements"]
    assert "submit" not in [cmd[1] for cmd in calls]


def test_l4_doctor_blocks_submit_when_route_is_not_ready(
    tmp_path: Path, monkeypatch
) -> None:
    """A blocked machine is refused before an execution exists, never blamed on the candidate."""
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(_agent_gates_config(tmp_path)))
    monkeypatch.setattr(surrogate_cannbench, "_resolve_agent_gates_bin", lambda _r: "agent-gates")
    monkeypatch.setattr(
        surrogate_cannbench,
        "_agent_gates_evidence_root",
        lambda: tmp_path / "v2-evidence",
    )
    candidate = tmp_path / "kernels.py"
    candidate.write_text("x = 1\n", encoding="utf-8")
    handler, calls = _v2_handler(
        report=_l4_report(candidate="kernels.py"),
        root=tmp_path,
        doctor={"status": "BLOCKED_ENV", "failures": ["CONFIG_CHECK_FAILED: device"]},
        doctor_returncode=1,
    )
    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", handler)
    result = surrogate_cannbench.run_l4_official_cannbench_via_agent_gates(
        operator_name="attention_residual",
        candidate_module_path=candidate,
        source_root=tmp_path,
    )
    assert result.passed is False
    assert result.failure_kind == surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE
    assert result.execution_id == ""
    assert result.execution_terminal is True
    assert "CONFIG_CHECK_FAILED: device" in result.error_message
    assert [cmd[1] for cmd in calls] == ["doctor"]


def test_l4_default_wait_covers_queue_prep_and_official_run() -> None:
    """The 3600s official recipe alone cannot fit the default L4 wait budget."""
    assert surrogate_cannbench.DEFAULT_L4_CANNBENCH_WAIT_SEC >= 4500


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
def test_l4_rejects_incomplete_official_final(tmp_path: Path, monkeypatch, flags) -> None:
    """Missing official_final, failed cases, or invalid timings never yield metrics."""
    report = _l4_report(candidate="kernels.py", flags=flags)
    result, _calls, _envs = _v2_l4_run(tmp_path, monkeypatch, report=report)
    assert result.passed is False
    assert result.per_case_speedups == {}
    assert result.failure_kind in {
        surrogate_cannbench.FAILURE_KIND_CANDIDATE,
        surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE,
    }


@pytest.mark.parametrize(
    "flags",
    [{"caseCount": 0}, {"timedCases": 0}, {"passedCases": None}],
)
def test_l4_missing_case_coverage_preserves_candidate(
    tmp_path: Path, monkeypatch, flags
) -> None:
    """Absent official cases or timings are unknown coverage, not a candidate regression."""
    report = _l4_report(candidate="kernels.py", flags=flags)
    result, _calls, _envs = _v2_l4_run(tmp_path, monkeypatch, report=report)
    assert result.passed is False
    assert result.failure_kind == surrogate_cannbench.FAILURE_KIND_INFRASTRUCTURE


def test_l4_rejects_candidate_mismatch_and_unsatisfied_requirements(
    tmp_path: Path, monkeypatch
) -> None:
    """The signed L4 receipt must name the candidate the campaign wrote."""
    report = _l4_report(candidate="some/other.py")
    result, _calls, _envs = _v2_l4_run(tmp_path, monkeypatch, report=report)
    assert result.passed is False
    assert "candidate" in result.error_message

    report = _l4_report(candidate="kernels.py")
    result, calls, _envs = _v2_l4_run(
        tmp_path,
        monkeypatch,
        report=report,
        verification=_verified(
            tmp_path,
            satisfied=False,
            failures=["REQUIREMENTS_UNSATISFIED: stale"],
        ),
    )
    assert result.passed is False
    assert "did not verify" in result.error_message
    assert result.run_id == "run-l4"
    assert result.receipt_path == "/receipt-l4.json"
    assert [cmd[1] for cmd in calls] == [
        "doctor",
        "submit",
        "wait",
        "report",
        "verify-requirements",
    ]


def test_l4_v2_timeout_includes_execution_id(tmp_path: Path, monkeypatch) -> None:
    """The wait budget reports the execution id so an independent worker is recoverable."""
    monkeypatch.setenv("AGENT_GATES_CONFIG", str(_agent_gates_config(tmp_path)))
    monkeypatch.setattr(surrogate_cannbench, "_resolve_agent_gates_bin", lambda _r: "agent-gates")
    candidate = tmp_path / "kernels.py"
    candidate.write_text("x = 1\n", encoding="utf-8")

    def _fake_run(cmd, *, cwd, env, timeout, **kwargs):
        _ = (cwd, env, timeout, kwargs)
        cmd_list = [str(part) for part in cmd]
        if cmd_list[1] == "doctor":
            return _completed(cmd_list, {"status": "PASS"})
        assert cmd_list[1] == "submit"
        return _completed(cmd_list, {"execution_id": "exec-timeout"})

    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", _fake_run)
    monkeypatch.setattr(surrogate_cannbench.time, "monotonic", _JumpClock())
    result = surrogate_cannbench.run_l4_official_cannbench_via_agent_gates(
        operator_name="attention_residual",
        candidate_module_path=candidate,
        source_root=tmp_path,
        timeout_sec=1,
    )
    assert result.passed is False
    assert result.execution_id == "exec-timeout"
    assert "exec-timeout" in result.error_message
    assert "timed out" in result.error_message


def test_agent_gates_bin_has_no_sibling_venv_fallback(tmp_path: Path, monkeypatch) -> None:
    """The CLI must come from explicit configuration or PATH, never a sibling checkout."""
    source_root = tmp_path / "TileFoundry"
    source_root.mkdir()
    sibling = tmp_path / "AgentGates" / ".venv" / "bin" / "agent-gates"
    sibling.parent.mkdir(parents=True)
    sibling.write_text("#!/bin/sh\n", encoding="utf-8")
    sibling.chmod(0o755)
    monkeypatch.delenv("TILEFOUNDRY_NPU_AGENT_GATES_BIN", raising=False)
    monkeypatch.setattr(surrogate_cannbench.shutil, "which", lambda _name: None)
    with pytest.raises(FileNotFoundError):
        surrogate_cannbench._resolve_agent_gates_bin(source_root)
