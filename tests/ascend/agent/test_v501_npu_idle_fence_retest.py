"""Offline unit tests for multi-card idle preflight, contamination retest, and CLI alignment."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path

import pytest

from tilefoundry_ascend.agent.campaign import cli, marathon, surrogate_cannbench
from tilefoundry_ascend.agent.campaign.anchor_specs import AnchorCaseSpec
from tilefoundry_ascend.agent.campaign.campaign_runner import CampaignConfig
from tilefoundry_ascend.agent.orchestrator.llm_client import MockLLMBackend


class _DummyProbeSpec:
    probe_id = "probe_1"
    probe_kind = "vector_unroll"
    factory_name = "build_probe"
    baseline_probe_source = "def build_probe():\n    return 1\n"
    optimized_probe_source = "def build_probe():\n    return 2\n"
    line_count_baseline = 2
    line_count_optimized = 2
    tensor_shape = (64, 64)
    tensor_dtype = "float16"


def test_order_candidates_by_idle_preflight_selects_idle_card_and_handles_locks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Card 0 busy (process or lock/quarantine), card 1 idle -> preflight orders card 1 first."""
    monkeypatch.setenv("AGENT_GATES_DEVICE_LOCK_ROOT", str(tmp_path / "locks"))
    (tmp_path / "locks").mkdir()
    monkeypatch.setattr(surrogate_cannbench.shutil, "which", lambda _cmd: "/usr/bin/npu-smi")
    monkeypatch.setattr(surrogate_cannbench.time, "sleep", lambda _s: None)

    smi_calls: list[list[str]] = []
    retry_state = {"count": 0}

    def _fake_smi(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        smi_calls.append(list(cmd))
        assert "-c" not in cmd
        if cmd[1:] == ["info", "-m"]:
            stdout = (
                "NPU ID  Slot ID  Chip ID  Chip Phy-ID  Chip Name\n"
                "0       74       0        0            Ascend950PR\n"
                "1       76       0        1            Ascend950PR\n"
                "2       84       0        2            Ascend950PR\n"
            )
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
        if cmd[1:] == ["info", "-t", "proc-mem", "-i", "0"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=(
                    "NPU ID                         : 0\n"
                    "Get pid name failed.\n"
                    "Process id:2330713 Process name:                  Process memory(MB):796\n"
                ),
                stderr="",
            )
        if cmd[1:] == ["info", "-t", "proc-mem", "-i", "1"]:
            if retry_state["count"] == 0:
                retry_state["count"] += 1
                return subprocess.CompletedProcess(cmd, 215, stdout="", stderr="busy")
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="NPU ID                         : 1\nNo process in device\n",
                stderr="",
            )
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="NPU ID                         : 2\nNo process in device\n",
            stderr="",
        )

    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", _fake_smi)
    ordered = surrogate_cannbench.order_candidates_by_idle_preflight("0", ("0", "1", "2"))
    assert ordered == ("1", "2", "0")
    assert surrogate_cannbench.select_idle_campaign_device("0", ("0", "1")) == "1"
    assert retry_state["count"] == 1

    lease_hash_1 = hashlib.sha256(b"npu:1").hexdigest()
    quarantine_1 = tmp_path / "locks" / f"{lease_hash_1}.lock.quarantine.json"
    quarantine_1.write_text('{"reason": "tainted"}\n', encoding="utf-8")
    assert surrogate_cannbench._probe_host_card_idle("1") is False
    assert surrogate_cannbench.order_candidates_by_idle_preflight("0", ("0", "1", "2")) == (
        "2",
        "0",
        "1",
    )
    quarantine_1.unlink()

    lock_2 = tmp_path / "locks" / f"{hashlib.sha256(b'npu:2').hexdigest()}.lock"
    lock_2.write_text("", encoding="utf-8")
    fd = os.open(lock_2, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert surrogate_cannbench._probe_host_card_idle("2") is False
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    def _raise_perm(*_args: object, **_kwargs: object) -> int:
        raise PermissionError("unreadable lock")

    monkeypatch.setattr(surrogate_cannbench.os, "open", _raise_perm)
    assert surrogate_cannbench._probe_host_card_idle("2") is False

    monkeypatch.setattr(surrogate_cannbench.shutil, "which", lambda _cmd: None)
    assert surrogate_cannbench.order_candidates_by_idle_preflight(
        "0",
        ("0", "1", "2"),
        excluded={"2"},
    ) == ("0", "1")


def test_submit_fast_gate_unlinks_dirty_response_and_retests_on_occupancy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """_submit_fast_gate unlinks dirty response_path on occupancy and switches to card 2."""
    req_path = tmp_path / "req.json"
    req_bytes = b'{"schema": "tilefoundry-fast-request/v1"}\n'
    req_path.write_bytes(req_bytes)
    req_sha = hashlib.sha256(req_bytes).hexdigest()
    resp_path = tmp_path / "resp.json"
    resp_path.write_text('{"status": "stale_contaminated"}\n', encoding="utf-8")

    monkeypatch.setattr(
        surrogate_cannbench,
        "order_candidates_by_idle_preflight",
        lambda device_id, pool=(), excluded=(): tuple(
            c for c in ("1", "2") if c not in excluded
        ),
    )
    monkeypatch.setattr(surrogate_cannbench, "_source_identity_matches", lambda *_a: True)

    seen_devices: list[str] = []
    observed_resp_exists_on_second_call: list[bool] = []

    def _fake_v2(**kwargs: object) -> surrogate_cannbench._V2GateExecution:
        dev = str(kwargs.get("device_id") or "")
        seen_devices.append(dev)
        if dev == "1":
            return surrogate_cannbench._V2GateExecution(
                execution_id="exec-1",
                report={
                    "status": "BLOCKED_ENV",
                    "gates": [
                        {
                            "gateId": surrogate_cannbench.AGENT_GATES_DEVICE_FAST_GATE,
                            "recordedStatus": "BLOCKED_ENV",
                            "failures": ["BLOCKED_EXIT_CODE: 75"],
                            "observations": {"errorCategory": "occupancy"},
                        }
                    ],
                },
                verification={"satisfied": False, "failures": ["BLOCKED_EXIT_CODE: 75"]},
                device_id="1",
            )
        observed_resp_exists_on_second_call.append(resp_path.exists())
        resp_path.write_text(
            json.dumps(
                {
                    "schema": surrogate_cannbench.FAST_RESPONSE_SCHEMA,
                    "request_id": "req-occ",
                    "request_sha256": req_sha,
                    "mode": "batch",
                    "status": "ok",
                }
            ),
            encoding="utf-8",
        )
        return surrogate_cannbench._V2GateExecution(
            execution_id="exec-2",
            run_id="run-ok-2",
            report={"status": "PASS", "gates": []},
            verification={"satisfied": True},
            device_id="2",
        )

    monkeypatch.setattr(surrogate_cannbench, "_run_v2_gate", _fake_v2)

    with caplog.at_level(logging.WARNING):
        raw, final_exec = surrogate_cannbench._submit_fast_gate(
            agent_gates_bin="agent-gates",
            root=tmp_path,
            workspace="tilefoundry",
            request_id="req-occ",
            request_relative_path="req.json",
            request_path=req_path,
            response_path=resp_path,
            request_sha256=req_sha,
            mode="batch",
            timeout_sec=60.0,
            device_id="1",
            device_pool=("1", "2"),
        )
    assert seen_devices == ["1", "2"]
    assert observed_resp_exists_on_second_call == [False]
    assert final_exec.device_id == "2"
    assert raw["status"] == "ok"
    assert "FAST_GATE_CONTAMINATION_RETEST" in caplog.text


def test_fast_gate_entrypoints_forward_device_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All L2 batch, msprof profile, and micro-probe entrypoints forward device_pool."""
    src_root = tmp_path / "repo"
    src_root.mkdir()
    env_file = tmp_path / "env.sh"
    work_root = tmp_path / "work"
    env_file.write_text(
        f"TILEFOUNDRY_NPU_WORK_ROOT={work_root}\n"
        f"TILEFOUNDRY_NPU_CANNBENCH_REPO={tmp_path / 'cann'}\n",
        encoding="utf-8",
    )
    (tmp_path / "cann" / "tasks").mkdir(parents=True)
    cand = src_root / "cand.py"
    cand.write_text("def op():\n    return 1\n", encoding="utf-8")
    anchor = AnchorCaseSpec("level2/softmax", 1, "small", 10.0, 2.0)

    monkeypatch.setattr(surrogate_cannbench, "_resolve_agent_gates_bin", lambda _r: "agent-gates")
    monkeypatch.setattr(surrogate_cannbench, "_agent_gates_workspace_name", lambda _r: "tilefoundry")
    monkeypatch.setattr(surrogate_cannbench, "prewarm_anchor_reference_cache", lambda *a, **k: None)
    monkeypatch.setattr(surrogate_cannbench, "prewarm_surrogate_tilelang_cache", lambda *a, **k: None)
    monkeypatch.setattr(surrogate_cannbench, "_prewarm_micro_probe_cache", lambda *a, **k: None)

    seen_pools: list[tuple[str, tuple[str, ...]]] = []

    def _fake_submit(**kwargs: object) -> tuple[dict[str, object], surrogate_cannbench._V2GateExecution]:
        mode = str(kwargs["mode"])
        seen_pools.append((mode, tuple(kwargs["device_pool"])))
        req_payload = json.loads(Path(str(kwargs["request_path"])).read_text(encoding="utf-8"))
        if mode == "batch":
            c0 = req_payload["batches"][0]["candidates"][0]
            resp = {
                "schema": surrogate_cannbench.FAST_RESPONSE_SCHEMA,
                "request_id": kwargs["request_id"],
                "request_sha256": kwargs["request_sha256"],
                "mode": "batch",
                "status": "ok",
                "candidates": {
                    "softmax": {
                        "c0": {
                            "module_path": c0["module_path"],
                            "sha256": c0["sha256"],
                            "post_sha256": c0["sha256"],
                        }
                    }
                },
                "coverage": {
                    "expected_candidates": 1,
                    "evaluated_candidates": 1,
                    "expected_msprof_targets": 0,
                    "observed_msprof_targets": 0,
                },
                "operators": {
                    "softmax": {
                        "c0": {
                            "candidate_id": "c0",
                            "operator_name": "softmax",
                            "all_passed": True,
                            "avg_speedup": 2.0,
                            "min_case_speedup": 2.0,
                            "avg_roofline_eta": 0.9,
                            "surrogate_score": 80.0,
                            "anchors": [],
                        }
                    }
                },
            }
        elif mode == "profile":
            t0 = req_payload["msprof_targets"][0]
            cid = t0["candidate_id"]
            resp = {
                "schema": surrogate_cannbench.FAST_RESPONSE_SCHEMA,
                "request_id": kwargs["request_id"],
                "request_sha256": kwargs["request_sha256"],
                "mode": "profile",
                "status": "ok",
                "candidates": {
                    "softmax": {
                        cid: {
                            "module_path": t0["module_path"],
                            "sha256": t0["sha256"],
                            "post_sha256": t0["sha256"],
                        }
                    }
                },
                "coverage": {
                    "expected_candidates": 1,
                    "evaluated_candidates": 1,
                    "expected_msprof_targets": 1,
                    "observed_msprof_targets": 1,
                },
                "msprof_telemetry": {"softmax": {cid: {"1": {"aiv_vec_ratio": 0.5}}}},
            }
        else:
            cands = req_payload["candidates"]
            resp = {
                "schema": surrogate_cannbench.FAST_RESPONSE_SCHEMA,
                "request_id": kwargs["request_id"],
                "request_sha256": kwargs["request_sha256"],
                "mode": "micro_probe",
                "status": "ok",
                "candidates": {
                    "micro_probe": {
                        c["candidate_id"]: {
                            "module_path": c["module_path"],
                            "sha256": c["sha256"],
                            "post_sha256": c["sha256"],
                        }
                        for c in cands
                    }
                },
                "coverage": {
                    "expected_candidates": 2,
                    "evaluated_candidates": 2,
                    "expected_msprof_targets": 0,
                    "observed_msprof_targets": 0,
                },
                "micro_probe": {"probe_id": "probe_1", "metrics": {"speedup": 1.25}},
            }
        return resp, surrogate_cannbench._V2GateExecution(
            execution_id="exec",
            run_id="run",
            report={"status": "PASS", "gates": []},
            verification={"satisfied": True},
        )

    monkeypatch.setattr(surrogate_cannbench, "_submit_fast_gate", _fake_submit)
    pool = ("0", "1", "2")
    surrogate_cannbench.run_surrogate_batch_via_agent_gates(
        "level2/softmax",
        [("c0", cand)],
        [anchor],
        source_root=src_root,
        env_file=env_file,
        device_id="0",
        device_pool=pool,
    )
    surrogate_cannbench.collect_bottleneck_msprof_via_agent_gates(
        "level2/softmax",
        cand,
        [anchor],
        source_root=src_root,
        env_file=env_file,
        device_id="0",
        device_pool=pool,
    )
    surrogate_cannbench.collect_p2_deep_dive_via_agent_gates(
        "level2/softmax",
        cand,
        [anchor],
        source_root=src_root,
        env_file=env_file,
        device_id="0",
        device_pool=pool,
    )
    surrogate_cannbench.run_micro_probe_via_agent_gates(
        _DummyProbeSpec(),
        source_root=src_root,
        env_file=env_file,
        device_id="0",
        device_pool=pool,
    )
    assert seen_pools == [
        ("batch", pool),
        ("profile", pool),
        ("profile", pool),
        ("micro_probe", pool),
    ]


def test_l3_acceptance_result_extracts_failed_case_nums_and_occupancy_rotates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L3AcceptanceResult captures failed_case_nums=(14,) and blocked_env_category='occupancy'."""
    cand = tmp_path / "kernels.py"
    cand.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(surrogate_cannbench, "_resolve_agent_gates_bin", lambda _r: "agent-gates")
    monkeypatch.setattr(surrogate_cannbench, "_agent_gates_workspace_name", lambda _r: "tilefoundry")
    monkeypatch.setattr(
        surrogate_cannbench,
        "order_candidates_by_idle_preflight",
        lambda device_id, pool=(), excluded=(): tuple(
            c for c in ("1", "2") if c not in excluded
        ),
    )

    seen_devices: list[str] = []

    def _fake_v2(**kwargs: object) -> surrogate_cannbench._V2GateExecution:
        dev = str(kwargs.get("device_id") or "")
        seen_devices.append(dev)
        if dev == "1":
            return surrogate_cannbench._V2GateExecution(
                execution_id="exec-l3-1",
                run_id="run-l3-1",
                report={
                    "status": "BLOCKED_ENV",
                    "gates": [
                        {
                            "gateId": surrogate_cannbench.AGENT_GATES_L3_PROMOTION_GATE,
                            "recordedStatus": "BLOCKED_ENV",
                            "failures": ["BLOCKED_EXIT_CODE: 75"],
                            "observations": {"errorCategory": "occupancy"},
                        }
                    ],
                },
                verification={"satisfied": False},
                device_id="1",
            )
        return surrogate_cannbench._V2GateExecution(
            execution_id="exec-l3-2",
            run_id="run-l3-2",
            report={
                "status": "FAIL",
                "gates": [
                    {
                        "gateId": surrogate_cannbench.AGENT_GATES_L3_PROMOTION_GATE,
                        "recordedStatus": "FAIL",
                        "observations": {
                            "failureCategory": "numeric",
                            "evidence": {
                                "phase": "finished",
                                "failure": {
                                    "kind": "candidate",
                                    "category": "numeric",
                                    "role": "candidate",
                                    "stage": "comparison",
                                    "comparator": "torch.testing.assert_close",
                                    "node": "test_case_14",
                                    "case": "14",
                                    "comparisons": [{"passed": False, "message": "mismatch"}],
                                },
                                "nodes": [
                                    {
                                        "name": "test_case_14",
                                        "status": "failed",
                                        "cases": {"14": "failed", "13": "passed"},
                                    }
                                ],
                            },
                        },
                    }
                ],
            },
            verification={"satisfied": False},
            device_id="2",
        )

    monkeypatch.setattr(surrogate_cannbench, "_run_v2_gate", _fake_v2)
    res = surrogate_cannbench.run_l3_full_contract_via_agent_gates(
        operator_name="attention_residual",
        candidate_module_path=cand,
        source_root=tmp_path,
        device_id="1",
        device_pool=("1", "2"),
    )
    assert seen_devices == ["1", "2"]
    assert res.passed is False
    assert res.device_id == "2"
    assert res.failed_case_nums == (14,)
    assert res.failure_stage == "comparison"
    assert res.failure_kind == surrogate_cannbench.FAILURE_KIND_CANDIDATE

    monkeypatch.setattr(
        surrogate_cannbench,
        "order_candidates_by_idle_preflight",
        lambda device_id, pool=(), excluded=(): ("1",),
    )
    occ_res = surrogate_cannbench.run_l3_full_contract_via_agent_gates(
        operator_name="attention_residual",
        candidate_module_path=cand,
        source_root=tmp_path,
        device_id="1",
    )
    assert occ_res.passed is False
    assert occ_res.blocked_env_category == "occupancy"


def test_cli_wires_pure_golden_and_disable_early_goal_stop_to_campaign_and_marathon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cli.py forwards --pure-golden, --disable-early-goal-stop, and device_pool."""
    captured_marathon: list[marathon.MarathonConfig] = []
    monkeypatch.setattr(
        cli,
        "run_operator_marathon",
        lambda cfg: captured_marathon.append(cfg) or 0,
    )
    rc_m = cli.main(
        [
            "--operator",
            "rms_norm",
            "--marathon",
            "--pure-golden",
            "--disable-early-goal-stop",
        ]
    )
    assert rc_m == 0
    assert len(captured_marathon) == 1
    assert captured_marathon[0].pure_golden is True
    assert captured_marathon[0].disable_early_goal_stop is True
    assert captured_marathon[0].run_final_l4 is True

    captured_configs: list[CampaignConfig] = []
    captured_probe_calls: list[dict[str, object]] = []

    class _FakeReport:
        def to_dict(self) -> dict[str, object]:
            return {"status": "ok"}

    class _FakeRunner:
        def __init__(self, **kwargs: object) -> None:
            captured_configs.append(kwargs["config"])
            self.flywheel = kwargs["flywheel_distiller"]

        async def run(self, *, resume: bool) -> _FakeReport:
            _ = resume
            probe = self.flywheel._probe_runner
            assert probe is not None
            probe(_DummyProbeSpec())
            return _FakeReport()

    env_file = tmp_path / "env.sh"
    env_file.write_text(
        "TILEFOUNDRY_NPU_LLM_BACKEND=mock\n"
        "TILEFOUNDRY_NPU_CAMPAIGN_DEVICE_IDS=0,1,2\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "PerfTargetCampaignRunner", _FakeRunner)
    monkeypatch.setattr(cli, "create_llm_backend_from_env", lambda **_k: MockLLMBackend())
    monkeypatch.setattr(
        cli,
        "run_micro_probe_via_agent_gates",
        lambda spec, **kwargs: captured_probe_calls.append(dict(kwargs)) or {"speedup": 1.2},
    )

    rc_c = cli.main(
        [
            "--operator",
            "rms_norm",
            "--checkpoint",
            str(tmp_path / "ckpt.json"),
            "--env-file",
            str(env_file),
            "--pure-golden",
            "--disable-early-goal-stop",
        ]
    )
    assert rc_c == 0
    assert len(captured_configs) == 1
    assert captured_configs[0].pure_golden is True
    assert captured_configs[0].disable_early_goal_stop is True
    assert captured_probe_calls[0]["device_pool"] == ("0", "1", "2")


def test_submit_fast_gate_handles_device_rejected_and_full_pool_backoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_submit_fast_gate excludes rejected cards and sleeps 4s between full pool passes."""
    req_path = tmp_path / "req.json"
    req_bytes = b'{"schema": "tilefoundry-fast-request/v1"}\n'
    req_path.write_bytes(req_bytes)
    req_sha = hashlib.sha256(req_bytes).hexdigest()
    resp_path = tmp_path / "resp.json"

    monkeypatch.setattr(
        surrogate_cannbench,
        "order_candidates_by_idle_preflight",
        lambda device_id, pool=(), excluded=(): tuple(
            c for c in ("0", "1") if c not in excluded
        ),
    )
    monkeypatch.setattr(surrogate_cannbench, "_source_identity_matches", lambda *_a: True)

    sleeps: list[float] = []
    monkeypatch.setattr(surrogate_cannbench.time, "sleep", lambda sec: sleeps.append(float(sec)))

    seen: list[str] = []

    def _fake_v2(**kwargs: object) -> surrogate_cannbench._V2GateExecution:
        dev = str(kwargs.get("device_id") or "")
        seen.append(dev)
        if dev == "0":
            return surrogate_cannbench._V2GateExecution(
                transport_error="DEVICE_NOT_AUTHORIZED",
                device_id="0",
                device_rejected=True,
            )
        if len(seen) == 2:
            return surrogate_cannbench._V2GateExecution(
                execution_id="exec-1-occ",
                report={
                    "status": "BLOCKED_ENV",
                    "gates": [
                        {
                            "gateId": surrogate_cannbench.AGENT_GATES_DEVICE_FAST_GATE,
                            "status": "BLOCKED_ENV",
                            "failures": ["BLOCKED_EXIT_CODE: 75"],
                            "observations": {"failureCategory": "occupancy"},
                        }
                    ],
                },
                verification={"satisfied": False},
                device_id="1",
            )
        resp_path.write_text(
            json.dumps(
                {
                    "schema": surrogate_cannbench.FAST_RESPONSE_SCHEMA,
                    "request_id": "req-2pass",
                    "request_sha256": req_sha,
                    "mode": "batch",
                    "status": "ok",
                }
            ),
            encoding="utf-8",
        )
        return surrogate_cannbench._V2GateExecution(
            execution_id="exec-1-ok",
            run_id="run-1-ok",
            report={"status": "PASS", "gates": []},
            verification={"satisfied": True},
            device_id="1",
        )

    monkeypatch.setattr(surrogate_cannbench, "_run_v2_gate", _fake_v2)
    raw, final_exec = surrogate_cannbench._submit_fast_gate(
        agent_gates_bin="agent-gates",
        root=tmp_path,
        workspace="tilefoundry",
        request_id="req-2pass",
        request_relative_path="req.json",
        request_path=req_path,
        response_path=resp_path,
        request_sha256=req_sha,
        mode="batch",
        timeout_sec=60.0,
        device_id="0",
        device_pool=("0", "1"),
    )
    assert seen == ["0", "1", "1"]
    assert sleeps == [4.0]
    assert final_exec.run_id == "run-1-ok"
    assert raw["status"] == "ok"


def test_marathon_cli_parses_disable_early_goal_stop_and_pure_golden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """marathon.main forwards --disable-early-goal-stop and --pure-golden to MarathonConfig."""
    captured: list[marathon.MarathonConfig] = []
    monkeypatch.setattr(marathon, "run_operator_marathon", lambda cfg: captured.append(cfg) or 0)
    rc = marathon.main(
        [
            "run",
            "--operator",
            "rms_norm",
            "--pure-golden",
            "--disable-early-goal-stop",
        ]
    )
    assert rc == 0
    assert captured[0].pure_golden is True
    assert captured[0].disable_early_goal_stop is True


def test_submit_fast_gate_progressive_occupancy_backoff_and_budget_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_submit_fast_gate uses progressive backoff across passes and stops at budget."""
    req_path = tmp_path / "req.json"
    req_bytes = b'{"schema": "tilefoundry-fast-request/v1"}\n'
    req_path.write_bytes(req_bytes)
    req_sha = hashlib.sha256(req_bytes).hexdigest()
    resp_path = tmp_path / "resp.json"

    monkeypatch.setattr(
        surrogate_cannbench,
        "order_candidates_by_idle_preflight",
        lambda device_id, pool=(), excluded=(): tuple(
            c for c in ("1", "2") if c not in excluded
        ),
    )
    monkeypatch.setattr(surrogate_cannbench, "_source_identity_matches", lambda *_a: True)

    sleeps: list[float] = []
    monkeypatch.setattr(surrogate_cannbench.time, "sleep", lambda sec: sleeps.append(float(sec)))

    attempts = 0

    def _fake_v2_pass4(**kwargs: object) -> surrogate_cannbench._V2GateExecution:
        nonlocal attempts
        attempts += 1
        dev = str(kwargs.get("device_id") or "")
        if attempts < 8:
            return surrogate_cannbench._V2GateExecution(
                execution_id=f"exec-occ-{attempts}",
                report={
                    "status": "BLOCKED_ENV",
                    "gates": [
                        {
                            "gateId": surrogate_cannbench.AGENT_GATES_DEVICE_FAST_GATE,
                            "status": "BLOCKED_ENV",
                            "failures": ["BLOCKED_EXIT_CODE: 75"],
                            "observations": {"errorCategory": "occupancy"},
                        }
                    ],
                },
                verification={"satisfied": False},
                device_id=dev,
            )
        resp_path.write_text(
            json.dumps(
                {
                    "schema": surrogate_cannbench.FAST_RESPONSE_SCHEMA,
                    "request_id": "req-prog",
                    "request_sha256": req_sha,
                    "mode": "batch",
                    "status": "ok",
                }
            ),
            encoding="utf-8",
        )
        return surrogate_cannbench._V2GateExecution(
            execution_id="exec-pass-4",
            run_id="run-pass-4",
            report={"status": "PASS", "gates": []},
            verification={"satisfied": True},
            device_id=dev,
        )

    monkeypatch.setattr(surrogate_cannbench, "_run_v2_gate", _fake_v2_pass4)
    raw, final_exec = surrogate_cannbench._submit_fast_gate(
        agent_gates_bin="agent-gates",
        root=tmp_path,
        workspace="tilefoundry",
        request_id="req-prog",
        request_relative_path="req.json",
        request_path=req_path,
        response_path=resp_path,
        request_sha256=req_sha,
        mode="batch",
        timeout_sec=180.0,
        device_id="1",
        device_pool=("1", "2"),
    )
    assert attempts == 8
    assert sleeps == [4.0, 6.0, 8.0]
    assert final_exec.run_id == "run-pass-4"
    assert raw["status"] == "ok"

    sleeps.clear()

    def _fake_v2_always_busy(**kwargs: object) -> surrogate_cannbench._V2GateExecution:
        return surrogate_cannbench._V2GateExecution(
            execution_id="exec-busy",
            report={
                "status": "BLOCKED_ENV",
                "gates": [
                    {
                        "gateId": surrogate_cannbench.AGENT_GATES_DEVICE_FAST_GATE,
                        "status": "BLOCKED_ENV",
                        "failures": ["BLOCKED_EXIT_CODE: 75"],
                        "observations": {"errorCategory": "occupancy"},
                    }
                ],
            },
            verification={"satisfied": False, "failures": ["BLOCKED_EXIT_CODE: 75"]},
            device_id=str(kwargs.get("device_id") or ""),
        )

    monkeypatch.setattr(surrogate_cannbench, "_run_v2_gate", _fake_v2_always_busy)
    with pytest.raises(RuntimeError, match="did not verify"):
        surrogate_cannbench._submit_fast_gate(
            agent_gates_bin="agent-gates",
            root=tmp_path,
            workspace="tilefoundry",
            request_id="req-exhaust",
            request_relative_path="req.json",
            request_path=req_path,
            response_path=resp_path,
            request_sha256=req_sha,
            mode="batch",
            timeout_sec=15.0,
            device_id="1",
            device_pool=("1", "2"),
        )
    assert sleeps == [4.0, 6.0, 8.0]


def test_order_candidates_by_idle_preflight_ranks_quiet_shared_before_heavy_and_locked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quiet shared cards rank by ascending HBM/proc count ahead of heavy and locked cards."""
    monkeypatch.setenv("AGENT_GATES_DEVICE_LOCK_ROOT", str(tmp_path / "locks"))
    (tmp_path / "locks").mkdir()
    monkeypatch.setattr(surrogate_cannbench.shutil, "which", lambda _cmd: "/usr/bin/npu-smi")
    monkeypatch.setattr(surrogate_cannbench.time, "sleep", lambda _s: None)

    def _fake_smi(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[1:] == ["info", "-m"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=(
                    "NPU ID  Slot ID  Chip ID  Chip Phy-ID  Chip Name\n"
                    "0       74       0        0            Ascend950PR\n"
                    "1       76       0        1            Ascend950PR\n"
                    "2       84       0        2            Ascend950PR\n"
                ),
                stderr="",
            )
        if cmd[1:] == ["info", "-t", "proc-mem", "-i", "0"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=(
                    "NPU ID                         : 0\n"
                    "Process id:101  Process name:python            Process memory(MB):47000\n"
                ),
                stderr="",
            )
        if cmd[1:] == ["info", "-t", "proc-mem", "-i", "1"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=(
                    "NPU ID                         : 1\n"
                    "Process id:201  Process name:python            Process memory(MB):1084\n"
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=(
                "NPU ID                         : 2\n"
                "Process id:301  Process name:python            Process memory(MB):2000\n"
                "Process id:302  Process name:python            Process memory(MB):1050\n"
                "Process id:303  Process name:python            Process memory(MB):1000\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", _fake_smi)
    assert surrogate_cannbench.order_candidates_by_idle_preflight(
        "0", ("0", "1", "2")
    ) == ("1", "2", "0")


def test_measure_npu_ring_burst_us_rejects_upper_tail_contention_spikes() -> None:
    """High-CV bursts retry up to 2x and take cleaner retry or lower-half envelope median."""

    class _ClockEvent:
        def __init__(self, clock: list[float]) -> None:
            self._clock = clock
            self._t = 0.0

        def record(self) -> None:
            self._t = self._clock[0]

        def elapsed_time(self, other: _ClockEvent) -> float:
            return (other._t - self._t) / 1000.0

    clock = [0.0]

    def _clean_fn(_x: object) -> None:
        clock[0] += 3.0

    clean_us, clean_k = surrogate_cannbench.measure_npu_ring_burst_us(
        _clean_fn,
        object(),
        warmup=1,
        repeat=3,
        t_hw_us=2.0,
        first_probe_us=3.0,
        event_factory=lambda: _ClockEvent(clock),
        synchronize_fn=lambda: None,
        clone_fn=lambda o: o,
    )
    assert clean_k >= 1
    assert clean_us == pytest.approx(3.0)

    k = surrogate_cannbench.compute_ring_burst_k(0, t_hw_us=2.0, first_probe_us=10.0)
    assert k >= 1
    warmup_calls = 1 + min(k, 1)

    step_idx = [0]
    pattern_per_call = [
        *([10.0] * warmup_calls),
        *([5.0] * k),
        *([6.0] * k),
        *([25.0] * k),
        *([5.0] * (3 * k)),
    ]

    def _retry_clean_fn(_x: object) -> None:
        val = pattern_per_call[min(step_idx[0], len(pattern_per_call) - 1)]
        step_idx[0] += 1
        clock[0] += val

    retried_us, retried_k = surrogate_cannbench.measure_npu_ring_burst_us(
        _retry_clean_fn,
        object(),
        warmup=1,
        repeat=3,
        t_hw_us=2.0,
        first_probe_us=10.0,
        event_factory=lambda: _ClockEvent(clock),
        synchronize_fn=lambda: None,
        clone_fn=lambda o: o,
    )
    assert retried_k == k
    assert retried_us == pytest.approx(5.0)

    step_idx_3 = [0]
    noisy_pattern = [
        *([10.0] * warmup_calls),
        *([5.0] * k), *([6.0] * k), *([30.0] * k),
        *([5.5] * k), *([6.5] * k), *([35.0] * k),
        *([5.2] * k), *([6.2] * k), *([40.0] * k),
    ]

    def _all_noisy_fn(_x: object) -> None:
        val = noisy_pattern[min(step_idx_3[0], len(noisy_pattern) - 1)]
        step_idx_3[0] += 1
        clock[0] += val

    env_us, _ = surrogate_cannbench.measure_npu_ring_burst_us(
        _all_noisy_fn,
        object(),
        warmup=1,
        repeat=3,
        t_hw_us=2.0,
        first_probe_us=10.0,
        event_factory=lambda: _ClockEvent(clock),
        synchronize_fn=lambda: None,
        clone_fn=lambda o: o,
    )
    assert env_us < 6.0
